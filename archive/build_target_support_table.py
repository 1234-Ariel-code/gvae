#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a disease-specific target-support gene table from:
  1) Open Targets GraphQL API
  2) Optional local DepMap dependency export

Output:
  target_support_gene_table.tsv

This table can then be used by the R pipeline as the drug/target
support reference gene set.

Expected behavior:
- If a local DepMap file exists, merge its support into the table.
- If DepMap is missing, Open Targets alone is still enough to build
  a useful first-pass target-support table.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

OPEN_TARGETS_GRAPHQL = "https://api.platform.opentargets.org/api/v4/graphql"

DISEASE_MAP = [
    ("ALZ", "Alzheimer's disease", False, None),
    ("ASD", "autistic disorder", False, None),
    ("BD",  "bipolar disorder", False, None),
    ("BMI", "obesity", False, None),
    ("BRC", "breast cancer", True, "breast"),
    ("CAD", "coronary artery disease", False, None),
    ("CD",  "ulcerative colitis", False, None),
    ("COL", "colon cancer", True, "colorectal|colon"),
    ("EOS", "Barrett's esophagus", True, "esophagus|upper_aerodigestive"),
    ("HDL", "metabolic syndrome X", False, None),
    ("HGT", "osteoporosis", False, None),
    ("HT",  "hypertension", False, None),
    ("LDL", "metabolic syndrome X", False, None),
    ("LUN", "lung cancer", True, "lung"),
    ("PRC", "prostate cancer", True, "prostate"),
    ("RA",  "rheumatoid arthritis", False, None),
    ("T1D", "type 1 diabetes mellitus", False, None),
    ("T2D", "type 2 diabetes mellitus", False, None),
]

SEARCH_DISEASE_QUERY = """
query SearchDisease($queryString: String!) {
  search(queryString: $queryString, entityNames: ["disease"], page: {index: 0, size: 10}) {
    hits {
      id
      name
      entity
    }
  }
}
"""

ASSOCIATIONS_QUERY = """
query DiseaseAssociations($diseaseId: String!, $pageIndex: Int!, $pageSize: Int!) {
  disease(efoId: $diseaseId) {
    id
    name
    associatedTargets(page: {index: $pageIndex, size: $pageSize}) {
      count
      rows {
        score
        target {
          id
          approvedSymbol
          approvedName
        }
      }
    }
  }
}
"""


def log(msg: str) -> None:
    print(msg, flush=True)


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr, flush=True)


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def post_graphql(query: str, variables: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
    response = requests.post(
        OPEN_TARGETS_GRAPHQL,
        json={"query": query, "variables": variables},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if "errors" in payload:
        raise RuntimeError(json.dumps(payload["errors"], indent=2))
    return payload.get("data", {})


def find_best_disease_hit(disease_name: str) -> Optional[Dict[str, Any]]:
    data = post_graphql(SEARCH_DISEASE_QUERY, {"queryString": disease_name})
    hits = data.get("search", {}).get("hits", [])

    if not hits:
        return None

    def sort_key(hit: Dict[str, Any]) -> Any:
        hit_name = str(hit.get("name", "")).strip().lower()
        target_name = disease_name.strip().lower()
        return (
            0 if hit_name == target_name else 1,
            abs(len(hit_name) - len(target_name)),
            len(hit_name),
        )

    hits_sorted = sorted(hits, key=sort_key)
    return hits_sorted[0]


def fetch_open_targets_genes(disease_name: str) -> pd.DataFrame:
    hit = find_best_disease_hit(disease_name)
    if hit is None:
        warn(f"No Open Targets disease hit found for: {disease_name}")
        return pd.DataFrame(columns=["GENE", "OpenTargets_score", "OpenTargets_disease_id", "OpenTargets_disease_name"])

    disease_id = hit["id"]
    matched_name = hit.get("name", disease_name)

    rows: List[Dict[str, Any]] = []
    page_index = 0
    page_size = 500

    while True:
      data = post_graphql(
          ASSOCIATIONS_QUERY,
          {"diseaseId": disease_id, "pageIndex": page_index, "pageSize": page_size},
      )

      disease_obj = data.get("disease")
      if not disease_obj:
          break

      assoc = disease_obj.get("associatedTargets", {})
      chunk = assoc.get("rows", [])
      if not chunk:
          break

      for row in chunk:
          target = row.get("target") or {}
          gene = target.get("approvedSymbol")
          score = row.get("score")
          if gene:
              rows.append(
                  {
                      "GENE": str(gene).upper().strip(),
                      "OpenTargets_score": float(score) if score is not None else 0.0,
                      "OpenTargets_disease_id": disease_id,
                      "OpenTargets_disease_name": matched_name,
                  }
              )

      total = assoc.get("count", 0)
      page_index += 1
      if page_index * page_size >= total:
          break

    if not rows:
        warn(f"No associated targets returned for disease: {disease_name} ({disease_id})")
        return pd.DataFrame(columns=["GENE", "OpenTargets_score", "OpenTargets_disease_id", "OpenTargets_disease_name"])

    df = pd.DataFrame(rows)
    df = (
        df.groupby(["GENE", "OpenTargets_disease_id", "OpenTargets_disease_name"], as_index=False)["OpenTargets_score"]
        .max()
    )
    df["OpenTargets_supported"] = 1
    return df


def pick_column(columns: List[str], options: List[str]) -> Optional[str]:
    for opt in options:
        if opt in columns:
            return opt
    return None


def load_depmap(depmap_file: str, disease_map_df: pd.DataFrame, threshold: float = -0.5) -> pd.DataFrame:
    if not depmap_file or not os.path.exists(depmap_file):
        warn("No DepMap file found. Continuing with Open Targets only.")
        return pd.DataFrame(columns=["Disease", "GENE", "DepMap_supported", "DepMap_score", "DepMap_min_score"])

    log(f"Loading DepMap file: {depmap_file}")
    dep = pd.read_csv(depmap_file)
    dep.columns = [str(c).strip().lower() for c in dep.columns]

    gene_col = pick_column(dep.columns.tolist(), ["gene", "gene_symbol", "symbol"])
    lineage_col = pick_column(dep.columns.tolist(), ["lineage", "lineage_1", "primary_disease", "disease", "oncotree_lineage"])
    score_col = pick_column(dep.columns.tolist(), ["dependency_score", "gene_effect", "dep_score", "chronos", "geneeffect"])

    if gene_col is None or score_col is None:
        warn("DepMap file missing required gene/score columns. Ignoring DepMap.")
        return pd.DataFrame(columns=["Disease", "GENE", "DepMap_supported", "DepMap_score", "DepMap_min_score"])

    keep_cols = [gene_col, score_col]
    if lineage_col is not None:
        keep_cols.append(lineage_col)

    dep = dep[keep_cols].copy()
    new_cols = ["GENE", "dep_score"] if lineage_col is None else ["GENE", "dep_score", "lineage"]
    if lineage_col is not None:
        dep.columns = ["GENE", "dep_score", "lineage"] if keep_cols == [gene_col, score_col, lineage_col] else dep.columns
        # safer explicit mapping
        dep = dep.rename(columns={gene_col: "GENE", score_col: "dep_score", lineage_col: "lineage"})
    else:
        dep = dep.rename(columns={gene_col: "GENE", score_col: "dep_score"})
        dep["lineage"] = None

    dep["GENE"] = dep["GENE"].astype(str).str.upper().str.strip()
    dep["dep_score"] = pd.to_numeric(dep["dep_score"], errors="coerce")
    dep["lineage"] = dep["lineage"].astype(str).str.lower()

    dep = dep.dropna(subset=["GENE", "dep_score"])

    out_parts: List[pd.DataFrame] = []

    for _, row in disease_map_df.iterrows():
        disease = row["Disease"]
        is_cancer_like = bool(row["Cancer_like"])
        lineage_pattern = row["DepMap_lineage_keywords"]

        if not is_cancer_like or pd.isna(lineage_pattern) or lineage_pattern is None:
            continue

        sub = dep[dep["lineage"].fillna("").str.contains(str(lineage_pattern), regex=True, na=False)].copy()
        if sub.empty:
            continue

        agg = (
            sub.groupby("GENE", as_index=False)
            .agg(
                DepMap_mean_score=("dep_score", "mean"),
                DepMap_min_score=("dep_score", "min"),
            )
        )

        agg["Disease"] = disease
        agg["DepMap_supported"] = (agg["DepMap_min_score"] <= threshold).astype(int)
        agg["DepMap_score"] = (-agg["DepMap_min_score"]).clip(lower=0)

        agg = agg[agg["DepMap_supported"] == 1][
            ["Disease", "GENE", "DepMap_supported", "DepMap_score", "DepMap_min_score"]
        ]

        if not agg.empty:
            out_parts.append(agg)

    if not out_parts:
        warn("No disease-specific DepMap support identified. Continuing with Open Targets only.")
        return pd.DataFrame(columns=["Disease", "GENE", "DepMap_supported", "DepMap_score", "DepMap_min_score"])

    return pd.concat(out_parts, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build target-support gene table from Open Targets + optional DepMap.")
    parser.add_argument("--out_file", required=True, help="Output TSV path.")
    parser.add_argument("--depmap_file", default="", help="Optional local DepMap CSV file.")
    args = parser.parse_args()

    ensure_parent_dir(args.out_file)

    disease_map_df = pd.DataFrame(
        DISEASE_MAP,
        columns=["Disease", "Disease_label", "Cancer_like", "DepMap_lineage_keywords"],
    )

    # ----------------------------
    # Open Targets
    # ----------------------------
    ot_parts: List[pd.DataFrame] = []

    for _, row in disease_map_df.iterrows():
        disease_code = row["Disease"]
        disease_label = row["Disease_label"]
        log(f"[OpenTargets] {disease_code}: {disease_label}")

        try:
            ot = fetch_open_targets_genes(disease_label)
            if not ot.empty:
                ot["Disease"] = disease_code
                ot_parts.append(
                    ot[
                        [
                            "Disease",
                            "GENE",
                            "OpenTargets_supported",
                            "OpenTargets_score",
                            "OpenTargets_disease_id",
                            "OpenTargets_disease_name",
                        ]
                    ]
                )
        except Exception as exc:
            warn(f"Open Targets failed for {disease_code}: {exc}")

    if ot_parts:
        ot_support = pd.concat(ot_parts, ignore_index=True)
    else:
        ot_support = pd.DataFrame(
            columns=[
                "Disease",
                "GENE",
                "OpenTargets_supported",
                "OpenTargets_score",
                "OpenTargets_disease_id",
                "OpenTargets_disease_name",
            ]
        )

    # ----------------------------
    # DepMap
    # ----------------------------
    dep_support = load_depmap(args.depmap_file, disease_map_df)

    # ----------------------------
    # Merge
    # ----------------------------
    target_support = pd.merge(
        ot_support,
        dep_support,
        on=["Disease", "GENE"],
        how="outer",
    )

    defaults = {
        "OpenTargets_supported": 0,
        "OpenTargets_score": 0.0,
        "OpenTargets_disease_id": "",
        "OpenTargets_disease_name": "",
        "DepMap_supported": 0,
        "DepMap_score": 0.0,
        "DepMap_min_score": float("nan"),
    }

    for col, fill_value in defaults.items():
        if col not in target_support.columns:
            target_support[col] = fill_value
        target_support[col] = target_support[col].fillna(fill_value)

    target_support["TargetSupportScore"] = (
        pd.to_numeric(target_support["OpenTargets_score"], errors="coerce").fillna(0.0)
        + pd.to_numeric(target_support["DepMap_score"], errors="coerce").fillna(0.0)
    )

    target_support["SupportTier"] = "None"
    mask_both = (target_support["OpenTargets_supported"] == 1) & (target_support["DepMap_supported"] == 1)
    mask_ot = (target_support["OpenTargets_supported"] == 1) & (target_support["DepMap_supported"] == 0)
    mask_dep = (target_support["OpenTargets_supported"] == 0) & (target_support["DepMap_supported"] == 1)

    target_support.loc[mask_both, "SupportTier"] = "OpenTargets+DepMap"
    target_support.loc[mask_ot, "SupportTier"] = "OpenTargets only"
    target_support.loc[mask_dep, "SupportTier"] = "DepMap only"

    target_support = target_support.sort_values(
        by=["Disease", "TargetSupportScore", "GENE"],
        ascending=[True, False, True],
    )

    target_support.to_csv(args.out_file, sep="\t", index=False)

    summary = (
        target_support.groupby("Disease", as_index=False)
        .agg(
            n_supported_genes=("GENE", "count"),
            n_ot_only=("SupportTier", lambda s: (s == "OpenTargets only").sum()),
            n_dep_only=("SupportTier", lambda s: (s == "DepMap only").sum()),
            n_both=("SupportTier", lambda s: (s == "OpenTargets+DepMap").sum()),
        )
        .sort_values("Disease")
    )

    summary_file = os.path.join(os.path.dirname(os.path.abspath(args.out_file)), "target_support_gene_table_summary.tsv")
    summary.to_csv(summary_file, sep="\t", index=False)

    log(f"[OK] Wrote target support table: {args.out_file}")
    log(f"[OK] Wrote summary: {summary_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
