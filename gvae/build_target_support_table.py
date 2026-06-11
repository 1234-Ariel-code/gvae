#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Build a disease-specific target-support gene table from:
  1) Open Targets GraphQL API
  2) Local DepMap files:
       - CRISPRGeneEffectUncorrected.csv
       - Model.csv

This version is for the exact DepMap files provided by the user.

Expected inputs
---------------
--out_file            Output TSV path
--depmap_gene_file    Path to CRISPRGeneEffectUncorrected.csv
--depmap_model_file   Path to Model.csv

Output
------
A TSV with columns such as:
  Disease
  GENE
  OpenTargets_supported
  OpenTargets_score
  OpenTargets_disease_id
  OpenTargets_disease_name
  DepMap_supported
  DepMap_score
  DepMap_min_score
  DepMap_n_models
  DepMap_lineage_match_type
  TargetSupportScore
  SupportTier
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests


OPEN_TARGETS_GRAPHQL = "https://api.platform.opentargets.org/api/v4/graphql"


# ---------------------------------------------------------------------
# Disease map
# ---------------------------------------------------------------------
# Columns:
#   Disease                  short code used in your project
#   Disease_label            label used to search Open Targets
#   Cancer_like              whether DepMap support should be attempted
#   Lineage_patterns         regex patterns for OncotreeLineage
#   Primary_disease_patterns regex patterns for OncotreePrimaryDisease
DISEASE_MAP = [
    ("ALZ", "Alzheimer's disease", False, None, None),
    ("ASD", "autistic disorder", False, None, None),
    ("BD",  "bipolar disorder", False, None, None),
    ("BMI", "obesity", False, None, None),
    ("BRC", "breast cancer", True,  r"breast", r"breast"),
    ("CAD", "coronary artery disease", False, None, None),
    ("CD",  "ulcerative colitis", False, None, None),
    ("COL", "colon cancer", True,  r"bowel|large intestine|colon|colorectal", r"colon|colorectal"),
    ("EOS", "Barrett's esophagus", True,  r"esophagus|upper aerodigestive", r"esophagus"),
    ("HDL", "metabolic syndrome X", False, None, None),
    ("HGT", "osteoporosis", False, None, None),
    ("HT",  "hypertension", False, None, None),
    ("LDL", "metabolic syndrome X", False, None, None),
    ("LUN", "lung cancer", True,  r"lung", r"lung"),
    ("PRC", "prostate cancer", True,  r"prostate", r"prostate"),
    ("RA",  "rheumatoid arthritis", False, None, None),
    ("T1D", "type 1 diabetes mellitus", False, None, None),
    ("T2D", "type 2 diabetes mellitus", False, None, None),
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


# ---------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------
def log(msg: str) -> None:
    print(msg, flush=True)


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr, flush=True)


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


# ---------------------------------------------------------------------
# Open Targets helpers
# ---------------------------------------------------------------------
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

    target_name = disease_name.strip().lower()

    def sort_key(hit: Dict[str, Any]) -> Tuple[int, int, int]:
        hit_name = str(hit.get("name", "")).strip().lower()
        return (
            0 if hit_name == target_name else 1,
            abs(len(hit_name) - len(target_name)),
            len(hit_name),
        )

    return sorted(hits, key=sort_key)[0]


def fetch_open_targets_genes(disease_name: str) -> pd.DataFrame:
    hit = find_best_disease_hit(disease_name)
    if hit is None:
        warn(f"No Open Targets disease hit found for: {disease_name}")
        return pd.DataFrame(
            columns=[
                "GENE",
                "OpenTargets_score",
                "OpenTargets_disease_id",
                "OpenTargets_disease_name",
                "OpenTargets_supported",
            ]
        )

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
        return pd.DataFrame(
            columns=[
                "GENE",
                "OpenTargets_score",
                "OpenTargets_disease_id",
                "OpenTargets_disease_name",
                "OpenTargets_supported",
            ]
        )

    df = pd.DataFrame(rows)
    df = (
        df.groupby(["GENE", "OpenTargets_disease_id", "OpenTargets_disease_name"], as_index=False)[
            "OpenTargets_score"
        ]
        .max()
    )
    df["OpenTargets_supported"] = 1
    return df


# ---------------------------------------------------------------------
# DepMap helpers
# ---------------------------------------------------------------------
GENE_COL_PATTERN = re.compile(r"^(?P<symbol>.+?)\s+\((?P<entrez>\d+)\)$")


def normalize_gene_symbol_from_depmap_column(col: str) -> Optional[str]:
    """
    DepMap gene-effect matrix columns look like:
      KRAS (3845)
      TP53 (7157)

    Returns:
      KRAS, TP53, ...
    """
    if col == "ModelID":
        return None

    m = GENE_COL_PATTERN.match(str(col).strip())
    if m:
        return m.group("symbol").upper().strip()

    # Fallback: allow plain symbol columns if ever encountered
    col = str(col).strip()
    if col and col.isupper():
        return col

    return None


def load_depmap_gene_effect(depmap_gene_file: str) -> pd.DataFrame:
    if not depmap_gene_file or not os.path.exists(depmap_gene_file):
        raise FileNotFoundError(f"DepMap gene-effect file not found: {depmap_gene_file}")

    log(f"Loading DepMap gene-effect file: {depmap_gene_file}")
    df = pd.read_csv(depmap_gene_file)

    if "ModelID" not in df.columns:
        raise ValueError("DepMap gene-effect file must contain a 'ModelID' column.")

    keep_cols = ["ModelID"]
    rename_map: Dict[str, str] = {}

    for col in df.columns:
        if col == "ModelID":
            continue
        gene = normalize_gene_symbol_from_depmap_column(col)
        if gene is not None:
            keep_cols.append(col)
            rename_map[col] = gene

    if len(keep_cols) <= 1:
        raise ValueError("No gene-effect columns could be parsed from the DepMap gene-effect file.")

    df = df[keep_cols].copy()
    df = df.rename(columns=rename_map)
    df["ModelID"] = df["ModelID"].astype(str).str.strip()

    return df


def load_depmap_model_metadata(depmap_model_file: str) -> pd.DataFrame:
    if not depmap_model_file or not os.path.exists(depmap_model_file):
        raise FileNotFoundError(f"DepMap model metadata file not found: {depmap_model_file}")

    log(f"Loading DepMap model metadata file: {depmap_model_file}")
    meta = pd.read_csv(depmap_model_file)

    required = ["ModelID", "OncotreeLineage", "OncotreePrimaryDisease"]
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise ValueError(
            f"DepMap model metadata file is missing required columns: {missing}"
        )

    meta = meta[
        [
            "ModelID",
            "CellLineName",
            "StrippedCellLineName",
            "OncotreeLineage",
            "OncotreePrimaryDisease",
            "OncotreeSubtype",
            "OncotreeCode",
        ]
    ].copy()

    for col in meta.columns:
        meta[col] = meta[col].astype(str).str.strip()

    return meta


def select_models_for_disease(
    meta: pd.DataFrame,
    lineage_pattern: Optional[str],
    primary_disease_pattern: Optional[str],
) -> Tuple[pd.DataFrame, str]:
    """
    Select relevant models for one disease using metadata.

    Preference:
      1) models matching both lineage and primary disease
      2) if none, models matching primary disease only
      3) if none, models matching lineage only
    """
    lineage = meta["OncotreeLineage"].fillna("").str.lower()
    primary = meta["OncotreePrimaryDisease"].fillna("").str.lower()

    lineage_mask = pd.Series(False, index=meta.index)
    primary_mask = pd.Series(False, index=meta.index)

    if lineage_pattern:
        lineage_mask = lineage.str.contains(lineage_pattern, regex=True, na=False)
    if primary_disease_pattern:
        primary_mask = primary.str.contains(primary_disease_pattern, regex=True, na=False)

    both = meta[lineage_mask & primary_mask].copy()
    if not both.empty:
        return both, "lineage+primary"

    if primary_disease_pattern:
        by_primary = meta[primary_mask].copy()
        if not by_primary.empty:
            return by_primary, "primary_only"

    if lineage_pattern:
        by_lineage = meta[lineage_mask].copy()
        if not by_lineage.empty:
            return by_lineage, "lineage_only"

    return meta.iloc[0:0].copy(), "none"


def compute_depmap_support_for_disease(
    disease: str,
    gene_effect_wide: pd.DataFrame,
    model_meta: pd.DataFrame,
    lineage_pattern: Optional[str],
    primary_disease_pattern: Optional[str],
    threshold: float = -0.5,
) -> pd.DataFrame:
    """
    Build gene-level support table for one disease from DepMap.

    A gene is considered DepMap-supported if its minimum gene-effect
    among selected models is <= threshold.
    """
    selected_models, match_type = select_models_for_disease(
        model_meta,
        lineage_pattern=lineage_pattern,
        primary_disease_pattern=primary_disease_pattern,
    )

    if selected_models.empty:
        warn(f"No DepMap models matched for disease {disease}.")
        return pd.DataFrame(
            columns=[
                "Disease",
                "GENE",
                "DepMap_supported",
                "DepMap_score",
                "DepMap_min_score",
                "DepMap_mean_score",
                "DepMap_n_models",
                "DepMap_lineage_match_type",
            ]
        )

    merged = gene_effect_wide.merge(
        selected_models[["ModelID"]],
        on="ModelID",
        how="inner",
    )

    if merged.empty:
        warn(f"No overlapping ModelID entries between gene-effect and metadata for disease {disease}.")
        return pd.DataFrame(
            columns=[
                "Disease",
                "GENE",
                "DepMap_supported",
                "DepMap_score",
                "DepMap_min_score",
                "DepMap_mean_score",
                "DepMap_n_models",
                "DepMap_lineage_match_type",
            ]
        )

    gene_cols = [c for c in merged.columns if c != "ModelID"]

    # Long format: one row per (model, gene)
    long_df = merged.melt(
        id_vars="ModelID",
        value_vars=gene_cols,
        var_name="GENE",
        value_name="dep_score",
    )

    long_df["dep_score"] = pd.to_numeric(long_df["dep_score"], errors="coerce")
    long_df = long_df.dropna(subset=["dep_score"])

    if long_df.empty:
        warn(f"No finite DepMap gene-effect values for disease {disease}.")
        return pd.DataFrame(
            columns=[
                "Disease",
                "GENE",
                "DepMap_supported",
                "DepMap_score",
                "DepMap_min_score",
                "DepMap_mean_score",
                "DepMap_n_models",
                "DepMap_lineage_match_type",
            ]
        )

    agg = (
        long_df.groupby("GENE", as_index=False)
        .agg(
            DepMap_mean_score=("dep_score", "mean"),
            DepMap_min_score=("dep_score", "min"),
            DepMap_n_models=("ModelID", "nunique"),
        )
    )

    agg["Disease"] = disease
    agg["DepMap_supported"] = (agg["DepMap_min_score"] <= threshold).astype(int)
    agg["DepMap_score"] = (-agg["DepMap_min_score"]).clip(lower=0)
    agg["DepMap_lineage_match_type"] = match_type

    agg = agg[
        [
            "Disease",
            "GENE",
            "DepMap_supported",
            "DepMap_score",
            "DepMap_min_score",
            "DepMap_mean_score",
            "DepMap_n_models",
            "DepMap_lineage_match_type",
        ]
    ].copy()

    # Keep only supported genes for downstream use
    agg = agg[agg["DepMap_supported"] == 1].copy()

    return agg


def load_depmap_support(
    depmap_gene_file: str,
    depmap_model_file: str,
    disease_map_df: pd.DataFrame,
    threshold: float = -0.5,
) -> pd.DataFrame:
    if not depmap_gene_file or not depmap_model_file:
        warn("DepMap files not provided. Continuing with Open Targets only.")
        return pd.DataFrame(
            columns=[
                "Disease",
                "GENE",
                "DepMap_supported",
                "DepMap_score",
                "DepMap_min_score",
                "DepMap_mean_score",
                "DepMap_n_models",
                "DepMap_lineage_match_type",
            ]
        )

    if not os.path.exists(depmap_gene_file) or not os.path.exists(depmap_model_file):
        warn("One or both DepMap files not found. Continuing with Open Targets only.")
        return pd.DataFrame(
            columns=[
                "Disease",
                "GENE",
                "DepMap_supported",
                "DepMap_score",
                "DepMap_min_score",
                "DepMap_mean_score",
                "DepMap_n_models",
                "DepMap_lineage_match_type",
            ]
        )

    gene_effect_wide = load_depmap_gene_effect(depmap_gene_file)
    model_meta = load_depmap_model_metadata(depmap_model_file)

    parts: List[pd.DataFrame] = []

    for _, row in disease_map_df.iterrows():
        if not bool(row["Cancer_like"]):
            continue

        disease = row["Disease"]
        lineage_pattern = row["Lineage_patterns"]
        primary_pattern = row["Primary_disease_patterns"]

        log(f"[DepMap] {disease}")
        sub = compute_depmap_support_for_disease(
            disease=disease,
            gene_effect_wide=gene_effect_wide,
            model_meta=model_meta,
            lineage_pattern=lineage_pattern,
            primary_disease_pattern=primary_pattern,
            threshold=threshold,
        )
        if not sub.empty:
            parts.append(sub)

    if not parts:
        warn("No disease-specific DepMap support identified. Continuing with Open Targets only.")
        return pd.DataFrame(
            columns=[
                "Disease",
                "GENE",
                "DepMap_supported",
                "DepMap_score",
                "DepMap_min_score",
                "DepMap_mean_score",
                "DepMap_n_models",
                "DepMap_lineage_match_type",
            ]
        )

    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build target-support gene table from Open Targets + DepMap."
    )
    parser.add_argument("--out_file", required=True, help="Output TSV path.")
    parser.add_argument(
        "--depmap_gene_file",
        default="",
        help="Path to CRISPRGeneEffectUncorrected.csv",
    )
    parser.add_argument(
        "--depmap_model_file",
        default="",
        help="Path to Model.csv",
    )
    parser.add_argument(
        "--depmap_threshold",
        type=float,
        default=-0.5,
        help="Gene-effect threshold for DepMap support (default: -0.5).",
    )
    args = parser.parse_args()

    ensure_parent_dir(args.out_file)

    disease_map_df = pd.DataFrame(
        DISEASE_MAP,
        columns=[
            "Disease",
            "Disease_label",
            "Cancer_like",
            "Lineage_patterns",
            "Primary_disease_patterns",
        ],
    )

    # --------------------------------------------------
    # Open Targets support
    # --------------------------------------------------
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

    # --------------------------------------------------
    # DepMap support
    # --------------------------------------------------
    dep_support = load_depmap_support(
        depmap_gene_file=args.depmap_gene_file,
        depmap_model_file=args.depmap_model_file,
        disease_map_df=disease_map_df,
        threshold=args.depmap_threshold,
    )

    # --------------------------------------------------
    # Merge
    # --------------------------------------------------
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
        "DepMap_mean_score": float("nan"),
        "DepMap_n_models": 0,
        "DepMap_lineage_match_type": "",
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

    summary_file = os.path.join(
        os.path.dirname(os.path.abspath(args.out_file)),
        "target_support_gene_table_summary.tsv",
    )
    summary.to_csv(summary_file, sep="\t", index=False)

    log(f"[OK] Wrote target support table: {args.out_file}")
    log(f"[OK] Wrote summary: {summary_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
