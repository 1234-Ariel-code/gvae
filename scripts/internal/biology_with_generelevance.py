#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

try:
    from gseapy import enrichr as gseapy_enrichr
    GSEAPY_AVAILABLE = True
except Exception:
    GSEAPY_AVAILABLE = False


def detect_gene_col(df: pd.DataFrame) -> str:
    for c in ["GENE", "Gene", "geneSymbol", "gene", "SYMBOL", "Symbol"]:
        if c in df.columns:
            return c
    raise ValueError(f"Could not detect gene column. Columns: {list(df.columns)}")


def detect_snp_col(df: pd.DataFrame) -> str:
    for c in ["SNP", "SNP_ID", "rsid", "RSID", "snp"]:
        if c in df.columns:
            return c
    raise ValueError(f"Could not detect SNP column. Columns: {list(df.columns)}")


def parse_shap_file(path: Path, latent_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if latent_col not in df.columns or "SNP_ID" not in df.columns:
        raise ValueError(f"Missing required columns in {path.name}")
    return df[[latent_col, "SNP_ID"]].copy()


def load_s2g(path: str) -> tuple[pd.DataFrame, str, str]:
    df = pd.read_csv(path, sep="\t")
    snp_col = detect_snp_col(df)
    gene_col = detect_gene_col(df)
    df[snp_col] = df[snp_col].astype(str).str.strip()
    df[gene_col] = df[gene_col].astype(str).str.strip()
    return df, snp_col, gene_col


def build_gene_table(shap_df: pd.DataFrame, s2g_df: pd.DataFrame, s2g_snp_col: str, s2g_gene_col: str, latent_col: str, gene_col_out: str) -> pd.DataFrame:
    out = shap_df.merge(s2g_df[[s2g_snp_col, s2g_gene_col]], left_on="SNP_ID", right_on=s2g_snp_col, how="left")
    out = out.dropna(subset=[s2g_gene_col]).rename(columns={s2g_gene_col: gene_col_out})
    out[gene_col_out] = out[gene_col_out].astype(str).str.replace(r"\s+", "", regex=True)
    out[gene_col_out] = out[gene_col_out].str.split(r"[;,|]")
    out = out.explode(gene_col_out)
    return out[[latent_col, gene_col_out, "SNP_ID"]].dropna().drop_duplicates()


def compute_gene_relevance(gene_df: pd.DataFrame, disgenet_tsv: str, disease_name: str, latent_col: str, gene_col: str) -> pd.DataFrame:
    dis = pd.read_csv(disgenet_tsv, sep="\t")
    cols = {c.lower(): c for c in dis.columns}
    gene_c = cols.get("genesymbol") or cols.get("gene") or cols.get("symbol")
    disease_c = cols.get("doid_name") or cols.get("diseasename") or cols.get("disease_name") or cols.get("disease")
    score_c = cols.get("score_max") or cols.get("score") or cols.get("score_mean")
    if not gene_c or not disease_c:
        raise ValueError("Could not detect gene or disease columns in DisGeNET TSV.")
    dis[gene_c] = dis[gene_c].astype(str).str.upper().str.strip()
    dis[disease_c] = dis[disease_c].astype(str).str.lower().str.strip()
    if score_c:
        dis[score_c] = pd.to_numeric(dis[score_c], errors="coerce")
    else:
        dis["_score"] = 1.0
        score_c = "_score"
    filt = dis[dis[disease_c].str.contains(str(disease_name).lower(), na=False)].copy()
    if filt.empty:
        return pd.DataFrame(columns=["Gene", "Group", "DisGeNET_score_max"])
    gene_df2 = gene_df.copy()
    gene_df2[gene_col] = gene_df2[gene_col].astype(str).str.upper().str.strip()
    out = gene_df2.merge(filt[[gene_c, score_c]], left_on=gene_col, right_on=gene_c, how="inner")
    summ = out.groupby([gene_col, latent_col], as_index=False)[score_c].max().rename(columns={gene_col: "Gene", latent_col: "Group", score_c: "DisGeNET_score_max"})
    return summ


def enrich_by_latent(gene_df: pd.DataFrame, gene_sets: list[str], latent_col: str, gene_col: str, out_dir: str):
    if not GSEAPY_AVAILABLE:
        return {}
    os.makedirs(out_dir, exist_ok=True)
    outputs = {}
    for grp, sub in gene_df.groupby(latent_col):
        genes = pd.unique(sub[gene_col].astype(str))[:200].tolist()
        if len(genes) < 3:
            continue
        all_results = []
        for gs in gene_sets:
            try:
                enr = gseapy_enrichr(gene_list=genes, gene_sets=gs, organism="Human", outdir=None, cutoff=1.0)
                if enr.results is None or enr.results.empty:
                    continue
                tmp = enr.results.copy()
                tmp["Library"] = gs
                tmp["Group"] = grp
                all_results.append(tmp)
            except Exception:
                continue
        if all_results:
            df = pd.concat(all_results, ignore_index=True)
            df.to_csv(os.path.join(out_dir, f"enrichr_all_{grp}.csv"), index=False)
            outputs[str(grp)] = os.path.join(out_dir, f"enrichr_all_{grp}.csv")
    return outputs


def write_manifest(path: str, payload: dict):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def run_single(args):
    out_root = Path(args.output_dir) / args.disease / f"LD{args.LD}_NS{args.NS}_L{args.L}_K{args.K}"
    if args.sample_id:
        out_root = out_root / args.sample_id
    out_root.mkdir(parents=True, exist_ok=True)

    if args.sample_id:
        shap_file = Path(args.base_dir) / "outputs" / f"{args.disease}_LD{args.LD}_NS{args.NS}_L{args.L}_K{args.K}_{args.sample_id}_top_snps_per_latent.csv"
    else:
        shap_file = Path(args.base_dir) / "outputs" / f"{args.disease}_LD{args.LD}_NS{args.NS}_L{args.L}_K{args.K}_top_snps_per_latent.csv"
    shap_df = parse_shap_file(shap_file, args.latent_col)
    s2g_df, s2g_snp_col, s2g_gene_col = load_s2g(args.s2g_path)
    gene_df = build_gene_table(shap_df, s2g_df, s2g_snp_col, s2g_gene_col, args.latent_col, args.gene_col_out)
    gene_df.to_csv(out_root / "latent_gene_table.csv", index=False)

    manifest = {"gene_table": str(out_root / "latent_gene_table.csv")}

    enrich_dir = out_root / "enrichment"
    if args.mode == "enrichr":
        outputs = enrich_by_latent(gene_df, args.gene_sets, args.latent_col, args.gene_col_out, str(enrich_dir))
        manifest["enrichment"] = outputs

    if args.run_gene_analysis:
        gene_rel_dir = out_root / "gene_relevance"
        gene_rel_dir.mkdir(exist_ok=True)
        gene_rel = compute_gene_relevance(gene_df, args.disgenet_tsv, args.disgenet_disease_name or args.disease, args.latent_col, args.gene_col_out)
        gene_rel.to_csv(gene_rel_dir / "gene_relevance_summary_per_gene_group.csv", index=False)
        manifest["gene_relevance"] = str(gene_rel_dir / "gene_relevance_summary_per_gene_group.csv")

    write_manifest(out_root / "outputs_manifest.json", manifest)
    print(f"[DONE] Output root: {out_root}")


def run_aggregated(args):
    pattern = Path(args.base_dir) / "outputs"
    files = sorted(pattern.glob(f"{args.disease}_LD{args.LD}_NS{args.NS}_L{args.L}_K{args.K}_S*_top_snps_per_latent.csv"))
    if not files:
        print("[WARN] No per-sample SHAP files found for aggregation")
        return
    out_root = Path(args.output_dir) / args.disease / f"LD{args.LD}_NS{args.NS}_L{args.L}_K{args.K}" / "ALL_SAMPLES"
    out_root.mkdir(parents=True, exist_ok=True)
    s2g_df, s2g_snp_col, s2g_gene_col = load_s2g(args.s2g_path)
    rows = []
    for fp in files:
        m = re.search(r"_S(\d+)_top_snps_per_latent\.csv$", fp.name)
        sample = f"S{m.group(1)}" if m else "S?"
        shap_df = parse_shap_file(fp, args.latent_col)
        gene_df = build_gene_table(shap_df, s2g_df, s2g_snp_col, s2g_gene_col, args.latent_col, args.gene_col_out)
        gene_df["Sample"] = sample
        rows.append(gene_df)
    long_df = pd.concat(rows, ignore_index=True)
    long_df.to_csv(out_root / "latent_gene_table.ALL_SAMPLES.long.csv", index=False)
    print(f"[DONE] Aggregated all samples to {out_root}")


def parse_args():
    p = argparse.ArgumentParser(description="SHAP -> SNP -> Gene -> pathway and DisGeNET relevance pipeline")
    p.add_argument("--base_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--disease", required=True)
    p.add_argument("--LD", required=True, type=int)
    p.add_argument("--NS", required=True, type=int)
    p.add_argument("--L", required=True, type=int)
    p.add_argument("--K", required=True, type=int)
    p.add_argument("--sample_id", default=None)
    p.add_argument("--s2g_path", required=True)
    p.add_argument("--latent_col", default="Latent_Dim")
    p.add_argument("--gene_col_out", default="GENE")
    p.add_argument("--mode", default="enrichr", choices=["enrichr", "gmt"])
    p.add_argument("--gene_sets", nargs="+", default=["Reactome_2022", "KEGG_2021_Human", "GO_Biological_Process_2023"])
    p.add_argument("--run_gene_analysis", action="store_true")
    p.add_argument("--disgenet_tsv", default=None)
    p.add_argument("--disgenet_disease_name", default=None)
    p.add_argument("--run_aggregated_samples", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.run_aggregated_samples:
        run_aggregated(args)
    else:
        run_single(args)
