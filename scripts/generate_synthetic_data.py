from __future__ import annotations
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name != "app" else Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from gvae.data import simulate_genotype_data, write_demo_dataset

def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic gVAE demo dataset, including PLINK-compatible exports.")
    ap.add_argument("--task", choices=["binary", "quantitative"], required=True)
    ap.add_argument("--n_samples", type=int, default=600)
    ap.add_argument("--n_snps", type=int, default=2000)
    ap.add_argument("--n_causal", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset_name", type=str, default="demo")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--no_plink", action="store_true", help="Skip .bed/.bim/.fam export")
    ap.add_argument("--no_assoc", action="store_true", help="Skip GWAS-style .assoc/.qassoc export")
    ap.add_argument("--no_tped", action="store_true", help="Skip .tped/.tfam export")
    args = ap.parse_args()

    X, y, snp_ids, causal_idx = simulate_genotype_data(args.n_samples, args.n_snps, args.n_causal, args.task, args.seed)
    write_demo_dataset(
        args.out_dir,
        X,
        y,
        snp_ids,
        args.task,
        causal_idx,
        dataset_name=args.dataset_name,
        export_plink=not args.no_plink,
        export_assoc=not args.no_assoc,
        export_tped=not args.no_tped,
    )
    print(f"Wrote synthetic dataset bundle to {args.out_dir}")
    print("Included formats: genotypes.csv, phenotype.csv, *_filtered.csv, *_origin.phen, and optional PLINK/GWAS exports.")

if __name__ == "__main__":
    main()
