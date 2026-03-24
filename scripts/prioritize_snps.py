from __future__ import annotations
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name != "app" else Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse, os
from gvae.data import load_demo_dataset
from gvae.train import train_models_for_config
from gvae.xai import simple_snp_importance

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--latent_dim", type=int, default=16)
    ap.add_argument("--num_samples", type=int, default=10)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--top_k", type=int, default=100)
    args = ap.parse_args()
    X, y, snp_ids, meta = load_demo_dataset(args.data_dir)
    _, artifacts = train_models_for_config(X, args.latent_dim, args.num_samples, args.depth, [1.0, 4.0], args.epochs, args.batch_size)
    os.makedirs(args.out_dir, exist_ok=True)
    for model_name, art in artifacts.items():
        snp_df = simple_snp_importance(X, art["features"], snp_ids, top_k=args.top_k)
        snp_df.to_csv(os.path.join(args.out_dir, f"{model_name}_top_snps_per_latent.csv"), index=False)
    print(f"Wrote SNP prioritization tables to {args.out_dir}")

if __name__ == "__main__":
    main()
