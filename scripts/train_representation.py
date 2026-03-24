from __future__ import annotations
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name != "app" else Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse, os, yaml, pandas as pd
from gvae.data import load_demo_dataset
from gvae.train import train_models_for_config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    X, y, snp_ids, meta = load_demo_dataset(cfg["data_dir"])
    os.makedirs(cfg["out_dir"], exist_ok=True)
    rows = []
    for ld in cfg["latent_dims"]:
        for ns in cfg["num_samples_list"]:
            for depth in cfg["num_layers_list"]:
                df, _ = train_models_for_config(X, ld, ns, depth, cfg["beta_values"], cfg["epochs"], cfg["batch_size"])
                df["latent_dim"] = ld
                df["num_samples"] = ns
                df["depth"] = depth
                rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(os.path.join(cfg["out_dir"], "grid_representation_metrics.csv"), index=False)
    print("Wrote grid_representation_metrics.csv")

if __name__ == "__main__":
    main()
