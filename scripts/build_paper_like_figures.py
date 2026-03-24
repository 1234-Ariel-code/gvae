from __future__ import annotations
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name != "app" else Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse, os, pandas as pd
from gvae.plotting import barplot_metric_table

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    args = ap.parse_args()
    rep = pd.read_csv(os.path.join(args.input_dir, "representation_metrics.csv"))
    for metric in ["R2", "MSE", "Robustness"]:
        if metric in rep.columns:
            barplot_metric_table(rep, metric, os.path.join(args.input_dir, f"figure_like_{metric}.png"), f"{metric} across models")
    pred = pd.read_csv(os.path.join(args.input_dir, "prediction_metrics.csv"))
    for metric in pred.columns:
        if metric != "model":
            barplot_metric_table(pred, metric, os.path.join(args.input_dir, f"figure_like_prediction_{metric}.png"), f"{metric} across models")
    print("Figure-like plots written.")

if __name__ == "__main__":
    main()
