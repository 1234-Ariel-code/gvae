from __future__ import annotations
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name != "app" else Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse, os
import pandas as pd
from gvae.data import load_demo_dataset
from gvae.train import train_models_for_config
from gvae.downstream import run_binary_prediction, run_quant_prediction
from gvae.xai import simple_snp_importance
from gvae.biology import run_mock_biology
from gvae.plotting import barplot_metric_table, plot_gene_counts
from gvae.utils import ensure_dir, save_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["binary", "quantitative"], required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--latent_dim", type=int, default=16)
    ap.add_argument("--num_samples", type=int, default=10)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--top_k", type=int, default=100)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    X, y, snp_ids, meta = load_demo_dataset(args.data_dir)
    metrics_df, artifacts = train_models_for_config(X, args.latent_dim, args.num_samples, args.depth, [1.0, 4.0], args.epochs, args.batch_size)
    metrics_df.to_csv(os.path.join(args.out_dir, "representation_metrics.csv"), index=False)

    prediction_rows = []
    summary = {}
    for model_name, art in artifacts.items():
        Z = art["features"]
        pred = run_binary_prediction(Z, y) if args.task == "binary" else run_quant_prediction(Z, y)
        prediction_rows.append({"model": model_name, **pred})
        snp_df = simple_snp_importance(X, Z, snp_ids=snp_ids, top_k=args.top_k)
        snp_out_dir = ensure_dir(os.path.join(args.out_dir, model_name, "xai"))
        snp_df.to_csv(os.path.join(snp_out_dir, "top_snps_per_latent.csv"), index=False)
        gene_df, path_df = run_mock_biology(snp_df, ensure_dir(os.path.join(args.out_dir, model_name, "biology")))
        plot_gene_counts(gene_df, os.path.join(args.out_dir, model_name, "biology", "gene_counts.png"))
        summary[model_name] = {"n_top_snps": int(len(snp_df)), "n_genes": int(gene_df["GENE"].nunique()), "n_pathways": int(len(path_df))}

    pred_df = pd.DataFrame(prediction_rows)
    pred_df.to_csv(os.path.join(args.out_dir, "prediction_metrics.csv"), index=False)

    for metric in ["R2", "MSE", "Robustness"]:
        if metric in metrics_df.columns:
            barplot_metric_table(metrics_df, metric, os.path.join(args.out_dir, f"{metric}.png"), f"{metric} across models")
    if args.task == "binary":
        for metric in ["AUC", "Accuracy"]:
            barplot_metric_table(pred_df, metric, os.path.join(args.out_dir, f"{metric}.png"), f"{metric} across models")
    else:
        for metric in ["R2", "RMSE", "MAE", "Correlation"]:
            if metric in pred_df.columns:
                barplot_metric_table(pred_df, metric, os.path.join(args.out_dir, f"prediction_{metric}.png"), f"{metric} across models")

    save_json({"task": args.task, "data_meta": meta, "analysis_summary": summary}, os.path.join(args.out_dir, "summary.json"))
    print(f"Demo pipeline finished. Outputs in {args.out_dir}")

if __name__ == "__main__":
    main()
