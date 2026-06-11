from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt

def barplot_metric_table(df: pd.DataFrame, metric_col: str, out_path: str, title: str):
    plt.figure(figsize=(8, 4.5))
    plt.bar(df["model"], df[metric_col])
    plt.ylabel(metric_col)
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_gene_counts(gene_df: pd.DataFrame, out_path: str):
    counts = gene_df.groupby("Latent_Dim")["GENE"].nunique().sort_index()
    plt.figure(figsize=(8, 4.5))
    plt.bar(counts.index, counts.values)
    plt.ylabel("Unique genes")
    plt.title("Gene capture by latent variable")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
