from __future__ import annotations
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name != "app" else Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tempfile
import pandas as pd
import streamlit as st
from gvae.data import simulate_genotype_data
from gvae.train import train_models_for_config
from gvae.downstream import run_binary_prediction, run_quant_prediction
from gvae.xai import simple_snp_importance
from gvae.biology import run_mock_biology

st.set_page_config(page_title="gVAE Explorer", layout="wide")
st.title("gVAE interactive explorer")
st.caption("Synthetic-data exploration for binary and quantitative genomic cohorts.")

with st.sidebar:
    task = st.selectbox("Cohort type", ["binary", "quantitative"])
    n_samples = st.slider("Sample size", 200, 2000, 600, step=100)
    n_snps = st.slider("Number of SNPs", 500, 5000, 2000, step=250)
    n_causal = st.slider("Number of causal SNPs", 10, 200, 40, step=10)
    latent_dim = st.selectbox("Latent dimension", [8, 16, 32, 64], index=1)
    num_samples = st.selectbox("Posterior samples K for gVAE", [5, 10, 20], index=1)
    depth = st.selectbox("Network depth", [2, 3, 4], index=0)
    epochs = st.slider("Training epochs", 3, 20, 8)
    run = st.button("Run analysis")

if run:
    with st.spinner("Generating synthetic cohort and training models..."):
        X, y, snp_ids, causal_idx = simulate_genotype_data(n_samples, n_snps, n_causal, task, seed=42)
        rep_df, artifacts = train_models_for_config(X, latent_dim, num_samples, depth, [1.0, 4.0], epochs, 64)
        pred_rows, xai_rows = [], []
        for model_name, art in artifacts.items():
            pred = run_binary_prediction(art["features"], y) if task == "binary" else run_quant_prediction(art["features"], y)
            pred_rows.append({"model": model_name, **pred})
            snp_df = simple_snp_importance(X, art["features"], snp_ids, top_k=50)
            gene_df, path_df = run_mock_biology(snp_df, tempfile.mkdtemp())
            xai_rows.append({"model": model_name, "top_snps": len(snp_df), "unique_genes": gene_df["GENE"].nunique(), "mock_pathways": len(path_df)})
    st.subheader("Representation metrics")
    st.dataframe(rep_df, use_container_width=True)
    pred_df = pd.DataFrame(pred_rows)
    st.subheader("Downstream prediction")
    st.dataframe(pred_df, use_container_width=True)
    st.subheader("Interpretability summary")
    st.dataframe(pd.DataFrame(xai_rows), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        cols = [c for c in ["R2", "MSE", "Robustness"] if c in rep_df.columns]
        st.bar_chart(rep_df.set_index("model")[cols])
    with c2:
        cols = [c for c in ["AUC", "Accuracy", "R2", "RMSE", "MAE", "Correlation"] if c in pred_df.columns]
        st.bar_chart(pred_df.set_index("model")[cols])
else:
    st.info("Choose settings in the sidebar and click 'Run analysis'.")
