#!/usr/bin/env python3
from __future__ import annotations
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name != "app" else Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import argparse
import os
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from gvae.models import build_gvae

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_matrix(csv_path: str) -> np.ndarray:
    return pd.read_csv(csv_path, header=None).fillna(0).to_numpy(dtype=np.float32).T


def read_snp_ids_from_tped(tped_file: str) -> list[str]:
    return pd.read_csv(tped_file, delim_whitespace=True, header=None)[1].astype(str).tolist()


def train_gvae(X: np.ndarray, latent_dim: int, num_samples: int, num_layers: int, epochs: int, batch_size: int):
    model = build_gvae(X.shape[1], latent_dim, num_layers, num_samples)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4))
    Xtr, Xva = train_test_split(X, test_size=0.2, random_state=SEED)
    ds_tr = tf.data.Dataset.from_tensor_slices(Xtr.astype(np.float32)).shuffle(min(len(Xtr), 10000), seed=SEED).batch(batch_size)
    ds_va = tf.data.Dataset.from_tensor_slices(Xva.astype(np.float32)).batch(batch_size)
    model.fit(ds_tr, validation_data=ds_va, epochs=epochs, verbose=2,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)])
    out, zfinal = model(X.astype(np.float32), training=False)[0], model(X.astype(np.float32), training=False)[3]
    return model, out.numpy(), zfinal.numpy()


def safe_shap_values(explainer, X_chunk):
    out = explainer(X_chunk)
    return out.values if hasattr(out, "values") else np.asarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disease", required=True)
    ap.add_argument("--base_path", required=True)
    ap.add_argument("--latent_dim", required=True, type=int)
    ap.add_argument("--num_layers", required=True, type=int)
    ap.add_argument("--num_samples", required=True, type=int)
    ap.add_argument("--tped_file", required=True)
    ap.add_argument("--output_dir", default=".")
    ap.add_argument("--top_k", default=500, type=int)
    ap.add_argument("--epochs", default=20, type=int)
    ap.add_argument("--batch_size", default=32, type=int)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    X = load_matrix(os.path.join(args.base_path, f"{args.disease}_filtered.csv"))
    snp_ids = read_snp_ids_from_tped(args.tped_file)
    model, recon, z = train_gvae(X, args.latent_dim, args.num_samples, args.num_layers, args.epochs, args.batch_size)

    pd.DataFrame(recon).to_csv(os.path.join(args.output_dir, f"rep_{args.disease}_LD{args.latent_dim}_NS{args.num_samples}_L{args.num_layers}.csv"), index=False)

    bg_idx = np.random.choice(len(X), size=min(100, len(X)), replace=False)
    background = X[bg_idx]
    records = []
    for j in range(z.shape[1]):
        y = z[:, j]
        lr = LinearRegression().fit(X, y)
        explainer = shap.Explainer(lr, background)
        sv = safe_shap_values(explainer, X)
        mean_abs = np.abs(sv).mean(axis=0)
        top_idx = np.argsort(mean_abs)[-min(args.top_k, len(mean_abs)):][::-1]
        for ii in top_idx:
            records.append({"Latent_Dim": f"LD_{j}", "SNP_ID": snp_ids[ii] if ii < len(snp_ids) else f"SNP_{ii}", "SHAP_Importance": float(mean_abs[ii])})
    out = pd.DataFrame(records)
    out.to_csv(os.path.join(args.output_dir, f"{args.disease}_LD{args.latent_dim}_NS{args.num_samples}_L{args.num_layers}_K{args.top_k}_top_snps_per_latent.csv"), index=False)
    print("[DONE] Wrote SHAP-ranked SNP table")


if __name__ == "__main__":
    main()
