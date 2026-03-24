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
import tensorflow as tf
from bed_reader import open_bed
from sklearn.model_selection import train_test_split

from gvae.models import build_baseline_vae, build_beta_vae, build_gvae
from gvae.metrics import reconstruction_metrics, latent_drift, robustness_from_latent_drift

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_bed_as_float32(bed_prefix: str, missing: str = "mean") -> np.ndarray:
    bed_path = bed_prefix + ".bed"
    if not os.path.exists(bed_path):
        raise FileNotFoundError(f"Missing bed file: {bed_path}")
    X = open_bed(bed_path, count_A1=True).read().astype(np.float32)
    if np.any(np.isnan(X)):
        if missing == "mean":
            col_means = np.nanmean(X, axis=0)
            col_means = np.nan_to_num(col_means, nan=0.0).astype(np.float32)
            inds = np.where(np.isnan(X))
            X[inds] = col_means[inds[1]]
        else:
            X = np.nan_to_num(X, nan=0.0)
    return X


def random_feature_subset(X: np.ndarray, downsample_d: int | None, seed: int = 42) -> np.ndarray:
    if downsample_d is None or downsample_d >= X.shape[1]:
        return X
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(X.shape[1], size=int(downsample_d), replace=False))
    return X[:, idx]


def make_dataset(X: np.ndarray, batch_size: int, shuffle: bool = True):
    ds = tf.data.Dataset.from_tensor_slices(X.astype(np.float32))
    if shuffle:
        ds = ds.shuffle(min(len(X), 10000), seed=SEED)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def extract_features(model, X, is_gvae: bool):
    if is_gvae:
        _, _, _, z = model(X.astype(np.float32), training=False)
        return z.numpy()
    mu, _ = model.encode(X.astype(np.float32))
    return mu.numpy()


def compute_robustness(model, X, is_gvae: bool):
    noise = np.random.normal(0, 0.05, size=X.shape).astype(np.float32)
    Xp = np.clip(X + noise, 0, 2)
    Z = extract_features(model, X, is_gvae)
    Zp = extract_features(model, Xp, is_gvae)
    drift = latent_drift(Z, Zp)
    return robustness_from_latent_drift(drift)


def train_one(model, X_train, X_val, epochs: int, batch_size: int, weight_path: str):
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(weight_path, save_weights_only=True, save_best_only=True, monitor="val_loss", verbose=1),
    ]
    model.fit(
        make_dataset(X_train, batch_size, shuffle=True),
        validation_data=make_dataset(X_val, batch_size, shuffle=False),
        epochs=epochs,
        verbose=2,
        callbacks=callbacks,
    )
    if os.path.exists(weight_path):
        model.load_weights(weight_path)
    return model


def main():
    ap = argparse.ArgumentParser(description="Train gVAE, baseline VAE, and beta-VAE directly from a PLINK BED prefix.")
    ap.add_argument("--disease", required=True)
    ap.add_argument("--bed_prefix", required=True)
    ap.add_argument("--latent_dim", type=int, required=True)
    ap.add_argument("--num_sample", type=int, required=True)
    ap.add_argument("--num_layer", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--beta_list", type=str, default="1.0,2.0,4.0,10.0")
    ap.add_argument("--feature_downsample", type=int, default=50000)
    ap.add_argument("--out_dir", type=str, default=".")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    X = load_bed_as_float32(args.bed_prefix, missing="mean")
    X = random_feature_subset(X, args.feature_downsample, seed=SEED)
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=SEED)

    models = []
    models.append((f"qgVAE_NS{args.num_sample}", build_gvae(X.shape[1], args.latent_dim, args.num_layer, args.num_sample), True, os.path.join(args.out_dir, f"{args.disease}_{args.latent_dim}_{args.num_sample}_{args.num_layer}_qgvae.weights.h5")))
    models.append(("BaselineVAE_NS1", build_baseline_vae(X.shape[1], args.latent_dim, args.num_layer), False, os.path.join(args.out_dir, f"{args.disease}_{args.latent_dim}_1_{args.num_layer}_baseline_vae.weights.h5")))
    for beta in [float(x) for x in args.beta_list.split(",") if x.strip()]:
        tag = str(beta).replace(".", "p")
        models.append((f"BetaVAE_NS1_beta{beta}", build_beta_vae(X.shape[1], args.latent_dim, args.num_layer, beta), False, os.path.join(args.out_dir, f"{args.disease}_{args.latent_dim}_1_{args.num_layer}_beta{tag}_vae.weights.h5")))

    rows = []
    for name, model, is_gvae, weight_path in models:
        print(f"[INFO] Training {name}")
        model = train_one(model, X_train, X_val, args.epochs, args.batch_size, weight_path)
        Xhat = model(X.astype(np.float32), training=False)[0].numpy()
        rec = reconstruction_metrics(X, Xhat)
        rob = compute_robustness(model, X, is_gvae)
        rows.append({"model": name, **rec, "Robustness_invNoiseSens": rob})

    out_csv = os.path.join(args.out_dir, f"{args.disease}_{args.latent_dim}_{args.num_sample}_{args.num_layer}_all_models_metrics.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[DONE] Wrote {out_csv}")


if __name__ == "__main__":
    main()
