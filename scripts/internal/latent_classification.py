#!/usr/bin/env python3
from __future__ import annotations
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name != "app" else Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import argparse
import json
import os
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl
import tensorflow as tf
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.metrics import AUC

from gvae.models import build_baseline_vae, build_beta_vae, build_gvae

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_genotype_csv(file_path: str, separator: str = ",") -> np.ndarray:
    df = pl.read_csv(file_path, has_header=False, separator=separator, null_values=["9", "NA", "NaN"], infer_schema_length=1000)
    df = df.with_columns(pl.all().cast(pl.Float32))
    return df.fill_null(strategy="mean").transpose(include_header=False).to_numpy().astype(np.float32)


def load_phenotype_auto(phen_path: str):
    df = pd.read_csv(phen_path, delim_whitespace=True, header=None)
    y_raw = df.iloc[:, 2].to_numpy()
    missing_codes = set([-9, 9, 99, 999])
    keep = ~np.isin(y_raw, list(missing_codes))
    y_kept = y_raw[keep]
    uniq = np.unique(y_kept)
    if set(uniq.tolist()).issubset({0, 1}):
        return y_raw.astype(np.int32), keep, "classification"
    if set(uniq.tolist()).issubset({1, 2}):
        y = np.full_like(y_raw, -1, dtype=np.int32)
        y[y_raw == 1] = 0
        y[y_raw == 2] = 1
        keep &= (y != -1)
        return y, keep, "classification"
    return y_raw.astype(np.float32), keep, "regression"


def train_representation(X: np.ndarray, model_type: str, latent_dim: int, num_samples: int, num_layers: int, beta: float | None, epochs: int, batch_size: int):
    if model_type == "baseline":
        model = build_baseline_vae(X.shape[1], latent_dim, num_layers)
        is_gvae = False
    elif model_type == "betavae":
        model = build_beta_vae(X.shape[1], latent_dim, num_layers, beta or 4.0)
        is_gvae = False
    else:
        model = build_gvae(X.shape[1], latent_dim, num_layers, num_samples)
        is_gvae = True
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    ds = tf.data.Dataset.from_tensor_slices(X.astype(np.float32)).shuffle(min(len(X), 10000), seed=SEED).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    model.fit(ds, epochs=epochs, verbose=2)
    if is_gvae:
        _, _, _, z = model(X.astype(np.float32), training=False)
        return model, z.numpy()
    mu, _ = model.encode(X.astype(np.float32))
    return model, mu.numpy()


def build_classifier(input_dim: int):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation="relu")(inp)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy", AUC(name="auc")])
    return model


def build_regressor(input_dim: int):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation="relu")(inp)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="linear")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disease", required=True)
    ap.add_argument("--base_path", required=True)
    ap.add_argument("--model_type", required=True, choices=["baseline", "qgvae", "betavae"])
    ap.add_argument("--latent_dim", required=True, type=int)
    ap.add_argument("--num_samples", required=True, type=int)
    ap.add_argument("--num_layers", required=True, type=int)
    ap.add_argument("--beta", default=None, type=float)
    ap.add_argument("--train_vae_epochs", default=25, type=int)
    ap.add_argument("--vae_batch_size", default=256, type=int)
    ap.add_argument("--epochs", default=120, type=int)
    ap.add_argument("--batch_size", default=256, type=int)
    ap.add_argument("--val_size", default=0.2, type=float)
    ap.add_argument("--out_root", default="latent_classification_outputs")
    ap.add_argument("--cache_latents", action="store_true")
    args = ap.parse_args()

    geno_csv = os.path.join(args.base_path, f"{args.disease}_filtered.csv")
    phen_path = os.path.join(args.base_path, f"{args.disease}_origin.phen")
    X = load_genotype_csv(geno_csv)
    y_all, keep, task = load_phenotype_auto(phen_path)
    X = X[keep]
    y = y_all[keep]

    os.makedirs(args.out_root, exist_ok=True)
    hist_dir = os.path.join(args.out_root, "class_history")
    met_dir = os.path.join(args.out_root, "class_metrics")
    lat_dir = os.path.join(args.out_root, "class_latents")
    sum_dir = os.path.join(args.out_root, "class_summary")
    for d in [hist_dir, met_dir, lat_dir, sum_dir]:
        os.makedirs(d, exist_ok=True)

    beta_tag = "NA" if args.beta is None else str(args.beta)
    run_tag = f"{args.disease}_{args.model_type}_LD{args.latent_dim}_NS{args.num_samples}_L{args.num_layers}_B{beta_tag}"
    cache_path = os.path.join(lat_dir, f"{run_tag}.npy")

    if args.cache_latents and os.path.exists(cache_path):
        Z = np.load(cache_path)
    else:
        _, Z = train_representation(X, args.model_type, args.latent_dim, args.num_samples, args.num_layers, args.beta, args.train_vae_epochs, args.vae_batch_size)
        if args.cache_latents:
            np.save(cache_path, Z)

    Xtr, Xva, ytr, yva = train_test_split(Z, y, test_size=args.val_size, random_state=SEED, stratify=y if task == "classification" else None)

    if task == "classification":
        clf = build_classifier(Z.shape[1])
        hist = clf.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=args.epochs, batch_size=args.batch_size, verbose=2,
                       callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=10, restore_best_weights=True)])
        pva = clf.predict(Xva, batch_size=args.batch_size, verbose=0).reshape(-1)
        final_metrics = {"val_auc_final": float(roc_auc_score(yva, pva)), "val_acc_final": float(accuracy_score(yva, (pva >= 0.5).astype(np.int32))), "n_val": int(len(yva))}
    else:
        reg = build_regressor(Z.shape[1])
        hist = reg.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=args.epochs, batch_size=args.batch_size, verbose=2,
                       callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_rmse", mode="min", patience=10, restore_best_weights=True)])
        pva = reg.predict(Xva, batch_size=args.batch_size, verbose=0).reshape(-1)
        mse = float(np.mean((yva - pva) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(yva - pva)))
        ss_res = float(np.sum((yva - pva) ** 2))
        ss_tot = float(np.sum((yva - np.mean(yva)) ** 2) + 1e-12)
        final_metrics = {"val_r2_final": float(1.0 - ss_res / ss_tot), "val_rmse_final": rmse, "val_mae_final": mae, "n_val": int(len(yva))}

    hist_path = os.path.join(hist_dir, f"{run_tag}.pickle")
    with open(hist_path, "wb") as f:
        pickle.dump(hist.history, f)
    met_path = os.path.join(met_dir, f"{run_tag}.json")
    with open(met_path, "w") as f:
        json.dump({"run": run_tag, "task": task, "final": final_metrics}, f, indent=2)

    summary_path = os.path.join(sum_dir, f"{args.disease}_summary.csv")
    row = {"run": run_tag, "disease": args.disease, "model_type": args.model_type, "LD": args.latent_dim, "NS": args.num_samples, "L": args.num_layers, "beta": args.beta if args.beta is not None else np.nan, **final_metrics}
    if os.path.exists(summary_path):
        sdf = pd.read_csv(summary_path)
        sdf = sdf[sdf["run"] != run_tag]
        sdf = pd.concat([sdf, pd.DataFrame([row])], ignore_index=True)
    else:
        sdf = pd.DataFrame([row])
    sdf.to_csv(summary_path, index=False)
    print(f"[DONE] Saved history, metrics, and summary for {run_tag}")


if __name__ == "__main__":
    main()
