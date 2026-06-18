#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train gVAE, baseline VAE, and beta-VAE models from PLINK BED files.

This script is the main model-training entry point. The model architecture is
imported from model.py so that training and downstream XAI use the same
implementation.
"""

from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split

from gvae.model import GVAE, BaselineVAE, BetaVAE
from gvae.metrics import (
    evaluate_mse,
    evaluate_r_square,
    r2_global_flat,
    r2_mean_per_snp,
    r2_median_per_snp,
)

SEED = 42
AUTOTUNE = tf.data.AUTOTUNE
np.random.seed(SEED)
tf.random.set_seed(SEED)
_rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------
# Optional BED reader import
# ---------------------------------------------------------------------
OPEN_BED_BACKEND = None
try:
    from pandas_plink import open_bed  # type: ignore
    OPEN_BED_BACKEND = "pandas-plink"
except Exception:
    try:
        from bed_reader import open_bed  # type: ignore
        OPEN_BED_BACKEND = "bed-reader"
    except Exception as exc:
        raise ImportError(
            "Could not import a PLINK BED reader. Install one of:\n"
            "  pip install pandas-plink\n"
            "  pip install bed-reader"
        ) from exc


# ---------------------------------------------------------------------
# Reproducibility / GPU setup
# ---------------------------------------------------------------------
def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def enable_gpu_memory_growth() -> None:
    physical_devices = tf.config.list_physical_devices("GPU")
    for device in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except Exception:
            pass


# ---------------------------------------------------------------------
# SNP-ID and GWAS helpers
# ---------------------------------------------------------------------
def _read_snp_ids_from_tped(tped_file: str) -> list[str]:
    snp_ids: list[str] = []
    with open(tped_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            snp_ids.append(line.split()[1])
    return snp_ids


def _read_snp_ids_from_bim(bim_file: str) -> list[str]:
    snp_ids: list[str] = []
    with open(bim_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            snp_ids.append(line.split()[1])
    return snp_ids


def _load_gwas_assoc(assoc_path: str) -> tuple[np.ndarray, np.ndarray]:
    snps: list[str] = []
    ps: list[float] = []

    with open(assoc_path, "r") as f:
        header = f.readline().strip().split()
        if not header:
            raise ValueError(f"Empty GWAS file: {assoc_path}")
        if "SNP" not in header or "P" not in header:
            raise ValueError(f"GWAS file must contain SNP and P columns. Header={header[:30]}")

        snp_i = header.index("SNP")
        p_i = header.index("P")

        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) <= max(snp_i, p_i):
                continue
            try:
                pv = float(parts[p_i])
            except Exception:
                continue
            snps.append(parts[snp_i])
            ps.append(pv)

    if not snps:
        raise ValueError(f"No usable SNP/P rows found in: {assoc_path}")

    return np.asarray(snps, dtype=str), np.asarray(ps, dtype=float)


# ---------------------------------------------------------------------
# BED loading and structured SNP filtering
# ---------------------------------------------------------------------
def load_bed_as_float32(
    bed_prefix: str,
    *,
    count_A1: bool = True,
    missing: str = "mean",
) -> np.ndarray:
    """
    Load PLINK bed/bim/fam genotype matrix into float32 array X (N, D).

    Missing values are imputed either with zero or with the per-SNP mean.
    """
    bed_path = bed_prefix + ".bed"
    bim_path = bed_prefix + ".bim"
    fam_path = bed_prefix + ".fam"

    for path in (bed_path, bim_path, fam_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required PLINK file: {path}")

    bed = open_bed(bed_path, count_A1=count_A1)
    X = bed.read().astype(np.float32)

    if np.any(np.isnan(X)):
        if missing == "zero":
            X = np.nan_to_num(X, nan=0.0)
        elif missing == "mean":
            col_means = np.nanmean(X, axis=0)
            col_means = np.nan_to_num(col_means, nan=0.0)
            nan_mask = np.isnan(X)
            X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
        else:
            raise ValueError("missing must be one of: 'zero' | 'mean'")

    return X.astype(np.float32, copy=False)


def load_data_bed(
    bed_prefix: str,
    *,
    feature_mode: str | None = "gwas_top",
    downsample_d: int | None = 50000,
    gwas_assoc_path: str | None = None,
    tped_file: str | None = None,
    bim_file: str | None = None,
    count_A1: bool = True,
    missing: str = "mean",
    return_indices: bool = False,
):
    """
    Load BED data and optionally apply structured GWAS-top SNP filtering.

    The reviewer-facing workflow does not perform random SNP downsampling here.
    When feature_mode='gwas_top', SNPs are selected by GWAS p-value ranking.
    """
    if feature_mode not in (None, "gwas_top"):
        raise ValueError("feature_mode must be either None or 'gwas_top'.")

    arr = load_bed_as_float32(bed_prefix, count_A1=count_A1, missing=missing)
    N, D = arr.shape

    if tped_file is not None:
        snp_ids = _read_snp_ids_from_tped(tped_file)
    else:
        if bim_file is None:
            bim_file = bed_prefix + ".bim"
        snp_ids = _read_snp_ids_from_bim(bim_file)

    if len(snp_ids) != D:
        raise ValueError(
            f"SNP-ID count ({len(snp_ids)}) does not match genotype D ({D}). "
            "The genotype columns must align with the TPED/BIM SNP order."
        )

    row_idx = np.arange(N)
    col_idx = np.arange(D)
    kept_snp_ids = list(snp_ids)

    if feature_mode == "gwas_top":
        if downsample_d is None:
            raise ValueError("feature_mode='gwas_top' requires downsample_d.")
        if gwas_assoc_path is None:
            raise ValueError("feature_mode='gwas_top' requires gwas_assoc_path.")

        gwas_snps, gwas_p = _load_gwas_assoc(gwas_assoc_path)
        order = np.argsort(gwas_p)
        topM = int(min(len(order), int(downsample_d)))
        gwas_top_snps = set(gwas_snps[order[:topM]].tolist())

        snp_ids_arr = np.asarray(snp_ids, dtype=str)
        mask = np.isin(snp_ids_arr, list(gwas_top_snps))

        if mask.sum() == 0:
            raise RuntimeError(
                f"No overlap between GWAS top {topM} SNPs and genotype columns.\n"
                "Likely SNP ID mismatch, e.g. rsIDs vs chr:pos, build mismatch, or different SNP order."
            )

        col_idx = np.where(mask)[0]
        arr = arr[:, col_idx].astype(np.float32, copy=False)
        kept_snp_ids = snp_ids_arr[col_idx].tolist()

        print(
            f"[INFO] Structured SNP filtering complete: "
            f"kept {arr.shape[1]} SNPs from GWAS top {topM} candidates."
        )
    else:
        print(f"[INFO] No SNP filtering applied; using all {arr.shape[1]} aligned SNPs.")

    if return_indices:
        return arr, row_idx, col_idx, kept_snp_ids
    return arr

# ---------------------------------------------------------------------
# Dataset and robustness helpers
# ---------------------------------------------------------------------
def make_tf_dataset(array: np.ndarray, batch_size: int, shuffle: bool = True):
    ds = tf.data.Dataset.from_tensor_slices(array.astype(np.float32))
    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(len(array), 10000),
            seed=SEED,
            reshuffle_each_iteration=True,
        )
    return ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)


def _encode_mu_batches(vae, X: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    mus = []
    for i in range(0, X.shape[0], batch_size):
        xb = tf.convert_to_tensor(X[i:i + batch_size], dtype=tf.float32)
        mu, _ = vae.encode(xb)
        mus.append(mu.numpy().astype(np.float32))
    return np.vstack(mus)


def compute_input_noise_robustness(
    vae,
    geno: np.ndarray,
    eps: float = 0.05,
    max_n: int = 4000,
    batch_size: int = 2048,
):
    n = geno.shape[0]
    take = _rng.choice(n, size=min(max_n, n), replace=False)
    X = geno[take].astype(np.float32, copy=False)

    sd = np.nanstd(X, axis=0, dtype=np.float64).astype(np.float32)
    sd[sd < 1e-6] = 1.0
    noise = _rng.normal(0.0, eps, size=X.shape).astype(np.float32) * sd
    Xp = X + noise

    Z = _encode_mu_batches(vae, X, batch_size=batch_size)
    Zp = _encode_mu_batches(vae, Xp, batch_size=batch_size)

    rel = (np.linalg.norm(Zp - Z, axis=1) / (np.linalg.norm(Z, axis=1) + 1e-12)).astype(np.float32)

    ns_mean = float(np.mean(rel))
    ns_median = float(np.median(rel))
    robustness_inv = float(1.0 / (ns_mean + 1e-12))
    return ns_mean, ns_median, robustness_inv


def make_optimizer():
    return tf.keras.optimizers.Adam(
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-5,
            decay_steps=10_000,
            decay_rate=0.9,
        )
    )

def evaluate_reconstruction(model, data: np.ndarray, batch_size: int):
    pred = model.predict(data, batch_size=batch_size, verbose=1)
    recon = pred[0] if isinstance(pred, (tuple, list)) else pred
    recon = recon.astype(np.float32)

    return {
        "R2": evaluate_r_square(data, recon),
        "MSE": evaluate_mse(data, recon),
        "R2_flat": r2_global_flat(data, recon),
        "R2_snp_mean": r2_mean_per_snp(data, recon),
        "R2_snp_median": r2_median_per_snp(data, recon),
        "reconstruction": recon,
    }

# ---------------------------------------------------------------------
# Training driver
# ---------------------------------------------------------------------
def train_vae_for_disease(
    disease_name: str,
    bed_prefix: str,
    latent_dim: int,
    num_epochs: int,
    batch_size: int,
    num_samples: int = 10,
    num_layers: int = 1,
    beta_values=None,
    feature_mode: str | None = "gwas_top",
    downsample_d: int = 50000,
    gwas_assoc_path: str | None = None,
    tped_file: str | None = None,
    bim_file: str | None = None,
    output_dir: str = ".",
):
    os.makedirs(output_dir, exist_ok=True)

    if beta_values is None:
        beta_values = [4.0]

    if feature_mode == "gwas_top" and gwas_assoc_path is None:
        gwas_assoc_path = f"/work/long_lab/for_Ariel/gwas_results/{disease_name}_gwas.assoc"

    print(f"[INFO] Loading BED prefix: {bed_prefix}")
    data, row_idx, col_idx, kept_snps = load_data_bed(
        bed_prefix=bed_prefix,
        feature_mode=feature_mode,
        downsample_d=downsample_d,
        gwas_assoc_path=gwas_assoc_path,
        tped_file=tped_file,
        bim_file=bim_file,
        missing="mean",
        return_indices=True,
    )
    print(f"[INFO] Data shape after SNP filtering: {data.shape}")

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=SEED)
    train_ds = make_tf_dataset(train_data, batch_size=batch_size, shuffle=True)
    val_ds = make_tf_dataset(test_data, batch_size=batch_size, shuffle=False)
    original_dim = train_data.shape[1]

    summary_rows = []

    # -----------------------
    # 1) gVAE
    # -----------------------
    gvae = GVAE(
        original_dim=original_dim,
        latent_dim=latent_dim,
        num_samples=num_samples,
        num_layers=num_layers,
    )
    gvae.compile(optimizer=make_optimizer())

    gvae_ckpt = os.path.join(
        output_dir,
        f"{disease_name}_{latent_dim}_{num_samples}_{num_layers}_gvae.weights.h5",
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=gvae_ckpt,
            save_weights_only=True,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
    ]

    print(f"[INFO] Train gVAE: {disease_name} LD={latent_dim} NS={num_samples} L={num_layers}")
    gvae_history = gvae.fit(
        train_ds,
        epochs=num_epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=2,
    )
    gvae.load_weights(gvae_ckpt)

    gvae_eval = evaluate_reconstruction(gvae, data, batch_size=batch_size)
    gvae_ns_mean, gvae_ns_median, gvae_rob = compute_input_noise_robustness(
        gvae, data, eps=0.05, max_n=5000, batch_size=batch_size
    )

    print(
        f"[gVAE] R2_global={gvae_eval['R2']:.4f} "
        f"MSE={gvae_eval['MSE']:.6f} "
        f"NoiseSens_mean={gvae_ns_mean:.6f} Robust={gvae_rob:.6f}"
    )

    summary_rows.append({
        "model": f"gVAE_NS{num_samples}",
        "R2": gvae_eval["R2"],
        "MSE": gvae_eval["MSE"],
        "NoiseSens_meanRelChange": gvae_ns_mean,
        "NoiseSens_medianRelChange": gvae_ns_median,
        "Robustness_invNoiseSens": gvae_rob,
    })

    # -----------------------
    # 2) Baseline VAE
    # -----------------------
    baseline = BaselineVAE(original_dim, latent_dim, num_layers=num_layers)
    baseline.compile(optimizer=make_optimizer())

    baseline_ckpt = os.path.join(
        output_dir,
        f"{disease_name}_{latent_dim}_1_{num_layers}_baseline_vae.weights.h5",
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=baseline_ckpt,
            save_weights_only=True,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
    ]

    print("[INFO] Train BaselineVAE (NS=1)")
    baseline.fit(train_ds, epochs=num_epochs, validation_data=val_ds, callbacks=callbacks, verbose=2)
    baseline.load_weights(baseline_ckpt)

    baseline_eval = evaluate_reconstruction(baseline, data, batch_size=batch_size)
    base_ns_mean, base_ns_median, base_rob = compute_input_noise_robustness(
        baseline, data, eps=0.05, max_n=5000, batch_size=batch_size
    )

    print(
        f"[BaselineVAE] R2_global={baseline_eval['R2']:.4f} "
        f"MSE={baseline_eval['MSE']:.6f} "
        f"NoiseSens_mean={base_ns_mean:.6f} Robust={base_rob:.6f}"
    )

    summary_rows.append({
        "model": "BaselineVAE_NS1",
        "R2": baseline_eval["R2"],
        "MSE": baseline_eval["MSE"],
        "NoiseSens_meanRelChange": base_ns_mean,
        "NoiseSens_medianRelChange": base_ns_median,
        "Robustness_invNoiseSens": base_rob,
    })

    # -----------------------
    # 3) Beta-VAE
    # -----------------------
    for beta in beta_values:
        beta_tag = str(beta).replace(".", "p")
        betavae = BetaVAE(
            original_dim,
            latent_dim,
            beta=beta,
            num_layers=num_layers,
        )
        betavae.compile(optimizer=make_optimizer())

        beta_ckpt = os.path.join(
            output_dir,
            f"{disease_name}_{latent_dim}_1_{num_layers}_beta{beta_tag}_vae.weights.h5",
        )

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=beta_ckpt,
                save_weights_only=True,
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
            ),
        ]

        print(f"[INFO] Train BetaVAE beta={beta}")
        betavae.fit(train_ds, epochs=num_epochs, validation_data=val_ds, callbacks=callbacks, verbose=2)
        betavae.load_weights(beta_ckpt)

        beta_eval = evaluate_reconstruction(betavae, data, batch_size=batch_size)
        beta_ns_mean, beta_ns_median, beta_rob = compute_input_noise_robustness(
            betavae, data, eps=0.05, max_n=5000, batch_size=batch_size
        )

        print(
            f"[BetaVAE beta={beta}] R2_global={beta_eval['R2']:.4f} "
            f"MSE={beta_eval['MSE']:.6f} "
            f"NoiseSens_mean={beta_ns_mean:.6f} Robust={beta_rob:.6f}"
        )

        summary_rows.append({
            "model": f"BetaVAE_NS1_beta{beta}",
            "R2": beta_eval["R2"],
            "MSE": beta_eval["MSE"],
            "NoiseSens_meanRelChange": beta_ns_mean,
            "NoiseSens_medianRelChange": beta_ns_median,
            "Robustness_invNoiseSens": beta_rob,
        })

    summary_df = pd.DataFrame(summary_rows)
    out_csv = os.path.join(
        output_dir,
        f"{disease_name}_{latent_dim}_{num_samples}_{num_layers}_all_models_metrics.csv",
    )
    summary_df.to_csv(out_csv, index=False)
    print(f"[INFO] Wrote: {out_csv}")

    return gvae_eval["R2"], gvae_eval["MSE"], gvae_rob, gvae_history


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train gVAE/BaselineVAE/BetaVAE from PLINK BED files.")
    parser.add_argument("--disease", type=str, required=True)
    parser.add_argument("--num_sample", type=int, required=True)
    parser.add_argument("--latent_dim", type=int, required=True)
    parser.add_argument("--num_layer", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--bed_prefix", type=str, required=True, help="Input prefix without .bed/.bim/.fam")
    parser.add_argument("--tped_file", type=str, default=None)
    parser.add_argument("--bim_file", type=str, default=None)

    parser.add_argument("--feature_mode", type=str, default="gwas_top", choices=["none", "gwas_top"])
    parser.add_argument("--downsample_d", type=int, default=50000,
                        help="Number of top GWAS-ranked SNPs retained when feature_mode='gwas_top'.")
    parser.add_argument("--gwas_assoc_path", type=str, default=None)

    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--beta_list", type=str, default="")
    parser.add_argument("--output_dir", type=str, default=".")

    args = parser.parse_args()

    set_seed(SEED)
    enable_gpu_memory_growth()

    betas = [float(b) for b in args.beta_list.split(",") if b.strip()] if args.beta_list.strip() else [args.beta]
    feature_mode = None if args.feature_mode == "none" else args.feature_mode

    r2, mse, rob, _ = train_vae_for_disease(
        disease_name=args.disease,
        bed_prefix=args.bed_prefix,
        latent_dim=args.latent_dim,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_samples=args.num_sample,
        num_layers=args.num_layer,
        beta_values=betas,
        feature_mode=feature_mode,
        downsample_d=args.downsample_d,
        gwas_assoc_path=args.gwas_assoc_path,
        tped_file=args.tped_file,
        bim_file=args.bim_file,
        output_dir=args.output_dir,
    )

    with open(os.path.join(args.output_dir, "finished_jobs.txt"), "a") as f:
        f.write(f"{args.disease}_{args.num_sample}_{args.latent_dim}_{args.num_layer}\n")

    print(f"[DONE] {args.disease}: gVAE R2={r2:.4f} MSE={mse:.6f} Robust={rob:.6f}")

if __name__ == "__main__":
    main()
