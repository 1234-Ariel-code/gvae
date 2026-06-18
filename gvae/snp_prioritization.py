#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHAP-based SNP prioritization using the shared gVAE architecture.

This script imports the model definition from model.py. It therefore uses the
same gVAE architecture as the main training pipeline and avoids a second,
independent model implementation inside the XAI script.
"""

from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import polars as pl
import shap

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from gvae.model import GVAE
from gvae.metrics import evaluate_mse, evaluate_r_square


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ---------------------------------------------------------------------
# GPU setup
# ---------------------------------------------------------------------
def enable_gpu_memory_growth() -> None:
    physical_devices = tf.config.list_physical_devices("GPU")
    for device in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except Exception:
            pass


# ---------------------------------------------------------------------
# SNP-ID readers and GWAS association loader
# ---------------------------------------------------------------------
def _read_snp_ids_from_tped(tped_file: str) -> list[str]:
    """
    TPED format: chr snp_id genetic_dist bp a1_i a2_i ...
    SNP ID is the second column.
    """
    snp_ids: list[str] = []
    with open(tped_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            snp_ids.append(line.split()[1])
    return snp_ids


def _read_snp_ids_from_bim(bim_file: str) -> list[str]:
    """
    BIM format: chr snp_id cm bp a1 a2.
    SNP ID is the second column.
    """
    snp_ids: list[str] = []
    with open(bim_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            snp_ids.append(line.split()[1])
    return snp_ids


def _load_gwas_assoc(assoc_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a PLINK .assoc file and return SNP IDs with p-values.

    Expected columns include SNP and P.
    """
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
# Genotype loading and structured SNP filtering
# ---------------------------------------------------------------------
def load_data(
    file_path: str,
    separator: str = ",",
    *,
    gwas_assoc_path: str,
    downsample_d: int = 50000,
    tped_file: str | None = None,
    bim_file: str | None = None,
    return_indices: bool = False,
):
    """
    Load genotype data and apply the manuscript XAI SNP filtering step.

    This reviewer-facing XAI workflow does not perform random row or SNP
    downsampling. SNP filtering is performed once using the structured
    GWAS-top SNP selection step.

    Parameters
    ----------
    file_path : str
        Genotype CSV path.
    separator : str
        CSV separator.
    gwas_assoc_path : str
        PLINK .assoc file used to rank SNPs by GWAS p-value.
    downsample_d : int
        Number of GWAS-ranked SNPs retained for the XAI analysis.
    tped_file : str, optional
        TPED file used to align SNP IDs with genotype columns.
    bim_file : str, optional
        BIM file used to align SNP IDs with genotype columns if TPED is not used.
    return_indices : bool
        If True, return selected column indices and kept SNP IDs.

    Returns
    -------
    arr : np.ndarray
        Filtered genotype matrix, shape (N individuals, D selected SNPs).
    optionally: arr, col_idx, kept_snp_ids
    """
    df = pl.read_csv(
        file_path,
        has_header=False,
        separator=separator,
        null_values=["9", "NA", "NaN"],
        infer_schema_length=1000,
    ).with_columns(pl.all().cast(pl.Float32))

    arr_noT = df.fill_null(strategy="mean").to_numpy()
    arr_T = df.fill_null(strategy="mean").transpose(include_header=False).to_numpy()

    if tped_file is not None:
        snp_ids = _read_snp_ids_from_tped(tped_file)
    elif bim_file is not None:
        snp_ids = _read_snp_ids_from_bim(bim_file)
    else:
        raise ValueError("A TPED or BIM file is required to align SNP IDs.")

    M = len(snp_ids)

    if arr_noT.shape[1] == M:
        arr = arr_noT
    elif arr_T.shape[1] == M:
        arr = arr_T
    else:
        raise ValueError(
            f"[ORIENTATION ERROR] Neither genotype orientation matches SNP count.\n"
            f"SNP IDs: {M}\n"
            f"arr_noT shape: {arr_noT.shape} with D={arr_noT.shape[1]}\n"
            f"arr_T   shape: {arr_T.shape} with D={arr_T.shape[1]}\n"
            "The genotype columns must align with the TPED/BIM SNP order."
        )

    arr = arr.astype(np.float32, copy=False)
    _, D = arr.shape

    if len(snp_ids) != D:
        raise ValueError(
            f"SNP-ID count ({len(snp_ids)}) does not match genotype D ({D}). "
            "The genotype columns must align with the TPED/BIM SNP order."
        )

    gwas_snps, gwas_p = _load_gwas_assoc(gwas_assoc_path)
    order = np.argsort(gwas_p)
    topM = int(min(len(order), downsample_d))
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

    if return_indices:
        return arr, col_idx, kept_snp_ids

    return arr


# ---------------------------------------------------------------------
# Model prediction helpers
# ---------------------------------------------------------------------
def predict_reconstruction_quantiles_and_draws(
    vae: GVAE,
    data: np.ndarray,
    batch_size: int,
):
    """
    Predict reconstruction, q25/q75 representation, and posterior samples.

    Posterior samples are returned as a list of K arrays, each with shape
    (N individuals, latent_dim). This format is used by the SHAP routine.
    """
    recon_blocks = []
    quantile_blocks = []
    draw_blocks = [[] for _ in range(vae.num_samples)]

    for start in range(0, data.shape[0], batch_size):
        xb = tf.convert_to_tensor(data[start:start + batch_size], dtype=tf.float32)

        x_hat, z_quantiles = vae(xb, training=False)
        mu, log_var = vae.encode(xb)
        z_samples = vae.reparameterize(mu, log_var)  # (K, B, LD)

        recon_blocks.append(x_hat.numpy().astype(np.float32))
        quantile_blocks.append(z_quantiles.numpy().astype(np.float32))

        z_np = z_samples.numpy().astype(np.float32)
        for k in range(vae.num_samples):
            draw_blocks[k].append(z_np[k])

    recon = np.vstack(recon_blocks)
    z_quantiles = np.vstack(quantile_blocks)
    z_draws = [np.vstack(parts) for parts in draw_blocks]

    return recon, z_quantiles, z_draws


# ---------------------------------------------------------------------
# SHAP helpers
# ---------------------------------------------------------------------
def _safe_shap_values(explainer, X_chunk):
    """Robustly obtain SHAP values across SHAP versions."""
    out = explainer(X_chunk)
    if hasattr(out, "values"):
        return out.values
    return np.asarray(out)


def _top_shap_snps_per_latent_given_targets(
    data: np.ndarray,
    targets: np.ndarray,
    snp_names: list[str],
    shap_top_k: int,
    chunk_size: int,
    out_prefix: str,
    output_dir: str,
):
    """
    Compute top SHAP-ranked SNPs per latent dimension.

    The random subset below is used only as the SHAP background set. It does not
    redefine or downsample the analyzed SNP set.
    """
    N, D = data.shape
    LD = targets.shape[1]
    os.makedirs(output_dir, exist_ok=True)

    top_snp_records = []
    extended_blocks = []

    bg_size = min(100, N)
    bg_idx = np.random.choice(N, size=bg_size, replace=False)
    background = data[bg_idx]

    for i in range(LD):
        print(f"  · Latent dim {i + 1}/{LD}")

        y = targets[:, i]
        model = LinearRegression()
        model.fit(data, y)

        explainer = shap.Explainer(model, background)

        mean_abs = np.zeros(D, dtype=np.float64)
        n_chunks = int(np.ceil(N / chunk_size))

        for j in range(n_chunks):
            s = j * chunk_size
            e = min((j + 1) * chunk_size, N)
            Xc = data[s:e]
            sv = _safe_shap_values(explainer, Xc)
            mean_abs += np.abs(sv).sum(axis=0)

        mean_abs /= float(N)

        k = min(shap_top_k, D)
        top_idx = np.argsort(mean_abs)[-k:]

        block_cols = []
        for idx in top_idx:
            w = mean_abs[idx]
            block_cols.append(data[:, idx][:, None] * w)

            top_snp_records.append({
                "Latent_Dim": f"LD_{i}",
                "SNP_ID": snp_names[idx],
                "SHAP_Importance": float(w),
            })

        block = np.hstack(block_cols) if block_cols else np.empty((N, 0))
        extended_blocks.append(block)

    extended_matrix = np.hstack(extended_blocks) if extended_blocks else np.empty((N, 0))

    mat_path = os.path.join(output_dir, f"{out_prefix}_SHAP_weighted_matrix.csv")
    pd.DataFrame(extended_matrix).to_csv(mat_path, index=False)

    top_df = pd.DataFrame(top_snp_records)
    top_base = os.path.join(output_dir, f"{out_prefix}_top_snps_per_latent")
    top_csv = f"{top_base}.csv"
    top_txt = f"{top_base}.txt"

    top_df.to_csv(top_csv, index=False)
    top_df["SNP_ID"].drop_duplicates().to_csv(top_txt, index=False, header=False)

    print(f"  -> Saved: {top_csv}, {top_txt}")
    print(f"  -> Matrix: {mat_path} shape={extended_matrix.shape}")

    return {
        "top_snps_csv": top_csv,
        "top_snps_txt": top_txt,
        "matrix_csv": mat_path,
        "matrix_shape": extended_matrix.shape,
    }


def compute_snp_contributions_shap_topk_per_latent_all_draws(
    data: np.ndarray,
    z_samples: list[np.ndarray],
    disease_name: str,
    latent_dim: int,
    num_samples: int,
    num_layers: int,
    shap_top_k: int = 10,
    tped_file: str | None = None,
    bim_file: str | None = None,
    selected_col_idx: np.ndarray | None = None,
    chunk_size: int = 500,
    output_dir: str = ".",
):
    """
    Compute SHAP top-ranked SNPs per latent dimension and per posterior draw.

    num_samples is the number of posterior samples used by gVAE.
    shap_top_k is the number of top SHAP-ranked SNPs retained per latent variable.
    These are intentionally separate quantities.
    """
    os.makedirs(output_dir, exist_ok=True)

    if tped_file is not None:
        snp_names = _read_snp_ids_from_tped(tped_file)
    elif bim_file is not None:
        snp_names = _read_snp_ids_from_bim(bim_file)
    else:
        raise ValueError("A TPED or BIM file is required to recover SNP names.")

    if selected_col_idx is not None:
        snp_names = np.asarray(snp_names, dtype=str)[selected_col_idx].tolist()

    if len(snp_names) != data.shape[1]:
        raise ValueError(
            f"SNP name count ({len(snp_names)}) does not match filtered genotype D ({data.shape[1]})."
        )

    if len(z_samples) != num_samples:
        print(f"[WARN] z_samples length {len(z_samples)} != num_samples {num_samples}")

    manifest_rows = []

    for s, targets in enumerate(z_samples, start=1):
        out_prefix = (
            f"{disease_name}_LD{latent_dim}_NS{num_samples}_"
            f"L{num_layers}_SHAPtop{shap_top_k}_S{s}"
        )

        print(f"\nProcessing posterior draw S={s}/{len(z_samples)} ...")

        paths = _top_shap_snps_per_latent_given_targets(
            data=data,
            targets=np.asarray(targets),
            snp_names=snp_names,
            shap_top_k=shap_top_k,
            chunk_size=chunk_size,
            out_prefix=out_prefix,
            output_dir=output_dir,
        )

        manifest_rows.append({
            "draw": s,
            "shap_top_k": shap_top_k,
            **paths,
        })

    manifest = pd.DataFrame(manifest_rows)

    manifest_path = os.path.join(
        output_dir,
        (
            f"{disease_name}_LD{latent_dim}_NS{num_samples}_"
            f"L{num_layers}_SHAPtop{shap_top_k}_SHAP_outputs_manifest.csv"
        ),
    )

    manifest.to_csv(manifest_path, index=False)
    print(f"\n>> Manifest saved: {manifest_path}")

    return manifest


# ---------------------------------------------------------------------
# Training wrapper
# ---------------------------------------------------------------------
def train_vae_for_disease(
    disease_name: str,
    base_path: str,
    latent_dim: int,
    num_epochs: int,
    batch_size: int,
    num_samples: int = 100,
    num_layers: int = 1,
    shap_top_k: int = 10,
    tped_file: str | None = None,
    bim_file: str | None = None,
    assoc_path: str | None = None,
    output_dir: str = ".",
):
    os.makedirs(output_dir, exist_ok=True)

    if assoc_path is None:
        assoc_path = f"/work/long_lab/for_Ariel/gwas_results/{disease_name}_gwas.assoc"
    if tped_file is None and bim_file is None:
        tped_file = f"{base_path}/{disease_name}_origin.tped"

    file_path = f"{base_path}/{disease_name}_filtered.csv"

    data, col_idx, kept_snp_ids = load_data(
        file_path=file_path,
        separator=",",
        gwas_assoc_path=assoc_path,
        downsample_d=50000,
        tped_file=tped_file,
        bim_file=bim_file,
        return_indices=True,
    )

    print(f"Training {disease_name} | LD={latent_dim} | L={num_layers} | NS={num_samples}")
    print(f"Data shape: {data.shape}")

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=SEED)
    original_dim = train_data.shape[1]

    vae = GVAE(
        original_dim=original_dim,
        latent_dim=latent_dim,
        num_samples=num_samples,
        num_layers=num_layers,
    )

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=10000,
        decay_rate=0.9,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    vae.compile(optimizer=optimizer)

    weights_path = os.path.join(
        output_dir,
        f"{disease_name}_gvae_LD{latent_dim}_NS{num_samples}_L{num_layers}.weights.h5",
    )

    history = vae.fit(
        train_data,
        train_data,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_data=(test_data, test_data),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=weights_path,
                save_weights_only=True,
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                verbose=1,
            ),
        ],
        verbose=1,
    )

    output, z_quantiles, z_samples = predict_reconstruction_quantiles_and_draws(
        vae=vae,
        data=data,
        batch_size=batch_size,
    )

    r2 = evaluate_r_square(data, output)
    mse = evaluate_mse(data, output)

    if np.isnan(r2):
        print(f"Skipping: NaN R² for {disease_name} (LD={latent_dim}, L={num_layers}, NS={num_samples})")
        return None, None

    print(
        f"R²={r2:.4f} | MSE={mse:.4f} | "
        f"Disease={disease_name} | LD={latent_dim} | L={num_layers} | NS={num_samples}"
    )

    rep_path = os.path.join(
        output_dir,
        f"rep_{disease_name}_LD{latent_dim}_NS{num_samples}_L{num_layers}_q25q75.csv",
    )
    pd.DataFrame(z_quantiles).to_csv(rep_path, index=False)
    print(f"Saved q25/q75 latent representation -> {rep_path}")

    compute_snp_contributions_shap_topk_per_latent_all_draws(
        data=data,
        z_samples=z_samples,
        disease_name=disease_name,
        latent_dim=latent_dim,
        num_samples=num_samples,
        num_layers=num_layers,
        shap_top_k=shap_top_k,
        tped_file=tped_file,
        bim_file=bim_file,
        selected_col_idx=col_idx,
        chunk_size=500,
        output_dir=output_dir,
    )

    return r2, history


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train shared gVAE model and run SHAP SNP prioritization.")
    parser.add_argument("--disease", type=str, required=True)
    parser.add_argument("--latent_dim", type=int, required=True)
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--base_path", type=str, default="/work/long_lab/for_Ariel/files")
    parser.add_argument("--tped_file", type=str, default=None)
    parser.add_argument("--bim_file", type=str, default=None)
    parser.add_argument("--assoc_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--shap_top_k", type=int, default=10,
                        help="Number of top SHAP-ranked SNPs retained per latent dimension.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()

    enable_gpu_memory_growth()

    train_vae_for_disease(
        disease_name=args.disease,
        base_path=args.base_path,
        latent_dim=args.latent_dim,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        num_layers=args.num_layers,
        shap_top_k=args.shap_top_k,
        tped_file=args.tped_file,
        bim_file=args.bim_file,
        assoc_path=args.assoc_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
