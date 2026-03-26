#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Model, mixed_precision

# NEW: bed_reader
from bed_reader import open_bed

# ----------------------------------------------------------
# Global config / seeds / GPU & mixed precision
# ----------------------------------------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
_rng = np.random.default_rng(SEED)

physical_devices = tf.config.list_physical_devices("GPU")
for device in physical_devices:
    try:
        tf.config.experimental.set_memory_growth(device, True)
    except Exception:
        pass

# Mixed precision ON by default; can be disabled via --no_mixed_precision
AUTOTUNE = tf.data.AUTOTUNE


# ----------------------------------------------------------
# BED loading + imputation
# ----------------------------------------------------------
def load_bedd_as_float32(bed_prefix: str, mean_impute: bool = True) -> np.ndarray:
    """
    Load PLINK bed/bim/fam genotype matrix into float32 array X (N,M).
    Uses count_A1=True (0/1/2 A1 allele count). Missing values often become NaN.
    """
    bed_path = bed_prefix + ".bed"
    if not os.path.exists(bed_path):
        raise FileNotFoundError(f"Missing bed file: {bed_path}")

    bed = open_bed(bed_path, count_A1=True)
    X = bed.read()  # float64, shape (N,M), NaN for missing (usually)
    X = X.astype(np.float32)

    if mean_impute:
        # mean impute per SNP (column)
        col_means = np.nanmean(X, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means).astype(np.float32)
        inds = np.where(np.isnan(X))
        if inds[0].size > 0:
            X[inds] = col_means[inds[1]]
    else:
        # fallback: replace NaNs with 0
        X = np.nan_to_num(X, nan=0.0)

    return X

def looad_bed_as_float32(bed_prefix: str) -> np.ndarray:
    """
    Load PLINK bed/bim/fam genotype matrix into float32 array X (N,M).
    Missing values are replaced with 0 (reference allele).
    """
    bed_path = bed_prefix + ".bed"
    if not os.path.exists(bed_path):
        raise FileNotFoundError(f"Missing bed file: {bed_path}")

    bed = open_bed(bed_path, count_A1=True)
    X = bed.read().astype(np.float32)  # NaNs for missing

    if np.any(np.isnan(X)):
        print("[INFO] Replacing missing genotypes with 0 (reference allele).")
        X = np.nan_to_num(X, nan=0.0)

    return X

"""
PLINK BED loader + downsampling pipeline (rows + SNPs), with optional GWAS-top SNP selection.

Key features:
- Loads genotype matrix from bed/bim/fam into float32: X shape (N_individuals, D_snps)
- Handles missing genotypes (NaN) via configurable imputation (default: 0.0)
- Row downsampling: downsample_n OR downsample_frac
- Feature downsampling:
    * feature_mode=None
    * feature_mode="random"  -> pick random SNP columns
    * feature_mode="gwas_top"-> keep SNPs that are among lowest-P SNPs from a PLINK .assoc file
- Returns indices + kept SNP IDs if return_indices=True

Dependencies:
- numpy
- pandas-plink (recommended) OR bed-reader (fallback)
    pip install pandas-plink
    # or: pip install bed-reader
"""

# -------------------------
# BED reader import (try pandas-plink first, then bed-reader)
# -------------------------
OPEN_BED_BACKEND = None
try:
    # pandas-plink provides open_bed in recent versions
    from pandas_plink import open_bed  # type: ignore
    OPEN_BED_BACKEND = "pandas-plink"
except Exception:
    try:
        # bed-reader also provides open_bed
        from bed_reader import open_bed  # type: ignore
        OPEN_BED_BACKEND = "bed-reader"
    except Exception as e:
        raise ImportError(
            "Could not import open_bed. Install one of:\n"
            "  pip install pandas-plink\n"
            "  pip install bed-reader\n"
        ) from e


# -------------------------
# SNP ID readers
# -------------------------
def _read_snp_ids_from_tped(tped_file: str) -> list[str]:
    """
    TPED format: chr  snp_id  genetic_dist  bp  a1_i a2_i a1_i a2_i ...
    We only need the snp_id column (2nd column).
    """
    snp_ids: list[str] = []
    with open(tped_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            snp_ids.append(parts[1])
    return snp_ids


def _read_snp_ids_from_bim(bim_file: str) -> list[str]:
    """
    BIM format: chr snp_id cm bp a1 a2
    SNP ID is 2nd column.
    """
    snp_ids: list[str] = []
    with open(bim_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            snp_ids.append(parts[1])
    return snp_ids


# -------------------------
# GWAS assoc loader
# -------------------------
def _load_gwas_assoc(assoc_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a PLINK .assoc file and return arrays (SNP, P).

    Expected columns include SNP and P.
    """
    snps: list[str] = []
    ps: list[float] = []

    with open(assoc_path, "r") as f:
        header = f.readline().strip().split()
        if not header:
            raise ValueError(f"Empty GWAS file: {assoc_path}")

        try:
            snp_i = header.index("SNP")
        except ValueError:
            raise ValueError(f"GWAS file missing SNP column: {assoc_path}. Header={header[:20]}")

        try:
            p_i = header.index("P")
        except ValueError:
            raise ValueError(f"GWAS file missing P column: {assoc_path}. Header={header[:20]}")

        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) <= max(snp_i, p_i):
                continue
            snp = parts[snp_i]
            p = parts[p_i]
            try:
                pv = float(p)
            except Exception:
                continue
            snps.append(snp)
            ps.append(pv)

    if not snps:
        raise ValueError(f"No usable SNP/P rows found in: {assoc_path}")

    return np.array(snps, dtype=str), np.array(ps, dtype=float)


# -------------------------
# Core BED loader
# -------------------------
def load_bed_as_float32(
    bed_prefix: str,
    *,
    count_A1: bool = True,
    missing: str = "zero",
) -> np.ndarray:
    """
    Load PLINK bed/bim/fam genotype matrix into float32 array X (N, D).

    Parameters
    ----------
    bed_prefix : str
        Prefix without extension (expects .bed/.bim/.fam).
    count_A1 : bool
        If True, encode as count of A1 alleles (0/1/2).
    missing : str
        Missing genotype handling: "zero" or "mean"
        - "zero": replace NaN with 0.0
        - "mean": replace NaN with per-SNP mean (column-wise)

    Returns
    -------
    X : np.ndarray
        float32 array shape (N_individuals, D_snps)
    """
    bed_path = bed_prefix + ".bed"
    bim_path = bed_prefix + ".bim"
    fam_path = bed_prefix + ".fam"

    for p in (bed_path, bim_path, fam_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required PLINK file: {p}")

    bed = open_bed(bed_path, count_A1=count_A1)
    X = bed.read().astype(np.float32)  # NaNs for missing, shape (N, D)

    if np.any(np.isnan(X)):
        if missing == "zero":
            X = np.nan_to_num(X, nan=0.0)
        elif missing == "mean":
            # column-wise mean imputation (per SNP)
            col_means = np.nanmean(X, axis=0)
            # If a column is all NaN, nanmean -> NaN; replace those means with 0
            col_means = np.nan_to_num(col_means, nan=0.0)
            nan_mask = np.isnan(X)
            X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
        else:
            raise ValueError("missing must be one of: 'zero' | 'mean'")

    return X


# -------------------------
# Unified pipeline: BED + downsampling + GWAS-top SNP selection
# -------------------------
def load_data_bed(
    bed_prefix: str,
    # ---- row downsampling (individuals)
    downsample_n: int | None = None,
    downsample_frac: float | None = None,
    # ---- feature downsampling (SNPs)
    feature_mode: str | None = None,   # None | "random" | "gwas_top"
    downsample_d: int | None = None,
    # GWAS inputs
    gwas_assoc_path: str | None = None,
    # SNP-ID mapping overrides (optional)
    tped_file: str | None = None,
    bim_file: str | None = None,
    # BED read options
    count_A1: bool = True,
    missing: str = "zero",
    # misc
    seed: int = 42,
    return_indices: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray, list[str] | None]:
    """
    BED-native version of your CSV loader.

    Returns
    -------
    arr : (N_down, D_down) float32
    optionally: (arr, row_idx, col_idx, kept_snp_ids)
    """
    # 1) Load X from bed
    arr = load_bed_as_float32(bed_prefix, count_A1=count_A1, missing=missing)
    arr = arr.astype(np.float32, copy=False)
    N, D = arr.shape

    rng = np.random.default_rng(seed)
    row_idx = np.arange(N)
    col_idx = np.arange(D)
    kept_snp_ids: list[str] | None = None

    # 2) SNP IDs (needed for gwas_top and for kept_snp_ids)
    snp_ids: list[str] | None = None
    if tped_file is not None:
        snp_ids = _read_snp_ids_from_tped(tped_file)
    else:
        # prefer explicit bim_file if provided, else infer from bed_prefix
        if bim_file is None:
            bim_file = bed_prefix + ".bim"
        snp_ids = _read_snp_ids_from_bim(bim_file)

    if snp_ids is not None and len(snp_ids) != D:
        raise ValueError(
            f"SNP-ID count ({len(snp_ids)}) does not match genotype D ({D}).\n"
            f"- Check that your bed_prefix matches the bim you provided.\n"
            f"- If you provided a TPED/BIM from a different SNP set/order, they must match exactly."
        )

    # -------------------------
    # Row downsampling
    # -------------------------
    if (downsample_n is not None) and (downsample_frac is not None):
        raise ValueError("Use only one of downsample_n or downsample_frac, not both.")

    target_n = None
    if downsample_frac is not None:
        if not (0.0 < downsample_frac <= 1.0):
            raise ValueError("downsample_frac must be in (0, 1].")
        target_n = max(1, int(round(N * downsample_frac)))
    elif downsample_n is not None:
        if downsample_n <= 0:
            raise ValueError("downsample_n must be > 0.")
        target_n = min(N, int(downsample_n))

    if target_n is not None and target_n < N:
        row_idx = rng.choice(N, size=target_n, replace=False)
        row_idx.sort()
        arr = arr[row_idx, :]
        N = arr.shape[0]

    # -------------------------
    # Feature downsampling
    # -------------------------
    if feature_mode is None:
        kept_snp_ids = snp_ids  # full set (useful when return_indices=True)

    elif feature_mode == "random":
        if downsample_d is None:
            raise ValueError("feature_mode='random' requires downsample_d.")
        target_d = min(D, int(downsample_d))
        if target_d < D:
            col_idx = rng.choice(D, size=target_d, replace=False)
            col_idx.sort()
            arr = arr[:, col_idx]
            kept_snp_ids = np.array(snp_ids, dtype=str)[col_idx].tolist()
        else:
            kept_snp_ids = snp_ids

    elif feature_mode == "gwas_top":
        if downsample_d is None:
            raise ValueError("feature_mode='gwas_top' requires downsample_d.")
        if gwas_assoc_path is None:
            raise ValueError("feature_mode='gwas_top' requires gwas_assoc_path.")
        if snp_ids is None:
            raise ValueError("Internal error: snp_ids should not be None here.")

        gwas_snps, gwas_p = _load_gwas_assoc(gwas_assoc_path)

        # Take top M SNPs by smallest p-value
        order = np.argsort(gwas_p)
        topM = int(min(len(order), int(downsample_d)))
        gwas_top_snps = set(gwas_snps[order[:topM]].tolist())

        snp_ids_arr = np.array(snp_ids, dtype=str)
        mask = np.isin(snp_ids_arr, list(gwas_top_snps))

        if mask.sum() == 0:
            # super common mismatch: GWAS uses rsIDs but BIM uses chr:pos (or vice-versa)
            example_bim = snp_ids_arr[:5].tolist()
            example_gwas = gwas_snps[:5].tolist()
            raise RuntimeError(
                f"No overlap between GWAS top {topM} SNPs and genotype columns.\n"
                f"Likely SNP ID mismatch (rsIDs vs chr:pos, different build, different naming).\n"
                f"Examples:\n"
                f"  BIM SNP IDs (first 5): {example_bim}\n"
                f"  GWAS SNP IDs (first 5): {example_gwas}\n"
            )

        col_idx = np.where(mask)[0]
        arr = arr[:, col_idx]
        kept_snp_ids = snp_ids_arr[col_idx].tolist()

    else:
        raise ValueError("feature_mode must be one of: None | 'random' | 'gwas_top'")

    arr = arr.astype(np.float32, copy=False)

    if return_indices:
        return arr, row_idx, col_idx, kept_snp_ids
    return arr


# ----------------------------------------------------------
# Shared encoder / decoder builders
# ----------------------------------------------------------
def build_encoder(original_dim: int, latent_dim: int, num_layers: int,
                  initial_neurons: int = 128) -> tf.keras.Sequential:
    layers_list = [layers.InputLayer(input_shape=(original_dim,))]
    layers_list.append(layers.LayerNormalization())
    neurons = initial_neurons
    for _ in range(num_layers):
        layers_list.append(layers.Dense(neurons, activation="relu"))
        neurons //= 2
        neurons = max(neurons, latent_dim * 2)
    layers_list.append(layers.Dense(latent_dim * 2))
    return tf.keras.Sequential(layers_list)


def build_qgvae_decoder(latent_dim: int, num_layers: int, original_dim: int,
                        initial_neurons: int = 128) -> tf.keras.Sequential:
    layers_list = [layers.InputLayer(input_shape=(latent_dim * 2,))]
    neurons = initial_neurons
    for _ in range(num_layers):
        layers_list.append(layers.Dense(neurons, activation="relu"))
        neurons *= 2
    layers_list.append(layers.Dense(original_dim))
    return tf.keras.Sequential(layers_list)


def build_baseline_decoder(latent_dim: int, num_layers: int, original_dim: int,
                           initial_neurons: int = 128) -> tf.keras.Sequential:
    layers_list = [layers.InputLayer(input_shape=(latent_dim,))]
    neurons = initial_neurons
    for _ in range(num_layers):
        layers_list.append(layers.Dense(neurons, activation="relu"))
        neurons *= 2
    layers_list.append(layers.Dense(original_dim))
    return tf.keras.Sequential(layers_list)


# ----------------------------------------------------------
# Loss components
# ----------------------------------------------------------
def discrete_mse_loss(x, x_hat):
    x = tf.cast(x, tf.float32)
    x_hat = tf.cast(x_hat, tf.float32)

    d0 = tf.square(x_hat - 0.0)
    d1 = tf.square(x_hat - 1.0)
    d2 = tf.square(x_hat - 2.0)

    m0 = tf.cast(tf.equal(x, 0.0), tf.float32)
    m1 = tf.cast(tf.equal(x, 1.0), tf.float32)
    m2 = tf.cast(tf.equal(x, 2.0), tf.float32)

    loss_discrete = tf.reduce_mean(m0 * d0 + m1 * d1 + m2 * d2)
    mse = tf.reduce_mean(tf.square(x_hat - x))
    return loss_discrete + mse


def kl_divergence(mu, log_var, clip_log_var: bool = False):
    # ALWAYS compute in float32 for numerical safety
    mu = tf.cast(mu, tf.float32)
    log_var = tf.cast(log_var, tf.float32)
    #if clip_log_var:
    #    log_var = tf.clip_by_value(log_var, -1.0, 1.0)
    kl = -0.5 * tf.reduce_sum(1.0 + log_var - tf.square(mu) - tf.exp(log_var), axis=-1)
    return tf.reduce_mean(kl)


# ----------------------------------------------------------
# qgVAE Model  (kept as class VAE for compatibility)
# ----------------------------------------------------------
class VAE(Model):
    def __init__(self, original_dim, latent_dim, num_samples=10, num_layers=1,
                 clip_log_var=False, **kwargs):
        super().__init__(**kwargs)
        self.original_dim = original_dim
        self.latent_dim = latent_dim
        self.num_samples = num_samples
        self.clip_log_var = bool(clip_log_var)

        self.encoder = build_encoder(original_dim, latent_dim, num_layers)
        self.decoder_continuous = build_qgvae_decoder(latent_dim, num_layers, original_dim)

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

    def encode(self, x):
        z = self.encoder(x)
        mu, log_var = tf.split(z, num_or_size_splits=2, axis=1)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        mu = tf.cast(mu, tf.float32)
        log_var = tf.cast(log_var, tf.float32)
        batch_size = tf.shape(mu)[0]

        eps = tf.random.normal((self.num_samples, batch_size, self.latent_dim), dtype=tf.float32)
        mu_exp = tf.expand_dims(mu, 0)
        log_var_exp = tf.expand_dims(log_var, 0)
        z = mu_exp + eps * tf.exp(0.5 * log_var_exp)
        return z

    def compute_row_wise_quantiles(self, z_samples):
        z_sorted = tf.sort(z_samples, axis=0)
        n = tf.shape(z_sorted)[0]
        idx_25 = tf.cast(0.25 * tf.cast(n, tf.float32), tf.int32)
        idx_75 = tf.cast(0.75 * tf.cast(n, tf.float32), tf.int32)
        q25 = z_sorted[idx_25]
        q75 = z_sorted[idx_75]
        z_final = tf.concat([q25, q75], axis=-1)
        return tf.cast(z_final, tf.float16)

    def decode(self, z_final):
        return self.decoder_continuous(z_final)

    def call(self, inputs, training=False):
        mu, log_var = self.encode(inputs)
        z_samples = self.reparameterize(mu, log_var)
        z_final = self.compute_row_wise_quantiles(z_samples)
        x_hat = self.decode(z_final)
        if training:
            return x_hat, mu, log_var, z_final
        return x_hat, z_final

    def train_step(self, data):
        x = data
        with tf.GradientTape() as tape:
            x_hat, mu, log_var, _ = self(x, training=True)
            recon_loss = discrete_mse_loss(x, x_hat)
            kl = kl_divergence(mu, log_var, clip_log_var=self.clip_log_var)
            total_loss = tf.reduce_mean(recon_loss + kl)

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl)
        return {"loss": self.total_loss_tracker.result(),
                "recon_loss": self.recon_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result()}

    def test_step(self, data):
        x = data
        mu, log_var = self.encode(x)
        z_samples = self.reparameterize(mu, log_var)
        z_final = self.compute_row_wise_quantiles(z_samples)
        x_hat = self.decode(z_final)

        recon_loss = discrete_mse_loss(x, x_hat)
        kl = kl_divergence(mu, log_var, clip_log_var=self.clip_log_var)
        total_loss = tf.reduce_mean(recon_loss + kl)

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl)
        return {"loss": self.total_loss_tracker.result(),
                "recon_loss": self.recon_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result()}


# ----------------------------------------------------------
# BaselineVAE + BetaVAE
# ----------------------------------------------------------
class BaselineVAE(Model):
    def __init__(self, original_dim, latent_dim, num_layers=1, clip_log_var=False, **kwargs):
        super().__init__(**kwargs)
        self.original_dim = original_dim
        self.latent_dim = latent_dim
        self.clip_log_var = bool(clip_log_var)

        self.encoder = build_encoder(original_dim, latent_dim, num_layers)
        self.decoder = build_baseline_decoder(latent_dim, num_layers, original_dim)

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

    def encode(self, x):
        z = self.encoder(x)
        mu, log_var = tf.split(z, 2, axis=1)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        mu = tf.cast(mu, tf.float32)
        log_var = tf.cast(log_var, tf.float32)
        eps = tf.random.normal(tf.shape(mu), dtype=tf.float32)
        return mu + eps * tf.exp(0.5 * log_var)

    def decode(self, z):
        return self.decoder(z)

    def call(self, inputs, training=False):
        mu, log_var = self.encode(inputs)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        if training:
            return x_hat, mu, log_var
        return x_hat

    def train_step(self, data):
        x = data
        with tf.GradientTape() as tape:
            x_hat, mu, log_var = self(x, training=True)
            recon_loss = discrete_mse_loss(x, x_hat)
            kl = kl_divergence(mu, log_var, clip_log_var=self.clip_log_var)
            total_loss = tf.reduce_mean(recon_loss + kl)

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl)
        return {"loss": self.total_loss_tracker.result(),
                "recon_loss": self.recon_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result()}

    def test_step(self, data):
        x = data
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        recon_loss = discrete_mse_loss(x, x_hat)
        kl = kl_divergence(mu, log_var, clip_log_var=self.clip_log_var)
        total_loss = tf.reduce_mean(recon_loss + kl)
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl)
        return {"loss": self.total_loss_tracker.result(),
                "recon_loss": self.recon_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result()}


class BetaVAE(BaselineVAE):
    def __init__(self, original_dim, latent_dim, beta=4.0, num_layers=1, clip_log_var=False, **kwargs):
        super().__init__(original_dim, latent_dim, num_layers=num_layers, clip_log_var=clip_log_var, **kwargs)
        self.beta = float(beta)

    def train_step(self, data):
        x = data
        with tf.GradientTape() as tape:
            x_hat, mu, log_var = self(x, training=True)
            recon_loss = discrete_mse_loss(x, x_hat)
            kl = kl_divergence(mu, log_var, clip_log_var=self.clip_log_var)
            total_loss = tf.reduce_mean(recon_loss + self.beta * kl)

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl)
        return {"loss": self.total_loss_tracker.result(),
                "recon_loss": self.recon_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result()}

    def test_step(self, data):
        x = data
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        recon_loss = discrete_mse_loss(x, x_hat)
        kl = kl_divergence(mu, log_var, clip_log_var=self.clip_log_var)
        total_loss = tf.reduce_mean(recon_loss + self.beta * kl)
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl)
        return {"loss": self.total_loss_tracker.result(),
                "recon_loss": self.recon_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result()}


# ----------------------------------------------------------
# Metrics + Robustness
# ----------------------------------------------------------
def check_for_nan(data):
    if np.any(np.isnan(data)):
        print("[WARN] NaNs detected; replacing with 0.")
        data = np.nan_to_num(data)
    return data


def evaluate_mse(original_data: np.ndarray, reconstructed_data: np.ndarray) -> float:
    original_data = check_for_nan(original_data)
    reconstructed_data = check_for_nan(reconstructed_data)
    #original_data = np.clip(original_data, -1e10, 1e10)
    reconstructed_data = np.clip(reconstructed_data, 0, 2)
    return mean_squared_error(original_data, reconstructed_data)


def evaluate_r_square(original_data: np.ndarray, reconstructed_data: np.ndarray) -> float:
    original_data = check_for_nan(original_data)
    reconstructed_data = check_for_nan(reconstructed_data)
    original_data = np.clip(original_data, -1e10, 1e10)
    reconstructed_data = np.clip(reconstructed_data, -1e10, 1e10)
    ss_res = np.sum((original_data - reconstructed_data) ** 2)
    ss_tot = np.sum((original_data - np.mean(original_data)) ** 2)
    return  (ss_res / ss_tot) - 1

def r2_global_flat(X, Y) -> float:
    # sklearn global R² on the flattened matrix (should match your evaluate_r_square)
    return float(r2_score(X.reshape(-1), Y.reshape(-1)))

def r2_mean_per_snp(X, Y) -> float:
    # average of per-SNP R² across columns
    return float(r2_score(X, Y, multioutput="uniform_average"))

def r2_median_per_snp(X, Y) -> float:
    # distribution across SNPs is often heavy-tailed; median is robust
    per_snp = r2_score(X, Y, multioutput="raw_values")  # shape (n_snps,)
    return float(np.median(per_snp))

def make_tf_dataset(array: np.ndarray, batch_size: int, shuffle: bool = True):
    ds = tf.data.Dataset.from_tensor_slices(array)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(array), 10000), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


def _encode_mu_batches(vae, X: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    mus = []
    for i in range(0, X.shape[0], batch_size):
        xb = X[i:i + batch_size]
        mu, _ = vae.encode(tf.constant(xb))
        mus.append(mu.numpy().astype("float32"))
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

    num = np.linalg.norm(Zp - Z, axis=1)
    den = np.linalg.norm(Z, axis=1) + 1e-12
    rel = (num / den).astype(np.float32)

    ns_mean = float(np.mean(rel))
    ns_median = float(np.median(rel))
    robustness_inv = 1.0 / (ns_mean + 1e-12)
    return ns_mean, ns_median, robustness_inv


# ----------------------------------------------------------
# Training driver
# ----------------------------------------------------------
def train_vae_for_disease(
    disease_name: str,
    bed_prefix: str,
    latent_dim: int,
    num_epochs: int,
    batch_size: int,
    num_samples: int = 10,
    num_layers: int = 1,
    beta_values=None,
    clip_log_var: bool = False,
):
    print(f"[INFO] Loading BED prefix: {bed_prefix}")
    #data = load_bedd_as_float32(bed_prefix, mean_impute=True)
    #print(f"[INFO] Data shape (N individuals x M SNPs): {data.shape}")

    data, row_idx, col_idx, kept_snps = load_data_bed(
        bed_prefix=bed_prefix,
        downsample_n=None,
        feature_mode="random",
        downsample_d=50000,
        missing="mean",
        seed=42,
        return_indices=True
    )

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=SEED)

    def make_optimizer():
        return tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=1e-5,
                decay_steps=10_000,
                decay_rate=0.9,
            )
        )

    train_ds = make_tf_dataset(train_data, batch_size=batch_size, shuffle=True)
    val_ds = make_tf_dataset(test_data, batch_size=batch_size, shuffle=False)
    original_dim = train_data.shape[1]

    # -----------------------
    # 1) qgVAE
    # -----------------------
    qgvae = VAE(original_dim, latent_dim, num_samples=num_samples, num_layers=num_layers,
                clip_log_var=clip_log_var)
    qgvae.compile(optimizer=make_optimizer())

    qg_ckpt = f"{disease_name}_{latent_dim}_{num_samples}_{num_layers}_qgvae.weights.h5"
    qg_ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=qg_ckpt, save_weights_only=True, monitor="val_loss",
        mode="min", save_best_only=True, verbose=1
    )
    qg_early_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    print(f"[INFO] Train qgVAE: {disease_name} LD={latent_dim} K={num_samples} L={num_layers}")
    qg_history = qgvae.fit(
        train_ds, epochs=num_epochs, validation_data=val_ds,
        callbacks=[qg_ckpt_cb, qg_early_cb], verbose=2
    )
    qgvae.load_weights(qg_ckpt)

    qg_recon, _ = qgvae.predict(data, batch_size=batch_size, verbose=1)
    qg_recon = qg_recon.astype("float32")
    qg_r2 = evaluate_r_square(data, qg_recon)
    qg_mse = evaluate_mse(data, qg_recon)
    qg_r2_flat = r2_global_flat(data, qg_recon)    # should ~equal qg_r2
    qg_r2_snp_mean = r2_mean_per_snp(data, qg_recon)   # mean per-SNP R²
    qg_r2_snp_median = r2_median_per_snp(data, qg_recon)
    qg_ns_mean, qg_ns_median, qg_rob = compute_input_noise_robustness(
        qgvae, data, eps=0.05, max_n=5000, batch_size=batch_size
    )

    print(
    f"[qgVAE] R2_global={qg_r2:.4f} "
    f"MSE={qg_mse:.6f} NoiseSens_mean={qg_ns_mean:.6f} Robust={qg_rob:.6f}"
    )

    #print(f"[qgVAE] R2={qg_r2:.4f} f"R2_flat_sklearn={qg_r2_flat:.4f} " f"R2_snp_mean={qg_r2_snp_mean:.4f} " f"R2_snp_median={qg_r2_snp_median:.4f} " MSE={qg_mse:.6f} NoiseSens_mean={qg_ns_mean:.6f} Robust={qg_rob:.6f}")

    # -----------------------
    # 2) BaselineVAE (NS=1)
    # -----------------------
    baseline_ns = 1
    base_ckpt = f"{disease_name}_{latent_dim}_{baseline_ns}_{num_layers}_baseline_vae.weights.h5"

    baseline = BaselineVAE(original_dim, latent_dim, num_layers=num_layers, clip_log_var=clip_log_var)
    baseline.compile(optimizer=make_optimizer())

    base_early_cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    base_ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=base_ckpt, save_weights_only=True, monitor="val_loss",
        mode="min", save_best_only=True, verbose=1
    )

    if os.path.exists(base_ckpt):
        print(f"[INFO] Baseline weights exist; loading: {base_ckpt}")
        print(f"[INFO] Train BaselineVAE (NS=1)")
        baseline.fit(train_ds, epochs=num_epochs, validation_data=val_ds,
                     callbacks=[base_ckpt_cb, base_early_cb], verbose=2)
        baseline.load_weights(base_ckpt)
    else:
        print(f"[INFO] Train BaselineVAE (NS=1)")
        baseline.fit(train_ds, epochs=num_epochs, validation_data=val_ds,
                     callbacks=[base_ckpt_cb, base_early_cb], verbose=2)
        baseline.load_weights(base_ckpt)

    base_recon = baseline.predict(data, batch_size=batch_size, verbose=1).astype("float32")
    base_r2 = evaluate_r_square(data, base_recon)
    base_mse = evaluate_mse(data, base_recon)
    base_r2_flat = r2_global_flat(data, base_recon)
    base_r2_snp_mean = r2_mean_per_snp(data, base_recon)
    base_r2_snp_median = r2_median_per_snp(data, base_recon)
    base_ns_mean, base_ns_median, base_rob = compute_input_noise_robustness(
        baseline, data, eps=0.05, max_n=5000, batch_size=batch_size
    )

    print(
    f"[Baseline] R2_global={base_r2:.4f} "
    f"MSE={base_mse:.6f} NoiseSens_mean={base_ns_mean:.6f} Robust={base_rob:.6f}"
    )
    #print(f"[Baseline] R2={base_r2:.4f} MSE={base_mse:.6f} NoiseSens_mean={base_ns_mean:.6f} Robust={base_rob:.6f}")

    # -----------------------
    # 3) BetaVAE (NS=1)
    # -----------------------
    beta_rows = []
    for beta in beta_values:
        beta_tag = str(beta).replace(".", "p")
        beta_ckpt = f"{disease_name}_{latent_dim}_{baseline_ns}_{num_layers}_beta{beta_tag}_vae.weights.h5"

        betavae = BetaVAE(original_dim, latent_dim, beta=beta, num_layers=num_layers,
                          clip_log_var=clip_log_var)
        betavae.compile(optimizer=make_optimizer())

        beta_early_cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        beta_ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=beta_ckpt, save_weights_only=True, monitor="val_loss",
            mode="min", save_best_only=True, verbose=1
        )

        if os.path.exists(beta_ckpt):
            print(f"[INFO] BetaVAE beta={beta} weights exist; loading.")
            print(f"[INFO] Train BetaVAE beta={beta}")
            betavae.fit(train_ds, epochs=num_epochs, validation_data=val_ds,
                        callbacks=[beta_ckpt_cb, beta_early_cb], verbose=2)
            betavae.load_weights(beta_ckpt)
        else:
            print(f"[INFO] Train BetaVAE beta={beta}")
            betavae.fit(train_ds, epochs=num_epochs, validation_data=val_ds,
                        callbacks=[beta_ckpt_cb, beta_early_cb], verbose=2)
            betavae.load_weights(beta_ckpt)

        beta_recon = betavae.predict(data, batch_size=batch_size, verbose=1).astype("float32")
        beta_r2 = evaluate_r_square(data, beta_recon)
        beta_mse = evaluate_mse(data, beta_recon)
        beta_r2_flat = r2_global_flat(data, beta_recon)
        beta_r2_snp_mean = r2_mean_per_snp(data, beta_recon)
        beta_r2_snp_median = r2_median_per_snp(data, beta_recon)
        beta_ns_mean, beta_ns_median, beta_rob = compute_input_noise_robustness(
            betavae, data, eps=0.05, max_n=5000, batch_size=batch_size
        )
        print(
        f"[BetaVAE beta={beta}] R2_global={beta_r2:.4f} "
        f"MSE={beta_mse:.6f} NoiseSens_mean={beta_ns_mean:.6f} Robust={beta_rob:.6f}"
        )
        #print(f"[BetaVAE beta={beta}] R2={beta_r2:.4f} MSE={beta_mse:.6f} NoiseSens_mean={beta_ns_mean:.6f} Robust={beta_rob:.6f}")

        beta_rows.append({
            "model": f"BetaVAE_NS1_beta{beta}",
            "R2": beta_r2,
            "MSE": beta_mse,
            "NoiseSens_meanRelChange": beta_ns_mean,
            "NoiseSens_medianRelChange": beta_ns_median,
            "Robustness_invNoiseSens": beta_rob,
        })

    # Save per-run summary
    summary_rows = [
        {"model": f"qgVAE_NS{num_samples}", "R2": qg_r2, "MSE": qg_mse,
         "NoiseSens_meanRelChange": qg_ns_mean, "NoiseSens_medianRelChange": qg_ns_median,
         "Robustness_invNoiseSens": qg_rob},
        {"model": "BaselineVAE_NS1", "R2": base_r2, "MSE": base_mse,
         "NoiseSens_meanRelChange": base_ns_mean, "NoiseSens_medianRelChange": base_ns_median,
         "Robustness_invNoiseSens": base_rob},
    ] + beta_rows

    summary_df = pd.DataFrame(summary_rows)
    out_csv = f"{disease_name}_{latent_dim}_{num_samples}_{num_layers}_all_models_metrics.csv"
    summary_df.to_csv(out_csv, index=False)
    print(f"[INFO] Wrote: {out_csv}")

    return qg_r2, qg_mse, qg_rob, qg_history


# ----------------------------------------------------------
# Main / CLI
# ----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train qgVAE/Baseline/Beta directly from BED.")
    parser.add_argument("--disease", type=str, required=True)
    parser.add_argument("--num_sample", type=int, required=True)
    parser.add_argument("--latent_dim", type=int, required=True)
    parser.add_argument("--num_layer", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)

    # Bed input
    parser.add_argument("--bed_prefix", type=str, required=True,
                        help="Input prefix (no extension): expects .bed/.bim/.fam")

    # Optional stability
    parser.add_argument("--clip_log_var", action="store_true",
                        help="Clip log_var to [-10,10] to reduce KL blowups.")
    parser.add_argument("--no_mixed_precision", action="store_true",
                        help="Disable mixed precision (more stable, slower).")

    # Beta grid
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--beta_list", type=str, default="")

    args = parser.parse_args()

    if args.no_mixed_precision:
        # disable mixed precision
        mixed_precision.set_global_policy("float32")
        print("[INFO] Mixed precision disabled (policy=float32).")
    else:
        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_global_policy(policy)
        print("[INFO] Mixed precision enabled (policy=mixed_float16).")

    if args.beta_list.strip():
        betas = [float(b) for b in args.beta_list.split(",") if b.strip()]
    else:
        betas = [args.beta]

    # Skip guard for qgVAE weights
    #weights_file = f"{args.disease}_{args.latent_dim}_{args.num_sample}_{args.num_layer}_qgvae.weights.h5"
    #if os.path.exists(weights_file):
    #    print(f"[SKIP] Found weights: {weights_file}")
    #    with open("finished_jobs.txt", "a") as f:
    #        f.write(f"{args.disease}_{args.num_sample}_{args.latent_dim}_{args.num_layer}\n")
    #    return

    r2, mse, rob, _ = train_vae_for_disease(
        disease_name=args.disease,
        bed_prefix=args.bed_prefix,
        latent_dim=args.latent_dim,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_samples=args.num_sample,
        num_layers=args.num_layer,
        beta_values=betas,
        clip_log_var=args.clip_log_var,
    )

    with open("finished_jobs.txt", "a") as f:
        f.write(f"{args.disease}_{args.num_sample}_{args.latent_dim}_{args.num_layer}\n")

    print(f"[DONE] {args.disease}: qgVAE R2={r2:.4f} MSE={mse:.6f} Robust={rob:.6f}")


if __name__ == "__main__":
    main()

