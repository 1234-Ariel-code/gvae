#!/usr/bin/env python3
import os
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras import layers, Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models, optimizers, regularizers
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import polars as pl
import shap
import argparse

#

# ------------------------------
# Repro, GPU, mixed precision
# ------------------------------
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# ------------------------------
# Sum of Shifted Sigmoid (0/1/2 bins)
# ------------------------------
class SumOfShiftedSigmoid(layers.Layer):
    def __init__(self, lambda_param=50.0, num_categories=2, **kwargs):
        super(SumOfShiftedSigmoid, self).__init__(**kwargs)
        self.lambda_param = lambda_param
        self.num_categories = num_categories  # K bins (0..num_categories)

    def call(self, inputs):
        x = tf.cast(inputs, dtype=tf.float32)
        category_shifts = tf.reshape(
            tf.range(self.num_categories + 1, dtype=tf.float32) + 0.5, (1, -1)
        )
        sum_sigmoids = tf.reduce_sum(
            tf.sigmoid(self.lambda_param * (tf.expand_dims(x, axis=-1) - category_shifts)),
            axis=-1
        )
        return tf.cast(sum_sigmoids, dtype=inputs.dtype)

# ------------------------------
# Model builders
# ------------------------------
def compute_initial_neurons(num_layers, latent_dim=32):
    return latent_dim * (2 ** (num_layers - 1))

def build_encoder(original_dim, latent_dim, num_layers):
    initial_neurons = compute_initial_neurons(num_layers, latent_dim)
    layers_list = [layers.InputLayer(input_shape=(original_dim,))]
    layers_list.append(layers.LayerNormalization())
    neurons = initial_neurons
    for _ in range(num_layers):
        layers_list.append(layers.Dense(neurons, activation='relu'))
        neurons //= 2
    layers_list.append(layers.Dense(latent_dim * 2))  # mu and log_var
    return tf.keras.Sequential(layers_list)

def buuild_decoder(latent_dim, num_layers, original_dim, K=100):
    _ = compute_initial_neurons(num_layers, latent_dim)  # not used directly
    layers_list = [layers.InputLayer(input_shape=(K * latent_dim,))]
    neurons = latent_dim
    for _ in range(num_layers):
        layers_list.append(layers.Dense(neurons, activation='relu'))
        neurons *= 2
    # Quantization-ish activation before final linear head
    layers_list.append(SumOfShiftedSigmoid(lambda_param=50.0, num_categories=2))
    layers_list.append(layers.Dense(original_dim))
    return tf.keras.Sequential(layers_list)

def build_decoder(latent_dim, num_layers, original_dim, input_dim):
    layers_list = [layers.InputLayer(input_shape=(input_dim,))]
    neurons = latent_dim
    for _ in range(num_layers):
        layers_list.append(layers.Dense(neurons, activation='relu'))
        neurons *= 2
    layers_list.append(SumOfShiftedSigmoid(lambda_param=50.0, num_categories=2))
    layers_list.append(layers.Dense(original_dim))
    return tf.keras.Sequential(layers_list)


# VAE
class VAE(tf.keras.Model):
    def __init__(self, original_dim, latent_dim, num_samples=100, num_layers=1):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_samples = num_samples

        # Build encoder
        self.encoder = build_encoder(original_dim, latent_dim, num_layers)

        # Build decoder
        self.decoder_continuous = build_decoder(latent_dim, num_layers, original_dim, input_dim=2 * latent_dim)

    def encode(self, x):
        z = self.encoder(x)
        mu, log_var = tf.split(z, num_or_size_splits=2, axis=1)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        mu = tf.cast(mu, dtype=tf.float16)
        log_var = tf.cast(log_var, dtype=tf.float16)
        z_samples = []
        for _ in range(self.num_samples):
            eps = tf.random.normal(shape=(tf.shape(mu)[0], self.latent_dim), dtype=tf.float16)
            z = mu + eps * tf.exp(0.5 * log_var)
            z_samples.append(z)
        return z_samples

    def decode(self, z):
        output = self.decoder_continuous(z)
        return output

    def compute_row_wise_quantiles(self, z_samples):
        # Sort z_samples across the num_samples axis (axis=0)
        sorted_samples = tf.sort(z_samples, axis=0)
        # Compute the indices for 25th and 75th percentiles
        num_samples = tf.shape(sorted_samples)[0]
        idx_25 = tf.cast(0.25 * tf.cast(num_samples, tf.float32), tf.int32)
        idx_75 = tf.cast(0.75 * tf.cast(num_samples, tf.float32), tf.int32)
        # Extract the 25th and 75th percentiles using tf.gather
        quantile_25 = tf.gather(sorted_samples, idx_25, axis=0)
        quantile_75 = tf.gather(sorted_samples, idx_75, axis=0)
        # Stack the quantiles together (no need for Python loops)
        quantiles_combined = tf.concat([quantile_25, quantile_75], axis=-1)  # Shape: (num_individuals, 2 * latent_dim)
        return quantiles_combined

    def call(self, inputs, training=False):
        mu, log_var = self.encode(inputs)
        z_samples = self.reparameterize(mu, log_var)
        # Get the combined quantiles (already concatenated)
        quantiles_combined = self.compute_row_wise_quantiles(z_samples)
        # No need to unpack anymore; just use quantiles_combined directly
        z_quantiles = quantiles_combined
        output = self.decode(z_quantiles)
        if training:
           return output, mu, log_var, z_quantiles
        return output, z_quantiles, z_samples

# ------------------------------
# Data IO
# ------------------------------
def load_and_preprocess_data(file_path, separator=","):
    df = pl.read_csv(
        file_path,
        has_header=False,
        separator=separator,
        null_values=["9", "NA", "NaN"],
        infer_schema_length=1000,
    )
    df = df.with_columns(pl.all().cast(pl.Float32))
    arr = (
        df.fill_null(strategy="mean")
          .transpose(include_header=False)
          .to_numpy()
    )
    return arr  # shape (N, D)


def _read_snp_ids_from_tped(tped_file: str) -> list[str]:
    """
    TPED format: chr  snp_id  genetic_dist  bp  a1_i a2_i a1_i a2_i ...
    We only need the snp_id column (2nd column).
    """
    snp_ids = []
    with open(tped_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            # parts[1] is SNP ID
            snp_ids.append(parts[1])
    return snp_ids


def _read_snp_ids_from_bim(bim_file: str) -> list[str]:
    """
    BIM format: chr snp_id cm bp a1 a2
    SNP ID is 2nd column.
    """
    snp_ids = []
    with open(bim_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            snp_ids.append(parts[1])
    return snp_ids


def _load_gwas_assoc(assoc_path: str) -> np.ndarray:
    """
    Load a PLINK .assoc file and return structured arrays of (SNP, P).
    Expected columns include SNP and P.
    """
    # Polars reads whitespace-separated via separator=" " but multiple spaces can be messy.
    # We'll read with python for robustness.
    snps = []
    ps = []
    with open(assoc_path, "r") as f:
        header = f.readline().strip().split()
        if not header:
            raise ValueError(f"Empty GWAS file: {assoc_path}")

        # Find column indices
        try:
            snp_i = header.index("SNP")
        except ValueError:
            # some plink outputs use "ID" or similar, but SNP is standard
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

    snps = np.array(snps, dtype=str)
    ps = np.array(ps, dtype=float)
    return snps, ps


def load_data(
    file_path: str,
    separator: str = ",",
    # ---- row downsampling (individuals)
    downsample_n: int | None = None,
    downsample_frac: float | None = None,
    # ---- feature downsampling (SNPs)
    feature_mode: str | None = None,   # None | "random" | "gwas_top"
    downsample_d: int | None = None,
    # GWAS inputs
    gwas_assoc_path: str | None = None,
    # SNP-ID mapping
    tped_file: str | None = None,
    bim_file: str | None = None,
    # misc
    seed: int = 42,
    return_indices: bool = False,
):
    """
    Returns:
      arr: (N_down, D_down) float32
      optionally (row_idx, col_idx, kept_snp_ids)
    """

    df = pl.read_csv(
        file_path,
        has_header=False,
        separator=separator,
        null_values=["9", "NA", "NaN"],
        infer_schema_length=1000,
    ).with_columns(pl.all().cast(pl.Float32))

    # two candidate orientations
    arr_noT = df.fill_null(strategy="mean").to_numpy()
    arr_T   = df.fill_null(strategy="mean").transpose(include_header=False).to_numpy()

    # Decide orientation
    chosen = None
    snp_ids = None

    if tped_file is not None:
        snp_ids = _read_snp_ids_from_tped(tped_file)
        M = len(snp_ids)

        # we want D == M
        if arr_noT.shape[1] == M:
            chosen = arr_noT
        elif arr_T.shape[1] == M:
            chosen = arr_T
        else:
            raise ValueError(
                f"[ORIENTATION ERROR] Neither orientation matches TPED SNP count.\n"
                f"TPED SNPs: {M}\n"
                f"arr_noT shape: {arr_noT.shape} (D={arr_noT.shape[1]})\n"
                f"arr_T   shape: {arr_T.shape} (D={arr_T.shape[1]})\n"
                f"Your CSV must contain exactly the same SNP set as the TPED."
            )
    else:
        # fallback: assume your original convention (transpose)
        chosen = arr_T

    arr = chosen.astype(np.float32, copy=False)
    N, D = arr.shape

    rng = np.random.default_rng(seed)
    row_idx = np.arange(N)
    col_idx = np.arange(D)
    kept_snp_ids = None

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
        pass

    elif feature_mode == "random":
        if downsample_d is None:
            raise ValueError("feature_mode='random' requires downsample_d.")
        target_d = min(D, int(downsample_d))
        if target_d < D:
            col_idx = rng.choice(D, size=target_d, replace=False)
            col_idx.sort()
            arr = arr[:, col_idx]
            if snp_ids is not None:
                kept_snp_ids = np.array(snp_ids, dtype=str)[col_idx].tolist()

    elif feature_mode == "gwas_top":
        if downsample_d is None:
            raise ValueError("feature_mode='gwas_top' requires downsample_d.")
        if gwas_assoc_path is None:
            raise ValueError("feature_mode='gwas_top' requires gwas_assoc_path.")
        if snp_ids is None:
            if tped_file is not None:
                snp_ids = _read_snp_ids_from_tped(tped_file)
            elif bim_file is not None:
                snp_ids = _read_snp_ids_from_bim(bim_file)
            else:
                raise ValueError("Need tped_file or bim_file for feature_mode='gwas_top'.")

        if len(snp_ids) != D:
            raise ValueError(
                f"SNP-ID count ({len(snp_ids)}) does not match D ({D}). "
                f"After orientation, CSV columns must align to TPED/BIM SNP IDs."
            )

        gwas_snps, gwas_p = _load_gwas_assoc(gwas_assoc_path)
        order = np.argsort(gwas_p)
        topM = int(min(len(order), downsample_d))
        gwas_top_snps = set(gwas_snps[order[:topM]].tolist())

        snp_ids_arr = np.array(snp_ids, dtype=str)
        mask = np.isin(snp_ids_arr, list(gwas_top_snps))

        if mask.sum() == 0:
            raise RuntimeError(
                f"No overlap between GWAS top {topM} SNPs and genotype columns.\n"
                f"Likely SNP ID mismatch (rsIDs vs chr:pos, etc.)."
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


# ------------------------------
# Losses / metrics
# ------------------------------
def check_for_nan(data):
    if np.any(np.isnan(data)):
        print("Warning: NaN values detected in the data! Replacing with zeros.")
        data = np.nan_to_num(data)
    return data

def evaluate_mse(original_data, reconstructed_data):
    original_data = check_for_nan(original_data)
    reconstructed_data = check_for_nan(reconstructed_data)
    original_data = np.clip(original_data, -1e10, 1e10)
    reconstructed_data = np.clip(reconstructed_data, -1e10, 1e10)
    return mean_squared_error(original_data, reconstructed_data)

def evaluate_r_square(original_data, reconstructed_data):
    original_data = check_for_nan(original_data)
    reconstructed_data = check_for_nan(reconstructed_data)
    original_data = np.clip(original_data, -1e10, 1e10)
    reconstructed_data = np.clip(reconstructed_data, -1e10, 1e10)
    ss_res = np.sum((original_data - reconstructed_data) ** 2)
    ss_tot = np.sum((original_data - np.mean(original_data)) ** 2)
    return 1 - (ss_res / ss_tot)

def mse_loss(original, predictions):
    original = tf.cast(original, dtype=tf.float16)
    predictions = tf.cast(predictions, dtype=tf.float16)
    dist_to_0 = tf.cast(tf.square(predictions - 0), dtype=tf.float32)
    dist_to_1 = tf.cast(tf.square(predictions - 1), dtype=tf.float32)
    dist_to_2 = tf.cast(tf.square(predictions - 2), dtype=tf.float32)
    mask_0 = tf.cast(tf.equal(original, 0), tf.float32)
    mask_1 = tf.cast(tf.equal(original, 1), tf.float32)
    mask_2 = tf.cast(tf.equal(original, 2), tf.float32)
    mse_comp = (mask_0 * dist_to_0) + (mask_1 * dist_to_1) + (mask_2 * dist_to_2)
    recon = tf.reduce_mean(tf.square(predictions - original))
    recon = tf.cast(recon, dtype=tf.float32)
    return tf.reduce_mean(mse_comp) + recon

def compute_vae_loss(model, x):
    recon, mu, log_var, _ = model(x, training=True)
    kl = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=-1)
    kl = tf.cast(kl, dtype=tf.float32)
    recon_loss = mse_loss(x, recon)
    return tf.reduce_mean(recon_loss + kl)

# ------------------------------
# SHAP helpers (streaming, per-draw)
# ------------------------------
def _safe_shap_values(explainer, X_chunk):
    """Robustly get SHAP values across shap versions."""
    out = explainer(X_chunk)
    # shap.Explanation -> .values; older versions may return np.ndarray directly
    if hasattr(out, "values"):
        return out.values
    return np.asarray(out)

def _topk_per_latent_given_targets(
    data, targets, snp_names, top_k, chunk_size, out_prefix, output_dir
):
    """
    data: (N, D) genotype matrix
    targets: (N, LD) target latent matrix for a given draw (or mu)
    snp_names: list length D
    Saves:
      - {out_prefix}_top_snps_per_latent.csv/.txt
      - {out_prefix}_SHAP_weighted_matrix.csv
    Returns (paths, shape)
    """
    N, D = data.shape
    LD = targets.shape[1]
    os.makedirs(output_dir, exist_ok=True)

    top_snp_records = []
    extended_blocks = []

    # Preselect a small background (cap at N)
    bg_size = min(100, N)
    bg_idx = np.random.choice(N, size=bg_size, replace=False)
    background = data[bg_idx]

    for i in range(LD):
        print(f"  · Latent dim {i+1}/{LD}")
        y = targets[:, i]
        model = LinearRegression()
        model.fit(data, y)

        explainer = shap.Explainer(model, background)

        mean_abs = np.zeros(D, dtype=np.float64)
        n_chunks = int(np.ceil(N / chunk_size))
        for j in range(n_chunks):
            s, e = j * chunk_size, min((j + 1) * chunk_size, N)
            Xc = data[s:e]
            sv = _safe_shap_values(explainer, Xc)  # (chunk, D)
            mean_abs += np.abs(sv).sum(axis=0)
        mean_abs /= float(N)

        # Top-k features
        k = min(top_k, D)
        top_idx = np.argsort(mean_abs)[-k:]

        # Build weighted block (N, k)
        block_cols = []
        for idx in top_idx:
            w = mean_abs[idx]
            col = data[:, idx][:, None] * w
            block_cols.append(col)
            top_snp_records.append({
                "Latent_Dim": f"LD_{i}",
                "SNP_ID": snp_names[idx],
                "SHAP_Importance": w
            })
        block = np.hstack(block_cols) if block_cols else np.empty((N, 0))
        extended_blocks.append(block)

    extended_matrix = np.hstack(extended_blocks) if extended_blocks else np.empty((N, 0))
    extended_df = pd.DataFrame(extended_matrix)
    mat_path = os.path.join(output_dir, f"{out_prefix}_SHAP_weighted_matrix.csv")
    extended_df.to_csv(mat_path, index=False)

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
        "matrix_shape": extended_matrix.shape
    }

def compute_snp_contributions_topk_per_latent_all_draws(
    data,
    z_samples,
    disease_name,
    latent_dim,
    num_samples,
    num_layers,
    top_k=500,
    tped_file=None,
    chunk_size=500,
    output_dir="."
):
    """
    Compute per-draw (per latent sample) SHAP top-k SNPs per latent.
    Writes one set of files per draw: ..._S{draw}_(top_snps|matrix).csv.

    Returns: manifest (pd.DataFrame) describing outputs per draw.
    """
    assert tped_file is not None and os.path.exists(tped_file), "Valid .tped path required"
    os.makedirs(output_dir, exist_ok=True)

    # SNP ids from .tped: column 2 is SNP name (0-based index 1)
    snp_names = pd.read_csv(tped_file, delim_whitespace=True, header=None)[1].tolist()

    # Normalize z_samples to list of np arrays (N, LD)
    z_draws = []
    for z in z_samples:
        if isinstance(z, tf.Tensor):
            z_draws.append(z.numpy())
        else:
            z_draws.append(np.asarray(z))
    # Sanity
    if len(z_draws) != num_samples:
        print(f"Warning: z_samples length {len(z_draws)} != num_samples {num_samples}")

    manifest_rows = []
    for s, targets in enumerate(z_draws, start=1):
        out_prefix = f"{disease_name}_LD{latent_dim}_NS{num_samples}_L{num_layers}_K{top_k}_S{s}"
        print(f"\nProcessing draw S={s}/{len(z_draws)} ...")
        paths = _topk_per_latent_given_targets(
            data=data,
            targets=targets,
            snp_names=snp_names,
            top_k=top_k,
            chunk_size=chunk_size,
            out_prefix=out_prefix,
            output_dir=output_dir
        )
        manifest_rows.append({
            "draw": s,
            **paths
        })

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(
        output_dir,
        f"{disease_name}_LD{latent_dim}_NS{num_samples}_L{num_layers}_K{top_k}_SHAP_outputs_manifest.csv"
    )
    manifest.to_csv(manifest_path, index=False)
    print(f"\n>> Manifest saved: {manifest_path}")
    return manifest

# ------------------------------
# Training wrapper
# ------------------------------
def train_vae_for_disease(
    disease_name,
    base_path,
    latent_dim,
    num_epochs,
    batch_size,
    num_samples=100,
    num_layers=1,
    top_k=500,
    tped_file=None,
    output_dir="."
):

    assoc_path = f"/work/long_lab/for_Ariel/gwas_results/{disease_name}_gwas.assoc"
    tped_path  = f"{base_path}/{disease_name}_origin.tped"
    file_path = f"{base_path}/{disease_name}_filtered.csv"

    data = load_data(
    file_path=file_path,
    separator=",",
    downsample_n=None,
    feature_mode="gwas_top",
    downsample_d=50000,          # keep top 50k SNPs by GWAS p-value
    gwas_assoc_path=assoc_path,
    tped_file=tped_path,
    seed=42,
    )

    #file_path = f"{base_path}/{disease_name}_filtered.csv"
    #data = load_and_preprocess_data(file_path)
    print(f"Training {disease_name} | LD={latent_dim} | L={num_layers} | K={num_samples}")
    print(f"Data shape: {data.shape}")

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=seed)
    original_dim = train_data.shape[1]

    vae = VAE(original_dim, latent_dim, num_samples, num_layers)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4, decay_steps=10000, decay_rate=0.9
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # Important: compute loss via closure so KL sees model params
    vae.compile(optimizer=optimizer, loss=lambda x, y: compute_vae_loss(vae, x))

    history = vae.fit(
        train_data, train_data,
        epochs=num_epochs, batch_size=batch_size,
        validation_data=(test_data, test_data),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{disease_name}_continuous_{latent_dim}_{num_samples}_{num_layers}.weights.h5",
                save_weights_only=True, monitor='val_loss', mode='min',
                save_best_only=True, verbose=1
            )
        ],
        verbose=1
    )

    # Predict BEFORE clearing session
    output, sorted_samples, z_samples = vae.predict(data, batch_size=batch_size, verbose=0)

    # Now safe to clear graph memory if desired
    tf.keras.backend.clear_session()

    # Metrics
    r2 = evaluate_r_square(data, output)
    mse = evaluate_mse(data, output)
    if np.isnan(r2):
        print(f"Skipping: NaN R² for {disease_name} (LD={latent_dim}, L={num_layers}, K={num_samples})")
        return None, None

    print(f"R²={r2:.4f} | MSE={mse:.4f} | Disease={disease_name} | LD={latent_dim} | L={num_layers} | K={num_samples}")

    # Save flattened sorted samples (N, K*LD)
    rep_path = f"rep_{disease_name}_LD{latent_dim}_NS{num_samples}_L{num_layers}.csv"
    pd.DataFrame(sorted_samples).to_csv(rep_path, index=False)
    print(f"Saved latent (sorted/flattened) → {rep_path}")

    # ---- Per-draw SHAP outputs ----
    if tped_file is None:
        tped_file = f"/work/long_lab/for_Ariel/{disease_name}_origin.tped"
    manifest = compute_snp_contributions_topk_per_latent_all_draws(
        data=data,
        z_samples=z_samples,
        disease_name=disease_name,
        latent_dim=latent_dim,
        num_samples=num_samples,
        num_layers=num_layers,
        top_k=top_k,
        tped_file=tped_file,
        chunk_size=500,
        output_dir=output_dir
    )

    return r2, history

# ------------------------------
# CLI
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--disease', type=str, required=True)
    parser.add_argument('--latent_dim', type=int, required=True)
    parser.add_argument('--num_layers', type=int, required=True)
    parser.add_argument('--num_samples', type=int, required=True)
    parser.add_argument('--base_path', type=str, default="/work/long_lab/for_Ariel/files")
    parser.add_argument('--tped_file', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=".")
    parser.add_argument('--top_k', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    r_square, history = train_vae_for_disease(
        disease_name=args.disease,
        base_path=args.base_path,
        latent_dim=args.latent_dim,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        num_layers=args.num_layers,
        top_k=args.top_k,
        tped_file=args.tped_file,
        output_dir=args.output_dir
    )

