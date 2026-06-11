from __future__ import annotations
import numpy as np
import pandas as pd
import tensorflow as tf
from .models import build_baseline_vae, build_beta_vae, build_gvae
from .metrics import reconstruction_metrics, latent_drift, robustness_from_latent_drift

def _dataset(X, batch_size: int = 64):
    ds = tf.data.Dataset.from_tensor_slices(X.astype(np.float32))
    return ds.shuffle(min(len(X), 10000), seed=42).batch(batch_size).prefetch(tf.data.AUTOTUNE)

def _extract_features(model, X, model_name: str):
    if model_name == "gVAE":
        _, _, _, z = model(X.astype(np.float32), training=False)
        return z.numpy()
    mu, _ = model.encode(X.astype(np.float32))
    return mu.numpy()

def train_models_for_config(X, latent_dim: int, num_samples: int, depth: int, beta_values: list[float], epochs: int, batch_size: int):
    results = []
    artifacts = {}
    configs = [("BaselineVAE", build_baseline_vae(X.shape[1], latent_dim, depth)), ("gVAE", build_gvae(X.shape[1], latent_dim, depth, num_samples))]
    for beta in beta_values:
        configs.append((f"BetaVAE_beta{beta}", build_beta_vae(X.shape[1], latent_dim, depth, beta)))
    ds = _dataset(X, batch_size=batch_size)
    for name, model in configs:
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
        model.fit(ds, epochs=epochs, verbose=0)
        Xhat = model(X.astype(np.float32), training=False)[0].numpy()
        feats = _extract_features(model, X, "gVAE" if name == "gVAE" else "other")
        noise = np.random.normal(0, 0.05, size=X.shape).astype(np.float32)
        Xp = np.clip(X + noise, 0, 2)
        feats_p = _extract_features(model, Xp, "gVAE" if name == "gVAE" else "other")
        rec = reconstruction_metrics(X, Xhat)
        drift = latent_drift(feats, feats_p)
        rob = robustness_from_latent_drift(drift)
        results.append({"model": name, **rec, "Robustness": rob})
        artifacts[name] = {"model": model, "features": feats, "recon": Xhat}
    return pd.DataFrame(results), artifacts
