from __future__ import annotations
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def reconstruction_metrics(X, Xhat):
    X = np.asarray(X)
    Xhat = np.asarray(Xhat)
    return {
        "R2": float(r2_score(X.reshape(-1), Xhat.reshape(-1))),
        "MSE": float(mean_squared_error(X.reshape(-1), Xhat.reshape(-1))),
    }

def latent_drift(z, z_perturbed, eps: float = 1e-12):
    z = np.asarray(z)
    zp = np.asarray(z_perturbed)
    rel = np.linalg.norm(zp - z, axis=1) / (np.linalg.norm(z, axis=1) + eps)
    return rel

def robustness_from_latent_drift(drift):
    drift = np.asarray(drift)
    return float(1.0 / (float(drift.mean()) + 1e-12))
