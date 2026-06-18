#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared evaluation metrics for the gVAE repository.

This module is intentionally small and is imported by both the model-training
and SNP-prioritization pipelines so that reconstruction metrics are defined in
one place only.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def check_for_nan(data: np.ndarray) -> np.ndarray:
    """Replace NaN values with zero before metric evaluation."""
    if np.any(np.isnan(data)):
        print("[WARN] NaNs detected; replacing with 0 before metric evaluation.")
        data = np.nan_to_num(data)
    return data


def evaluate_mse(original_data: np.ndarray, reconstructed_data: np.ndarray) -> float:
    """Mean squared reconstruction error."""
    original_data = check_for_nan(original_data)
    reconstructed_data = check_for_nan(reconstructed_data)
    original_data = np.clip(original_data, -1e10, 1e10)
    reconstructed_data = np.clip(reconstructed_data, -1e10, 1e10)
    return float(mean_squared_error(original_data, reconstructed_data))


def evaluate_r_square(original_data: np.ndarray, reconstructed_data: np.ndarray) -> float:
    """
    Global reconstruction R² computed on the full genotype matrix.

    This is equivalent to applying the standard R² definition to the flattened
    matrix and is the metric used throughout the reviewer-facing workflow.
    """
    original_data = check_for_nan(original_data)
    reconstructed_data = check_for_nan(reconstructed_data)
    original_data = np.clip(original_data, -1e10, 1e10)
    reconstructed_data = np.clip(reconstructed_data, -1e10, 1e10)

    ss_res = np.sum((original_data - reconstructed_data) ** 2)
    ss_tot = np.sum((original_data - np.mean(original_data)) ** 2)
    if ss_tot <= 0:
        return float("nan")
    return float(1.0 - (ss_res / ss_tot))


def r2_global_flat(original_data: np.ndarray, reconstructed_data: np.ndarray) -> float:
    """Global R² using sklearn on flattened matrices."""
    return float(r2_score(original_data.reshape(-1), reconstructed_data.reshape(-1)))


def r2_mean_per_snp(original_data: np.ndarray, reconstructed_data: np.ndarray) -> float:
    """Mean of per-SNP R² values."""
    return float(r2_score(original_data, reconstructed_data, multioutput="uniform_average"))


def r2_median_per_snp(original_data: np.ndarray, reconstructed_data: np.ndarray) -> float:
    """Median of per-SNP R² values."""
    per_snp = r2_score(original_data, reconstructed_data, multioutput="raw_values")
    return float(np.median(per_snp))
