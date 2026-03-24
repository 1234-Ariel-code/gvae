from __future__ import annotations
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LogisticRegression, Ridge
from scipy.stats import pearsonr

def run_binary_prediction(Z, y, seed: int = 42):
    Xtr, Xte, ytr, yte = train_test_split(Z, y, test_size=0.2, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    pred = (p >= 0.5).astype(int)
    return {"AUC": float(roc_auc_score(yte, p)), "Accuracy": float(accuracy_score(yte, pred))}

def run_quant_prediction(Z, y, seed: int = 42):
    Xtr, Xte, ytr, yte = train_test_split(Z, y, test_size=0.2, random_state=seed)
    reg = Ridge(alpha=1.0)
    reg.fit(Xtr, ytr)
    pred = reg.predict(Xte)
    corr = pearsonr(yte, pred)[0] if np.std(pred) > 0 and np.std(yte) > 0 else np.nan
    return {
        "R2": float(r2_score(yte, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(yte, pred))),
        "MAE": float(mean_absolute_error(yte, pred)),
        "Correlation": float(corr) if np.isfinite(corr) else np.nan,
    }
