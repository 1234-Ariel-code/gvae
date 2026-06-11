from __future__ import annotations
import numpy as np
import pandas as pd

def simple_snp_importance(X, Z, snp_ids, top_k: int = 100):
    X = np.asarray(X)
    Z = np.asarray(Z)
    rows = []
    for j in range(Z.shape[1]):
        z = Z[:, j]
        corrs = []
        for i in range(X.shape[1]):
            xi = X[:, i]
            if np.std(xi) == 0 or np.std(z) == 0:
                c = 0.0
            else:
                c = np.corrcoef(xi, z)[0, 1]
                c = 0.0 if not np.isfinite(c) else c
            corrs.append(abs(c))
        idx = np.argsort(corrs)[-min(top_k, len(corrs)):]
        for ii in idx[::-1]:
            rows.append({"Latent_Dim": f"LV_{j}", "SNP_ID": snp_ids[ii], "Importance": float(corrs[ii])})
    return pd.DataFrame(rows)
