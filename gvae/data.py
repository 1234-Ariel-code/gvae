from __future__ import annotations
import json, os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

ALLELE_PAIRS = [("A", "G"), ("C", "T"), ("G", "A"), ("T", "C")]


def simulate_genotype_data(n_samples: int, n_snps: int, n_causal: int, task: str, seed: int = 42):
    rng = np.random.default_rng(seed)
    maf = rng.uniform(0.05, 0.5, size=n_snps)
    X = rng.binomial(2, maf, size=(n_samples, n_snps)).astype(np.float32)
    causal_idx = rng.choice(n_snps, size=min(n_causal, n_snps), replace=False)
    beta = rng.normal(0, 1, size=len(causal_idx))
    score = X[:, causal_idx] @ beta
    score = (score - score.mean()) / (score.std() + 1e-8)
    if task == "binary":
        logits = 0.9 * score + rng.normal(0, 0.7, size=n_samples)
        probs = 1.0 / (1.0 + np.exp(-logits))
        y = rng.binomial(1, probs).astype(np.int32)
    elif task == "quantitative":
        y = (0.9 * score + rng.normal(0, 0.8, size=n_samples)).astype(np.float32)
    else:
        raise ValueError("task must be 'binary' or 'quantitative'")
    snp_ids = [f"rsSIM{i+1}" for i in range(n_snps)]
    return X, y, snp_ids, causal_idx


def _make_sample_ids(n_samples: int) -> Tuple[List[str], List[str]]:
    fids = [f"FAM{i+1:04d}" for i in range(n_samples)]
    iids = [f"ID{i+1:04d}" for i in range(n_samples)]
    return fids, iids


def _make_bim_df(snp_ids: List[str]) -> pd.DataFrame:
    rows = []
    for j, snp in enumerate(snp_ids):
        chrom = str((j % 22) + 1)
        bp = 100000 + j * 10
        a1, a2 = ALLELE_PAIRS[j % len(ALLELE_PAIRS)]
        rows.append([chrom, snp, 0, bp, a1, a2])
    return pd.DataFrame(rows, columns=["CHR", "SNP", "CM", "BP", "A1", "A2"])


def _make_fam_df(y: np.ndarray, task: str, fids: List[str], iids: List[str]) -> pd.DataFrame:
    sex = [1 if i % 2 == 0 else 2 for i in range(len(iids))]
    if task == "binary":
        pheno = [2 if int(v) == 1 else 1 for v in y]
    else:
        pheno = [float(v) for v in y]
    return pd.DataFrame({
        "FID": fids,
        "IID": iids,
        "PID": [0] * len(iids),
        "MID": [0] * len(iids),
        "SEX": sex,
        "PHENO": pheno,
    })


def _write_phen_file(path: str, y: np.ndarray, task: str, fids: List[str], iids: List[str]) -> None:
    if task == "binary":
        pheno = [2 if int(v) == 1 else 1 for v in y]
    else:
        pheno = [float(v) for v in y]
    pd.DataFrame({"FID": fids, "IID": iids, "PHENO": pheno}).to_csv(path, sep=" ", header=False, index=False)


def _write_tped_tfam(prefix: str, X: np.ndarray, snp_ids: List[str], y: np.ndarray, task: str, fids: List[str], iids: List[str]) -> None:
    bim = _make_bim_df(snp_ids)
    tfam = _make_fam_df(y, task, fids, iids)
    tfam.to_csv(prefix + ".tfam", sep=" ", header=False, index=False)

    with open(prefix + ".tped", "w", encoding="utf-8") as f:
        for j, row in bim.iterrows():
            a1, a2 = row["A1"], row["A2"]
            geno_tokens: List[str] = []
            for g in X[:, j].astype(int):
                if g <= 0:
                    geno_tokens.extend([a2, a2])
                elif g == 1:
                    geno_tokens.extend([a1, a2])
                else:
                    geno_tokens.extend([a1, a1])
            fields = [str(row["CHR"]), str(row["SNP"]), "0", str(int(row["BP"]))] + geno_tokens
            f.write(" ".join(fields) + "\n")


def _pack_plink_genotypes(genotypes_a1_count: np.ndarray) -> bytearray:
    n = int(len(genotypes_a1_count))
    out = bytearray()
    for start in range(0, n, 4):
        byte = 0
        for offset, g in enumerate(genotypes_a1_count[start:start + 4].astype(int)):
            code = 0b11
            if g >= 2:
                code = 0b00
            elif g == 1:
                code = 0b10
            elif g <= 0:
                code = 0b11
            byte |= (code << (2 * offset))
        out.append(byte)
    return out


def _write_bed(prefix: str, X: np.ndarray) -> None:
    with open(prefix + ".bed", "wb") as f:
        f.write(bytes([0x6C, 0x1B, 0x01]))
        for j in range(X.shape[1]):
            f.write(_pack_plink_genotypes(X[:, j]))


def _write_plink_files(prefix: str, X: np.ndarray, y: np.ndarray, snp_ids: List[str], task: str, fids: List[str], iids: List[str]) -> None:
    bim = _make_bim_df(snp_ids)
    fam = _make_fam_df(y, task, fids, iids)
    bim.to_csv(prefix + ".bim", sep="	", header=False, index=False)
    fam.to_csv(prefix + ".fam", sep=" ", header=False, index=False)
    _write_bed(prefix, X)


def _per_snp_assoc_binary(X: np.ndarray, y: np.ndarray, snp_ids: List[str]) -> pd.DataFrame:
    bim = _make_bim_df(snp_ids)
    rows = []
    y = y.astype(float)
    for j, snp in enumerate(snp_ids):
        x = X[:, j].astype(float)
        try:
            slope, intercept, r, p, stderr = stats.linregress(x, y)
            or_est = float(np.exp(slope))
            stat = 0.0 if stderr in (0, None) or np.isnan(stderr) else float(slope / stderr)
        except Exception:
            slope, p, stderr, stat, or_est = 0.0, 1.0, np.nan, 0.0, 1.0
        rows.append([bim.loc[j, "CHR"], snp, bim.loc[j, "BP"], bim.loc[j, "A1"], "ADD", int(len(y)), slope, stat, max(float(p), np.nextafter(0, 1)), or_est])
    return pd.DataFrame(rows, columns=["CHR", "SNP", "BP", "A1", "TEST", "NMISS", "BETA", "STAT", "P", "OR"])


def _per_snp_assoc_quant(X: np.ndarray, y: np.ndarray, snp_ids: List[str]) -> pd.DataFrame:
    bim = _make_bim_df(snp_ids)
    rows = []
    y = y.astype(float)
    for j, snp in enumerate(snp_ids):
        x = X[:, j].astype(float)
        try:
            slope, intercept, r, p, stderr = stats.linregress(x, y)
            tstat = 0.0 if stderr in (0, None) or np.isnan(stderr) else float(slope / stderr)
            r2 = float(r * r)
        except Exception:
            slope, p, stderr, tstat, r2 = 0.0, 1.0, np.nan, 0.0, 0.0
        rows.append([bim.loc[j, "CHR"], snp, bim.loc[j, "BP"], int(len(y)), slope, 0.0 if pd.isna(stderr) else float(stderr), r2, tstat, max(float(p), np.nextafter(0, 1))])
    return pd.DataFrame(rows, columns=["CHR", "SNP", "BP", "NMISS", "BETA", "SE", "R2", "T", "P"])


def _write_assoc_files(out_dir: str, dataset_name: str, X: np.ndarray, y: np.ndarray, snp_ids: List[str], task: str) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    if task == "binary":
        assoc = _per_snp_assoc_binary(X, y, snp_ids)
        p = os.path.join(out_dir, f"{dataset_name}_gwas.assoc")
        assoc.to_csv(p, sep="	", index=False)
        paths["assoc"] = p
    else:
        qassoc = _per_snp_assoc_quant(X, y, snp_ids)
        p = os.path.join(out_dir, f"{dataset_name}_gwas.qassoc")
        qassoc.to_csv(p, sep="	", index=False)
        paths["qassoc"] = p
    return paths


def write_demo_dataset(out_dir: str, X, y, snp_ids, task: str, causal_idx, dataset_name: str = "demo", export_plink: bool = True, export_assoc: bool = True, export_tped: bool = True):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fids, iids = _make_sample_ids(int(X.shape[0]))
    pd.DataFrame(X, columns=snp_ids).to_csv(out / "genotypes.csv", index=False)
    pd.DataFrame({"phenotype": y}).to_csv(out / "phenotype.csv", index=False)
    pd.DataFrame(X.T).to_csv(out / f"{dataset_name}_filtered.csv", header=False, index=False)
    _write_phen_file(str(out / f"{dataset_name}_origin.phen"), y, task, fids, iids)

    written: Dict[str, str] = {
        "genotypes_csv": str(out / "genotypes.csv"),
        "phenotype_csv": str(out / "phenotype.csv"),
        "filtered_csv": str(out / f"{dataset_name}_filtered.csv"),
        "phen_file": str(out / f"{dataset_name}_origin.phen"),
    }

    if export_plink:
        prefix = str(out / dataset_name)
        _write_plink_files(prefix, X, y, snp_ids, task, fids, iids)
        written.update({"bed": prefix + ".bed", "bim": prefix + ".bim", "fam": prefix + ".fam"})

    if export_tped:
        tprefix = str(out / f"{dataset_name}_origin")
        _write_tped_tfam(tprefix, X, snp_ids, y, task, fids, iids)
        written.update({"tped": tprefix + ".tped", "tfam": tprefix + ".tfam"})

    if export_assoc:
        written.update(_write_assoc_files(str(out), dataset_name, X, y, snp_ids, task))

    meta = {
        "task": task,
        "dataset_name": dataset_name,
        "n_samples": int(X.shape[0]),
        "n_snps": int(X.shape[1]),
        "causal_snps": [snp_ids[i] for i in causal_idx[:min(20, len(causal_idx))]],
        "format_bundle": sorted(list(written.keys())),
        "paths": written,
    }
    with open(out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_demo_dataset(data_dir: str):
    X_df = pd.read_csv(os.path.join(data_dir, "genotypes.csv"))
    X = X_df.to_numpy(dtype=np.float32)
    snp_ids = list(X_df.columns)
    y = pd.read_csv(os.path.join(data_dir, "phenotype.csv"))["phenotype"].to_numpy()
    meta = json.load(open(os.path.join(data_dir, "metadata.json"), "r", encoding="utf-8"))
    return X, y, snp_ids, meta
