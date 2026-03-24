from __future__ import annotations
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name != "app" else Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse, os, glob
import pandas as pd
from gvae.biology import run_mock_biology

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    args = ap.parse_args()
    xai_files = glob.glob(os.path.join(args.input_dir, "*", "xai", "top_snps_per_latent.csv"))
    if not xai_files:
        raise FileNotFoundError("No xai/top_snps_per_latent.csv files found.")
    for fp in xai_files:
        model_dir = os.path.dirname(os.path.dirname(fp))
        snp_df = pd.read_csv(fp)
        run_mock_biology(snp_df, os.path.join(model_dir, "biology"))
    print("Gene/pathway mock analysis complete.")

if __name__ == "__main__":
    main()
