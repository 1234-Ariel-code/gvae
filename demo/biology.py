from __future__ import annotations
import os
import pandas as pd

def mock_snp_to_gene(snp_df: pd.DataFrame) -> pd.DataFrame:
    out = snp_df.copy()
    out["GENE"] = out["SNP_ID"].astype(str).str.replace("rsSIM", "GENE", regex=False)
    return out[["Latent_Dim", "SNP_ID", "GENE", "Importance"]]

def mock_pathway_enrichment(gene_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lv, sub in gene_df.groupby("Latent_Dim"):
        genes = sorted(sub["GENE"].astype(str).unique())
        rows.append({"Latent_Dim": lv, "Pathway": f"Mock pathway A ({lv})", "Score": min(5.0, 1.0 + 0.02 * len(genes)), "Genes": ";".join(genes[:15])})
        rows.append({"Latent_Dim": lv, "Pathway": f"Mock pathway B ({lv})", "Score": min(4.0, 0.8 + 0.015 * len(genes)), "Genes": ";".join(genes[:15])})
    return pd.DataFrame(rows)

def run_mock_biology(snp_df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    gene_df = mock_snp_to_gene(snp_df)
    path_df = mock_pathway_enrichment(gene_df)
    gene_df.to_csv(os.path.join(out_dir, "gene_table.csv"), index=False)
    path_df.to_csv(os.path.join(out_dir, "pathway_table.csv"), index=False)
    return gene_df, path_df
