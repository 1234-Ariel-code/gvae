#!/bin/bash
#SBATCH --job-name=dynamic_gwas_repr
#SBATCH --output=/work/long_lab/Ariel_Kemogne/Representation_learning/AJHG/logs/%x_%j.out
#SBATCH --error=/work/long_lab/Ariel_Kemogne/Representation_learning/AJHG/logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
module purge || true
module load R/4.4.1 || module load r/4.4.1 || module load R || exit 1

R_SCRIPT="scripts/internal/dynamic_representation_vs_gwas.R"
LATENT_ROOT="/work/long_lab/Ariel_Kemogne/Representation_learning/AJHG/latent"
GWAS_ROOT="/work/long_lab/for_Ariel/gwas_results"
BIM_ROOT="/work/long_lab/for_Ariel/files"
CS2G_FILE="/work/long_lab/for_Ariel/files/combined_cS2G.tsv"
DISGENET_FILE="/work/long_lab/for_Ariel/files/consolidated.tsv"
DRUG_TARGET_FILE="/work/long_lab/for_Ariel/files/drug_target_gene_sets.tsv"
OUT_DIR="/work/long_lab/Ariel_Kemogne/Representation_learning/AJHG/_dynamic_gwas_comparison"

Rscript "${R_SCRIPT}" \
  --mode cluster \
  --latent_root "${LATENT_ROOT}" \
  --gwas_root "${GWAS_ROOT}" \
  --bim_root "${BIM_ROOT}" \
  --cs2g_file "${CS2G_FILE}" \
  --disgenet_file "${DISGENET_FILE}" \
  --drug_target_file "${DRUG_TARGET_FILE}" \
  --out_dir "${OUT_DIR}"
