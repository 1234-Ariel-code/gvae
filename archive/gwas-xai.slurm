#!/bin/bash
#SBATCH --job-name=dynamic_gwas_repr_tsv
#SBATCH --output=/work/long_lab/Ariel_Kemogne/Representation_learning/AJHG/logs/%x_%j.out
#SBATCH --error=/work/long_lab/Ariel_Kemogne/Representation_learning/AJHG/logs/%x_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

OUT_DIR="/work/long_lab/Ariel_Kemogne/Representation_learning/AJHG/_dynamic_gwas_comparison"
LOG_DIR="/work/long_lab/Ariel_Kemogne/Representation_learning/AJHG/logs"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

R_SCRIPT="/work/long_lab/Ariel_Kemogne/Representation_learning/AJHG/gwas-xai.R"
PY_SCRIPT="/work/long_lab/Ariel_Kemogne/Representation_learning/AJHG/build_target_support_table.py"

LATENT_ROOT="/work/long_lab/Ariel_Kemogne/Representation_learning/AJHG/latent"

# IMPORTANT:
# this folder should contain files like:
# ALZ_MONDO_0004975_associations_export.tsv
# ASD_EFO_0003756_associations_export.tsv
GWAS_ROOT="/work/long_lab/for_Ariel/GWAS-summary"

# kept for compatibility, although the new R script no longer relies on BIM mapping
BIM_ROOT="/work/long_lab/for_Ariel/files"

CS2G_FILE="/work/long_lab/for_Ariel/files/combined_cS2G.tsv"
DISGENET_FILE="/work/long_lab/for_Ariel/files/consolidated.tsv"
DRUG_TARGET_FILE="/work/long_lab/for_Ariel/files/drug_target_gene_sets.tsv"

# Inputs for building target support table if needed
DEPMAP_GENE_FILE="/work/long_lab/for_Ariel/files/CRISPRGeneEffectUncorrected.csv"
DEPMAP_MODEL_FILE="/work/long_lab/for_Ariel/files/Model.csv"

echo "============================================================"
echo "Starting job at $(date)"
echo "Host: $(hostname)"
echo "============================================================"

module purge || true

module load R/4.4.1 || module load r/4.4.1 || module load R || {
  echo "[ERROR] Could not load an R module."
  exit 1
}

module load python/3.11 || module load python || true

export R_LIBS_USER="$HOME/R/x86_64-pc-linux-gnu-library/4.4"
mkdir -p "${R_LIBS_USER}"

unset CONDA_PREFIX || true
unset CONDA_DEFAULT_ENV || true
unset CONDA_PROMPT_MODIFIER || true
unset PYTHONPATH || true
unset PYTHONHOME || true
unset PKG_CONFIG_PATH || true

if [[ "${LD_LIBRARY_PATH:-}" == *"miniconda"* ]] || [[ "${LD_LIBRARY_PATH:-}" == *"conda"* ]]; then
  CLEAN_LD=$(echo "${LD_LIBRARY_PATH}" | tr ':' '\n' | grep -vi 'conda' | paste -sd ':' -)
  export LD_LIBRARY_PATH="${CLEAN_LD}"
fi

echo "R executable: $(which R || true)"
echo "Rscript executable: $(which Rscript || true)"
echo "Python executable: $(which python || true)"
echo "R_LIBS_USER=${R_LIBS_USER}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<empty>}"

[ -d "${LATENT_ROOT}" ] || { echo "[ERROR] LATENT_ROOT not found: ${LATENT_ROOT}"; exit 1; }
[ -d "${GWAS_ROOT}" ] || { echo "[ERROR] GWAS_ROOT not found: ${GWAS_ROOT}"; exit 1; }
[ -d "${BIM_ROOT}" ] || { echo "[ERROR] BIM_ROOT not found: ${BIM_ROOT}"; exit 1; }
[ -f "${CS2G_FILE}" ] || { echo "[ERROR] CS2G_FILE not found: ${CS2G_FILE}"; exit 1; }
[ -f "${DISGENET_FILE}" ] || { echo "[ERROR] DISGENET_FILE not found: ${DISGENET_FILE}"; exit 1; }
[ -f "${DEPMAP_GENE_FILE}" ] || { echo "[ERROR] DEPMAP_GENE_FILE not found: ${DEPMAP_GENE_FILE}"; exit 1; }
[ -f "${DEPMAP_MODEL_FILE}" ] || { echo "[ERROR] DEPMAP_MODEL_FILE not found: ${DEPMAP_MODEL_FILE}"; exit 1; }

if [ ! -f "${DRUG_TARGET_FILE}" ]; then
  echo "[INFO] Target support file not found. Building it now..."
  python "${PY_SCRIPT}" \
    --out_file "${DRUG_TARGET_FILE}" \
    --depmap_gene_file "${DEPMAP_GENE_FILE}" \
    --depmap_model_file "${DEPMAP_MODEL_FILE}"
else
  echo "[INFO] Using existing target support file: ${DRUG_TARGET_FILE}"
fi

Rscript -e "pkgs <- c('readr','dplyr','tidyr','ggplot2','stringr','purrr','tibble','patchwork','scales','glue'); miss <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly=TRUE)]; if(length(miss)>0){cat('Missing packages:', paste(miss, collapse=', '), '\n'); quit(status=1)} else {cat('All required R packages found.\n')}" \
  || { echo "[ERROR] Required R packages are missing in R_LIBS_USER=${R_LIBS_USER}"; exit 1; }

echo "[INFO] Listing latent files from ${LATENT_ROOT}"
find "${LATENT_ROOT}" -type f -name "latent_gene_table.ALL_SAMPLES.long_*.csv" | head -30 || true

echo "[INFO] Listing GWAS summary TSV files from ${GWAS_ROOT}"
find "${GWAS_ROOT}" -type f -name "*_associations_export.tsv" | head -50 || true

echo "[INFO] Launching main R script..."
Rscript "${R_SCRIPT}" \
  --mode cluster \
  --latent_root "${LATENT_ROOT}" \
  --gwas_root "${GWAS_ROOT}" \
  --bim_root "${BIM_ROOT}" \
  --cs2g_file "${CS2G_FILE}" \
  --disgenet_file "${DISGENET_FILE}" \
  --drug_target_file "${DRUG_TARGET_FILE}" \
  --out_dir "${OUT_DIR}"

echo "============================================================"
echo "Finished job at $(date)"
echo "============================================================"
