#!/bin/bash
#SBATCH --job-name=dynamic_gwas_repr
#SBATCH --output=/work/long_lab/Ariel_Kemogne/Representation_learning/AJHG/logs/%x_%j.out
#SBATCH --error=/work/long_lab/Ariel_Kemogne/Representation_learning/AJHG/logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail OUT_DIR="/work/long_lab/Ariel_Kemogne/Representation_learning/AJHG/_dynamic_gwas_comparison" LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${OUT_DIR}" "${LOG_DIR}" R_SCRIPT="/work/long_lab/Ariel_Kemogne/Representation_learning/AJHG/dynamic_representation_vs_gwas.R" PY_SCRIPT="/work/long_lab/Ariel_Kemogne/Representation_learning/AJHG/build_target_support_table.py" LATENT_ROOT="/work/long_lab/Ariel_Kemogne/Representation_learning/AJHG/latent" GWAS_ROOT="/work/long_lab/for_Ariel/gwas_results" BIM_ROOT="/work/long_lab/for_Ariel/files" CS2G_FILE="/work/long_lab/for_Ariel/files/combined_cS2G.tsv" DISGENET_FILE="/work/long_lab/for_Ariel/files/consolidated.tsv" DRUG_TARGET_FILE="/work/long_lab/for_Ariel/files/drug_target_gene_sets.tsv" DEPMAP_FILE="/work/long_lab/for_Ariel/files/depmap_dependency_export.csv"
echo "============================================================"
echo "Starting job at $(date)"
echo "Host: $(hostname)"
echo "============================================================"
module purge || true
module load R/4.4.1 ||
module load r/4.4.1 ||
module load R || {
echo "[ERROR] Could not load an R module." exit 1 }
module load python/3.11 ||
module load
python || true
export R_LIBS_USER="$HOME/R/x86_64-pc-linux-gnu-library/4.4"
mkdir -p "${R_LIBS_USER}"
unset CONDA_PREFIX || true
unset CONDA_DEFAULT_ENV || true
unset CONDA_PROMPT_MODIFIER || true
unset PYTHONPATH || true
unset PYTHONHOME || true
unset PKG_CONFIG_PATH || true
if [[ "${LD_LIBRARY_PATH:-}" == *"miniconda"* ]] || [[ "${LD_LIBRARY_PATH:-}" == *"conda"* ]]; then CLEAN_LD=$(echo "${LD_LIBRARY_PATH}" | tr ':' '\n' | grep -vi 'conda' | paste -sd ':' -)
export LD_LIBRARY_PATH="${CLEAN_LD}"
fi
echo "R executable: $(which R || true)"
echo "Rscript executable: $(which
Rscript || true)"
echo "Python executable: $(which
python || true)"
echo "R_LIBS_USER=${R_LIBS_USER}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<empty>}" [ -d "${LATENT_ROOT}" ] || {
echo "[ERROR] LATENT_ROOT not found: ${LATENT_ROOT}
