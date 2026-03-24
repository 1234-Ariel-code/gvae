#!/bin/bash
#SBATCH --job-name=enr_HT
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --time=23:00:00
#SBATCH --array=1-144
#SBATCH --output=logs/enr_%A_%a.out
#SBATCH --error=logs/enr_%A_%a.err

set -euo pipefail
module purge
module load python/3.12.5

DISEASE="HT"
K=500
LDs=(10 30 50 100 150 200)
NSs=(10 30 50 100 150 200)
Ls=(2 3 4 5)

idx=$((SLURM_ARRAY_TASK_ID - 1))
i_ld=$((idx % ${#LDs[@]})); tmp=$((idx / ${#LDs[@]}))
i_l=$((tmp % ${#Ls[@]})); i_ns=$((tmp / ${#Ls[@]}))
LD=${LDs[$i_ld]}; L=${Ls[$i_l]}; NS=${NSs[$i_ns]}

BASE_DIR="/work/long_lab/Ariel_Kemogne/Representation_learning/AJHG/${DISEASE}"
OUT_DIR="${BASE_DIR}/pathway_outputs"
S2G="/work/long_lab/Ariel_Kemogne/Representation_learning/eSPRESSO/${DISEASE}/combined_cS2G.tsv"
DISGENET_TSV="/work/long_lab/Ariel_Kemogne/Representation_learning/eSPRESSO/${DISEASE}/consolidated.tsv"
PY="scripts/internal/biology_with_generelevance.py"

python "${PY}" \
  --base_dir "${BASE_DIR}" \
  --output_dir "${OUT_DIR}" \
  --disease "${DISEASE}" \
  --LD "${LD}" --NS "${NS}" --L "${L}" --K "${K}" \
  --s2g_path "${S2G}" \
  --mode enrichr \
  --gene_sets Reactome_2022 KEGG_2021_Human GO_Biological_Process_2023 \
  --run_gene_analysis \
  --disgenet_tsv "${DISGENET_TSV}" \
  --disgenet_disease_name "Hypertension"
