#!/bin/bash
#SBATCH --job-name=qgVAE_grid_prune
#SBATCH --partition=cpu2023
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/qgvae_grid_%A_%a.out
#SBATCH --error=logs/qgvae_grid_%A_%a.err
#SBATCH --array=0-2879

set -euo pipefail

module purge
module load python/3.12.5

QGVAE_SCRIPT="scripts/internal/qgvae_bed.py"
BASE_DATA="/work/long_lab/for_Ariel/files"
RUN_ROOT="/work/long_lab/Ariel_Kemogne/Representation_learning/AJHG"
PLINK="/work/long_lab/for_Ariel/plink"

DISEASES=(BD CD CAD T1D T2D HT RA ALZ ASD BMI BRC COL EOS HDL HGT LDL LUN OCD PRC)
LDs=(10 30 50 100 150 200)
NSs=(10 30 50 100 150 200)
Ls=(2 3 4 5)
BETAS="1.0,2.0,4.0,10.0"

nD=${#DISEASES[@]}; nLD=${#LDs[@]}; nNS=${#NSs[@]}; nL=${#Ls[@]}
idx=${SLURM_ARRAY_TASK_ID}
i_D=$(( idx % nD )); tmp=$(( idx / nD ))
i_LD=$(( tmp % nLD )); tmp=$(( tmp / nLD ))
i_NS=$(( tmp % nNS )); i_L=$(( tmp / nNS ))

DISEASE=${DISEASES[$i_D]}
LATENT_DIM=${LDs[$i_LD]}
NUM_SAMPLE=${NSs[$i_NS]}
NUM_LAYER=${Ls[$i_L]}

WORK_DIR="${RUN_ROOT}/${DISEASE}/outputs_bed"
mkdir -p "${WORK_DIR}" "${WORK_DIR}/logs"
cd "${WORK_DIR}"

RAW_BED_PREFIX="${BASE_DATA}/${DISEASE}"
python "${QGVAE_SCRIPT}" \
  --disease "${DISEASE}" \
  --num_sample "${NUM_SAMPLE}" \
  --latent_dim "${LATENT_DIM}" \
  --num_layer "${NUM_LAYER}" \
  --epochs 20 \
  --batch_size 32 \
  --bed_prefix "${RAW_BED_PREFIX}" \
  --beta_list "${BETAS}" \
  --out_dir "${WORK_DIR}"
