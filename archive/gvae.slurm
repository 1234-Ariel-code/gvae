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

echo "[$(date)] Starting grid job TASK_ID=${SLURM_ARRAY_TASK_ID}"

#############################
# 0) Environment
#############################
module purge
module load python/3.12.5

cd /work/long_lab/janith/VAE_projs
source venv/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTEROP_THREADS=2

#############################
# 1) Paths
#############################
QGVAE_SCRIPT="/work/long_lab/Ariel_Kemogne/Representation_learning/AJHG/qgvae_bed.py"
BASE_DATA="/work/long_lab/for_Ariel/files"
RUN_ROOT="/work/long_lab/Ariel_Kemogne/Representation_learning/AJHG"

# Local PLINK 1.9 binary you confirmed works
PLINK="/work/long_lab/for_Ariel/plink"
if [[ ! -x "${PLINK}" ]]; then
  echo "[FATAL] PLINK not executable at: ${PLINK}"
  exit 1
fi
echo "[INFO] Using PLINK: ${PLINK}"
"${PLINK}" --version || true

#############################
# 2) Grid definition
#############################
DISEASES=(BD CD CAD T1D T2D HT RA ALZ ASD BMI BRC COL EOS HDL HGT LDL LUN OCD PRC)

LDs=(10 30 50 100 150 200)
NSs=(10 30 50 100 150 200)
Ls=(2 3 4 5)

BETAS="1.0,2.0,4.0,10.0"

nD=${#DISEASES[@]}
nLD=${#LDs[@]}
nNS=${#NSs[@]}
nL=${#Ls[@]}
TOTAL=$((nD * nLD * nNS * nL))

if [[ "${SLURM_ARRAY_TASK_ID}" -ge "${TOTAL}" ]]; then
  echo "[ERROR] TASK_ID out of range (0..$((TOTAL-1)))."
  exit 1
fi

#############################
# 3) Index -> config
#############################
idx=${SLURM_ARRAY_TASK_ID}

i_D=$(( idx % nD ))
tmp=$(( idx / nD ))

i_LD=$(( tmp % nLD ))
tmp=$(( tmp / nLD ))

i_NS=$(( tmp % nNS ))
i_L=$(( tmp / nNS ))

DISEASE=${DISEASES[$i_D]}
LATENT_DIM=${LDs[$i_LD]}
NUM_SAMPLE=${NSs[$i_NS]}
NUM_LAYER=${Ls[$i_L]}

CFG_TAG="${DISEASE}_LD${LATENT_DIM}_NS${NUM_SAMPLE}_L${NUM_LAYER}"
echo "[$(date)] CONFIG: ${CFG_TAG}"

#############################
# 4) Work dir
#############################
WORK_DIR="${RUN_ROOT}/${DISEASE}/outputs_bed"
mkdir -p "${WORK_DIR}/qc_prune" "${WORK_DIR}/logs"
cd "${WORK_DIR}"

#############################
# 5) Input bed prefix per disease
#############################
RAW_BED_PREFIX="${BASE_DATA}/${DISEASE}"

if [[ ! -f "${RAW_BED_PREFIX}.bed" ]]; then
  echo "[FATAL] Missing input bed for ${DISEASE}: ${RAW_BED_PREFIX}.bed"
  exit 1
fi

#############################
# 6) QC + LD pruning (cached per disease, with lock)
#############################
QC_DIR="${WORK_DIR}/qc_prune"
QC1_PREFIX="${QC_DIR}/${DISEASE}.QC1"
PRUNE_PREFIX="${QC_DIR}/${DISEASE}.PRUNE"
PRUNED_PREFIX="${QC_DIR}/${DISEASE}.QC1_PRUNED"
MANIFEST="${QC_DIR}/${DISEASE}.qc_prune_manifest.txt"

# QC thresholds (adjust if needed)
MIND="0.02"
GENO="0.02"
MAF="0.01"
HWE="1e-6"

# LD prune params (plink1.9 compatible)
WINDOW="200"
STEP="50"
R2TH="0.2"

# Optional: high-LD regions file (PLINK1 supports --exclude range)
HIGH_LD_RANGES=""  # e.g., "${RUN_ROOT}/resources/high_ld_regions.txt"

# Lock to avoid multiple jobs pruning the same disease simultaneously
LOCKDIR="${QC_DIR}/.${DISEASE}.LOCK"
acquired_lock=0

if [[ -f "${PRUNED_PREFIX}.bed" && -f "${PRUNED_PREFIX}.bim" && -f "${PRUNED_PREFIX}.fam" ]]; then
  echo "[$(date)] Found cached PRUNED bed: ${PRUNED_PREFIX}.bed"
else
  echo "[$(date)] Need QC+PRUNE for ${DISEASE}"

  # Try to acquire lock
  if mkdir "${LOCKDIR}" 2>/dev/null; then
    acquired_lock=1
    echo "[$(date)] Lock acquired: ${LOCKDIR}"
  else
    echo "[$(date)] Another task is pruning ${DISEASE}. Waiting..."
    # Wait up to 2 hours, checking every 30s
    for _ in $(seq 1 240); do
      if [[ -f "${PRUNED_PREFIX}.bed" && -f "${PRUNED_PREFIX}.bim" && -f "${PRUNED_PREFIX}.fam" ]]; then
        echo "[$(date)] PRUNED appeared. Proceeding."
        break
      fi
      sleep 30
    done
  fi

  # If still not present, and we have the lock, run pruning
  if [[ ! -f "${PRUNED_PREFIX}.bed" ]]; then
    if [[ "${acquired_lock}" -ne 1 ]]; then
      echo "[FATAL] PRUNED not found and could not acquire lock for ${DISEASE}."
      exit 1
    fi

    echo "[$(date)] Running PLINK1.9 QC for ${DISEASE}"

    # QC
    "${PLINK}" \
      --bfile "${RAW_BED_PREFIX}" \
      --mind "${MIND}" \
      --geno "${GENO}" \
      --maf "${MAF}" \
      --hwe "${HWE}" \
      --make-bed \
      --out "${QC1_PREFIX}" \
      > "${QC_DIR}/${DISEASE}.QC1.plink.log" 2>&1

    QC_BASE="${QC1_PREFIX}"

    # Optional: remove high-LD regions
    if [[ -n "${HIGH_LD_RANGES}" && -f "${HIGH_LD_RANGES}" ]]; then
      echo "[$(date)] Excluding high-LD regions: ${HIGH_LD_RANGES}"
      "${PLINK}" \
        --bfile "${QC1_PREFIX}" \
        --exclude range "${HIGH_LD_RANGES}" \
        --make-bed \
        --out "${QC1_PREFIX}.noHLD" \
        > "${QC_DIR}/${DISEASE}.noHLD.plink.log" 2>&1
      QC_BASE="${QC1_PREFIX}.noHLD"
    fi

    echo "[$(date)] LD pruning (${WINDOW}, ${STEP}, ${R2TH})"
    "${PLINK}" \
      --bfile "${QC_BASE}" \
      --indep-pairwise "${WINDOW}" "${STEP}" "${R2TH}" \
      --out "${PRUNE_PREFIX}" \
      > "${QC_DIR}/${DISEASE}.PRUNE.plink.log" 2>&1

    echo "[$(date)] Extract pruned SNPs"
    "${PLINK}" \
      --bfile "${QC_BASE}" \
      --extract "${PRUNE_PREFIX}.prune.in" \
      --make-bed \
      --out "${PRUNED_PREFIX}" \
      > "${QC_DIR}/${DISEASE}.PRUNED.plink.log" 2>&1

    # Manifest
    {
      echo "disease=${DISEASE}"
      echo "raw_bed_prefix=${RAW_BED_PREFIX}"
      echo "qc_prefix=${QC1_PREFIX}"
      echo "qc_base=${QC_BASE}"
      echo "pruned_prefix=${PRUNED_PREFIX}"
      echo "MIND=${MIND}"
      echo "GENO=${GENO}"
      echo "MAF=${MAF}"
      echo "HWE=${HWE}"
      echo "LD_prune_window=${WINDOW}"
      echo "LD_prune_step=${STEP}"
      echo "LD_prune_r2=${R2TH}"
      echo "HIGH_LD_RANGES=${HIGH_LD_RANGES}"
      echo "date=$(date)"
      echo "hostname=$(hostname)"
      echo "pwd=$(pwd)"
    } > "${MANIFEST}"

    echo "[$(date)] QC+PRUNE done. Manifest: ${MANIFEST}"
  fi

  # Release lock if we acquired it
  if [[ "${acquired_lock}" -eq 1 ]]; then
    rmdir "${LOCKDIR}" || true
    echo "[$(date)] Lock released: ${LOCKDIR}"
  fi
fi

#############################
# 7) Train on pruned bed
#############################
echo "[$(date)] Training on PRUNED bed prefix: ${PRUNED_PREFIX}"

python "${QGVAE_SCRIPT}" \
  --disease "${DISEASE}" \
  --num_sample "${NUM_SAMPLE}" \
  --latent_dim "${LATENT_DIM}" \
  --num_layer "${NUM_LAYER}" \
  --epochs 20 \
  --batch_size 32 \
  --bed_prefix "${PRUNED_PREFIX}" \
  --beta_list "${BETAS}" \
  --clip_log_var \
  > "logs/log_${CFG_TAG}.out" 2> "logs/log_${CFG_TAG}.err"

echo "[$(date)] Done: ${CFG_TAG}"

