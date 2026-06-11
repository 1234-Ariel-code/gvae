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
#
module purge
module load python/3.12.5

############################
# User-configurable bits
############################
DISEASE="HT"
# K=10
K=500

# Sweep grids
# LDs=(10 30 50 100 150 200)
# Ls=(2 3 4 5)
# NSs=(10 30 50 100 150 200)

LDs=(30 30 50 100 150 200)
NSs=(100 30 50 100 150 200)
Ls=(2 3 4 5)

nLD=${#LDs[@]}   # 6
nL=${#Ls[@]}     # 4
nNS=${#NSs[@]}   # 6

############################
# Map SLURM_ARRAY_TASK_ID -> (LD, L, NS)
############################
idx=$((SLURM_ARRAY_TASK_ID - 1))

i_ld=$((idx % nLD))
tmp=$((idx / nLD))
i_l=$((tmp % nL))
i_ns=$((tmp / nL))

LD=${LDs[$i_ld]}
L=${Ls[$i_l]}
NS=${NSs[$i_ns]}

TAG="LD${LD}_NS${NS}_L${L}_K${K}"

############################
# Paths
############################
BASE_DIR="/work/long_lab/Ariel_Kemogne/Representation_learning/AJHG/${DISEASE}"
CODE_DIR="/work/long_lab/Ariel_Kemogne/Representation_learning/AJHG/VAE_projs"
OUT_DIR="${BASE_DIR}/pathway_outputs"

# Your mapping resources
S2G="/work/long_lab/Ariel_Kemogne/Representation_learning/eSPRESSO/${DISEASE}/combined_cS2G.tsv"
BIM="/work/long_lab/for_Ariel/${DISEASE}.bim"

# DisGeNET consolidated TSV (your local dump)
DISGENET_TSV="/work/long_lab/Ariel_Kemogne/Representation_learning/eSPRESSO/${DISEASE}/consolidated.tsv"

PY="${CODE_DIR}/biology_with_generelevance.py"
# PY="${CODE_DIR}/relevance.py"

############################
# Activate venv + prep dirs
############################
cd "${BASE_DIR}"
# source venv/bin/activate

mkdir -p logs
mkdir -p "${OUT_DIR}/${DISEASE}/${TAG}"

echo "[INFO] ArrayID=${SLURM_ARRAY_TASK_ID}  Config: ${DISEASE} ${TAG}"
echo "[INFO] Using DisGeNET TSV: ${DISGENET_TSV}"

############################
# Disease matching settings
############################
# Since consolidated.tsv is disease-specific, name matching is typically enough.
# If you want exact match, you can set:
#   DISGENET_DISEASE_IDS=("DOID:xxxxxxx")
# DISGENET_DISEASE_NAME="${DISEASE}"

############################
# Disease name mapping (short code -> full name)
############################
# You keep using DISEASE as the short code for paths/files,
# but we pass the full name to DisGeNET matching.

get_disgenet_name () {
  case "$1" in
    BD)  echo "Bipolar disorder" ;;
    CAD) echo "Coronary artery disease" ;;
    CD)  echo "Crohn's disease" ;;
    T1D) echo "Type 1 diabetes mellitus" ;;
    T2D) echo "Type 2 diabetes mellitus" ;;
    HT)  echo "Hypertension" ;;
    RA)  echo "Rheumatoid arthritis" ;;

    ALZ) echo "Alzheimer's disease" ;;
    ASD) echo "Autism spectrum disorder" ;;
    OCD) echo "Obsessive-compulsive disorder" ;;
    BRC) echo "Breast cancer" ;;
    COL) echo "Colorectal cancer" ;;
    PRC) echo "Prostate cancer" ;;
    LUN) echo "Lung cancer" ;;

    BMI) echo "Body mass index" ;;
    GLU) echo "Fasting glucose" ;;
    HDL) echo "HDL cholesterol" ;;
    HGT) echo "Height" ;;
    LDL) echo "LDL cholesterol" ;;
    EOS) echo "Eosinophil count" ;;

    *)   echo "$1" ;;  # fallback: use whatever was provided
  esac
}

# Allow manual override if you ever want:
#   sbatch --export=DISEASE=ALZ,DISGENET_DISEASE_NAME="Alzheimer's disease" enr.slurm
DISGENET_DISEASE_NAME="${DISGENET_DISEASE_NAME:-$(get_disgenet_name "${DISEASE}")}"

echo "[INFO] DisGeNET disease name: ${DISGENET_DISEASE_NAME}"


############################
# (1) Individual samples
############################
FOUND=0
FOUND_N=0

for s in $(seq 1 "${NS}"); do
  SID="S${s}"
  SHAP_FILE="${BASE_DIR}/outputs/${DISEASE}_LD${LD}_NS${NS}_L${L}_K${K}_${SID}_top_snps_per_latent.csv"

  if [[ ! -f "${SHAP_FILE}" ]]; then
    echo "[SKIP] Missing SHAP for ${DISEASE} ${SID}: ${SHAP_FILE}"
    continue
  fi

  FOUND=1
  FOUND_N=$((FOUND_N + 1))

  echo "[RUN] ${DISEASE} ${SID} LD=${LD} NS=${NS} L=${L} K=${K}"
  python "${PY}" \
    --base_dir "${BASE_DIR}" \
    --output_dir "${OUT_DIR}" \
    --disease "${DISEASE}" \
    --LD "${LD}" --NS "${NS}" --L "${L}" --K "${K}" \
    --sample_id "${SID}" \
    --s2g_path "${S2G}" \
    --bim_file "${BIM}" \
    --mode enrichr \
    --gene_sets Reactome_2022 KEGG_2021_Human GO_Biological_Process_2023 \
    --genes_per_lv 200 \
    --top_per_library 10 \
    --heatmap_top_n 10 \
    --heatmap_top_method sig_count \
    --lv_order sig_count \
    --run_gene_analysis \
    --disgenet_mode tsv \
    --disgenet_tsv "${DISGENET_TSV}" \
    --disgenet_disease_name "${DISGENET_DISEASE_NAME}" \
    --disgenet_genes_per_group 200 \
    --disgenet_relevance_thresh 0.1 \
    --disgenet_heatmap_top_n 50
done

echo "[INFO] Finished per-sample loop. Found ${FOUND_N} sample(s) with SHAP files."

############################
# (2) Aggregated samples (ALL_SAMPLES)
############################
if [[ "${FOUND}" -eq 1 ]]; then
  echo "[RUN] ${DISEASE} ALL_SAMPLES (pooled + by-sample) — using available samples only"

  python "${PY}" \
    --base_dir "${BASE_DIR}" \
    --output_dir "${OUT_DIR}" \
    --disease "${DISEASE}" \
    --LD "${LD}" --NS "${NS}" --L "${L}" --K "${K}" \
    --s2g_path "${S2G}" \
    --bim_file "${BIM}" \
    --mode enrichr \
    --gene_sets Reactome_2022 KEGG_2021_Human GO_Biological_Process_2023 \
    --genes_per_lv 200 \
    --top_per_library 10 \
    --heatmap_top_n 10 \
    --heatmap_top_method sig_count \
    --lv_order sig_count \
    --run_aggregated_samples \
    --run_gene_analysis \
    --disgenet_mode tsv \
    --disgenet_tsv "${DISGENET_TSV}" \
    --disgenet_disease_name "${DISGENET_DISEASE_NAME}" \
    --disgenet_genes_per_group 200 \
    --disgenet_relevance_thresh 0.1 \
    --disgenet_heatmap_top_n 50
else
  echo "[WARN] No per-sample SHAP files found for ${DISEASE} ${TAG}. Skipping ALL_SAMPLES."
fi

echo "[DONE] ${DISEASE} ${TAG}"

