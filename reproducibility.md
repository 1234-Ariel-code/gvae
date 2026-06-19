# Reproducibility Guide

This document describes the recommended steps for reproducing the main computational workflows in this repository.

The repository is organized around a manuscript-facing gVAE implementation, shared model utilities, SNP attribution, downstream prediction, SNP-to-gene mapping, pathway enrichment, and GWAS-XAI comparison workflows.

---

## 1. Computational environment

The recommended environment can be created using Conda:

```bash
conda env create -f environment.yml
conda activate gvae
```

Alternatively, the Python requirements can be installed with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For editable installation:

```bash
pip install -e .
```

After installation, the main Python workflows should be run from the repository root using module-style commands such as:

```bash
python -m gvae.gvae
python -m gvae.snp_prioritization
```

---

## 2. Quick smoke test

Users can verify that the package imports correctly with:

```bash
python -c "from gvae.model import GVAE; from gvae.metrics import evaluate_r_square; print('gVAE imports OK')"
```

The command-line interfaces can be inspected with:

```bash
python -m gvae.gvae --help
python -m gvae.snp_prioritization --help
```

A minimal model-training test can be run on a small PLINK BED dataset using:

```bash
python -m gvae.gvae \
  --disease TEST \
  --bed_prefix /path/to/example \
  --latent_dim 4 \
  --num_sample 5 \
  --num_layer 2 \
  --epochs 2 \
  --batch_size 16 \
  --feature_mode none \
  --output_dir test_outputs
```

This command is intended only to verify installation, imports, data loading, model construction, training, and output writing. Manuscript-scale analyses require the full genotype, GWAS, SNP-to-gene, and pathway resources described below.

---

## 3. Expected input files

The scripts assume that genotype, phenotype, GWAS, SNP-to-gene, and pathway annotation files are available locally.

Typical inputs include:

```text
<DISEASE>_filtered.csv
<DISEASE>_origin.phen
<DISEASE>_origin.tped
<DISEASE>.bed
<DISEASE>.bim
<DISEASE>.fam
<DISEASE>_gwas.assoc
SNP-to-gene mapping table
GMT pathway files or Enrichr libraries
DisGeNET TSV file
```

The exact file paths should be adjusted in the command-line arguments or SLURM scripts.

For PLINK BED input, the model-training script expects a file prefix rather than a full filename. For example, if the files are:

```text
/path/to/plink/T2D.bed
/path/to/plink/T2D.bim
/path/to/plink/T2D.fam
```

then the corresponding argument is:

```bash
--bed_prefix /path/to/plink/T2D
```

---

## 4. Model training

The main model-training workflow is implemented in:

```text
gvae/gvae.py
```

The reviewer-facing command should be run from the repository root using module-style execution:

```bash
python -m gvae.gvae \
  --disease T2D \
  --bed_prefix /path/to/plink/T2D \
  --latent_dim 100 \
  --num_sample 150 \
  --num_layer 4 \
  --epochs 50 \
  --batch_size 256 \
  --feature_mode gwas_top \
  --downsample_d 50000 \
  --gwas_assoc_path /path/to/T2D_gwas.assoc \
  --output_dir /path/to/model_outputs
```

For a quick test without GWAS-based SNP filtering, use:

```bash
python -m gvae.gvae \
  --disease TEST \
  --bed_prefix /path/to/plink/TEST \
  --latent_dim 4 \
  --num_sample 5 \
  --num_layer 2 \
  --epochs 2 \
  --batch_size 16 \
  --feature_mode none \
  --output_dir test_outputs
```

---

## 5. SNP prioritization

Latent-variable-specific SNP prioritization is implemented in:

```text
gvae/snp_prioritization.py
```

This script uses the shared gVAE architecture from `gvae/model.py` and should be run from the repository root using module-style execution:

```bash
python -m gvae.snp_prioritization \
  --disease T2D \
  --base_path /path/to/genotype/files \
  --latent_dim 100 \
  --num_samples 150 \
  --num_layers 4 \
  --shap_top_k 10 \
  --tped_file /path/to/T2D_origin.tped \
  --assoc_path /path/to/T2D_gwas.assoc \
  --output_dir /path/to/xai_outputs
```

The `shap_top_k` argument controls the number of top SHAP-ranked SNPs retained per latent variable. This is distinct from `num_samples`, which controls the number of posterior latent samples used by gVAE.

---

## 6. Latent-space prediction

Downstream classification or regression is implemented in:

```text
gvae/latent_classification.py
```

Example:

```bash
python gvae/latent_classification.py \
  --disease T2D \
  --base_path /path/to/genotype/files \
  --model_type gvae \
  --latent_dim 100 \
  --num_samples 150 \
  --num_layers 4 \
  --feature_mode gwas_top \
  --downsample_d 50000 \
  --assoc_path /path/to/T2D_gwas.assoc \
  --tped_file /path/to/T2D_origin.tped \
  --train_vae_epochs 50 \
  --vae_batch_size 256 \
  --batch_size 256 \
  --epochs 120 \
  --cache_latents \
  --out_root /path/to/latent_classification_outputs \
  --make_plots
```

This workflow evaluates whether the learned latent features preserve phenotype-relevant structure for downstream classification or regression tasks.

---

## 7. Gene and pathway analysis

The SHAP-to-biology interpretation workflow is implemented in:

```text
gvae/gene-pathway_enrichment.py
```

Example:

```bash
python gvae/gene-pathway_enrichment.py \
  --disease T2D \
  --base_dir /path/to/xai_outputs \
  --s2g_path /path/to/snp_to_gene.tsv \
  --bim_file /path/to/T2D.bim \
  --run_gene_analysis \
  --disgenet_mode tsv \
  --disgenet_tsv /path/to/disgenet.tsv \
  --disgenet_disease_name "type 2 diabetes" \
  --out_root /path/to/gene_pathway_outputs
```

For reproducibility, local TSV mode is recommended for DisGeNET analyses. API mode is available but depends on network access, credentials, API availability, and database version.

---

## 8. GWAS-XAI comparison

GWAS and gVAE-XAI comparison analyses are implemented in:

```text
gvae/gwas-xai.R
```

This script supports matched-budget comparisons between GWAS-ranked and gVAE-prioritized signals after SNP-to-gene mapping.

---

## 9. SLURM execution

Example SLURM scripts are provided for high-performance computing environments:

```text
gvae/gvae.slurm
gvae/gene-pathway_enrichment.slurm
gvae/gwas-xai.slurm
```

These scripts should be edited to match local paths, account names, memory limits, module systems, and runtime requirements.

---

## 10. Random seeds and reproducibility notes

The main scripts set fixed random seeds where applicable. Exact reproducibility may still depend on hardware, TensorFlow backend behavior, GPU determinism, package versions, and annotation resource versions.

For reviewer-facing runs, we recommend recording:

```text
Git commit hash
Command-line arguments
Software environment
Input data versions
Annotation resource versions
Random seed
Output directory
```

The gene/pathway pipeline writes run parameters to `run_parameters.json`.

---

## 11. Output organization

Outputs are written to the user-specified output directories and may include:

```text
trained model weights
reconstruction summaries
robustness summaries
latent representations
top SNPs per latent variable
SHAP attribution summaries
classification/regression metrics
SNP-to-gene mapped tables
pathway enrichment tables
LV-by-pathway heatmaps
DisGeNET disease-gene relevance summaries
target-support tables
```

---

## 12. Recommended reviewer workflow

A typical reviewer-facing workflow is:

1. Create the computational environment.
2. Run the quick smoke test.
3. Prepare genotype, phenotype, GWAS, and annotation files.
4. Run model training.
5. Run SNP prioritization.
6. Run latent-space prediction analyses.
7. Run SNP-to-gene and pathway enrichment analyses.
8. Run GWAS-XAI matched-budget comparisons.
9. Check generated summaries and figures.
