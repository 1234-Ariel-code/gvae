# Reproducibility Guide

This document describes the recommended steps for reproducing the main computational workflows in this repository.

## 1. Computational environment

The recommended environment can be created using Conda:

```bash
conda env create -f environment.yml
conda activate gvae
````

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

## 2. Expected input files

The scripts assume that genotype, phenotype, GWAS, SNP-to-gene, and pathway annotation files are available locally.

Typical inputs include:

```text
<DISEASE>_filtered.csv
<DISEASE>_origin.phen
<DISEASE>_origin.tped
<DISEASE>.bim
<DISEASE>_gwas.assoc
SNP-to-gene mapping table
GMT pathway files or Enrichr libraries
DisGeNET TSV file
```

Paths should be adjusted in the command-line arguments or SLURM scripts.

## 3. Model training

The main model-training workflow is implemented in:

```text
gvae/gvae.py
```

Example:

```bash
python gvae/gvae.py \
  --disease T2D \
  --base_path /path/to/genotype/files \
  --latent_dim 100 \
  --num_samples 150 \
  --num_layers 4 \
  --out_root /path/to/outputs
```

## 4. SNP prioritization

Latent-variable-specific SNP prioritization is implemented in:

```text
gvae/snp_prioritization.py
```

Example:

```bash
python gvae/snp_prioritization.py \
  --disease T2D \
  --base_path /path/to/genotype/files \
  --latent_dim 100 \
  --num_samples 150 \
  --num_layers 4 \
  --shap_top_k 10 \
  --out_root /path/to/xai_outputs
```

## 5. Latent-space prediction

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

## 6. Gene and pathway analysis

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

## 7. GWAS-XAI comparison

GWAS and gVAE-XAI comparison analyses are implemented in:

```text
gvae/gwas-xai.R
```

This script supports matched-budget comparisons between GWAS-ranked and gVAE-prioritized signals after SNP-to-gene mapping.

## 8. SLURM execution

Example SLURM scripts are provided for high-performance computing environments:

```text
gvae/gvae.slurm
gvae/gene-pathway_enrichment.slurm
gvae/gwas-xai.slurm
```

These scripts should be edited to match local paths, account names, memory limits, module systems, and runtime requirements.

## 9. Random seeds and reproducibility notes

Where supported, scripts expose a `--seed` argument. Exact reproducibility may still depend on hardware, TensorFlow backend behavior, GPU determinism, and package versions.

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

## 10. Output organization

Outputs are written to the user-specified output directories and may include:

```text
trained model outputs
latent representations
top SNPs per latent variable
SHAP attribution summaries
classification/regression metrics
SNP-to-gene mapped tables
pathway enrichment tables
LV-by-pathway heatmaps
DisGeNET disease-gene relevance summaries
support tables
```

## 11. Recommended reviewer workflow

A typical reviewer-facing workflow is:

1. Create the computational environment.
2. Prepare genotype, phenotype, GWAS, and annotation files.
3. Run model training.
4. Run SNP prioritization.
5. Run latent-space prediction analyses.
6. Run SNP-to-gene and pathway enrichment analyses.
7. Run GWAS-XAI matched-budget comparisons.
8. Check generated summaries and figures.

