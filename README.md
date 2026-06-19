
# gVAE: Genomic Variational Autoencoder

<p align="center">
  <img width="1376" height="768" alt="gVAE overview image" src="https://github.com/user-attachments/assets/f4428d27-8fbd-4ee0-a365-c00bfb8c16a2" />
</p>

<p align="center">
  <strong>Stable and interpretable genomic representation learning for high-dimensional genotype data with moderate sample sizes.</strong>
</p>

> Representation learning is powerful for genomic discovery, but population-scale genotype datasets remain challenging because they combine very high dimensionality with moderate sample sizes. gVAE addresses this setting by using a quantile-gated variational autoencoder that integrates multiple posterior latent realizations into a more stable representation. Across 18 disease and trait datasets, gVAE improves latent stability and predictive performance relative to established VAE baselines while preserving reconstruction fidelity. Coupled with explainable AI, SNP-to-gene mapping, and pathway analysis, gVAE recovers disease- and drug-relevant signals, recurrent pathway programs, and interpretable disease mechanisms, positioning its latent variables as biological coordinates rather than opaque compression axes.

---

## About

This repository contains the manuscript-facing implementation of **gVAE**, a genomic variational autoencoder framework for stable and interpretable representation learning in high-dimensional genotype data with moderate sample sizes.

The code supports model training, latent representation extraction, SNP prioritization, downstream prediction, SNP-to-gene mapping, pathway enrichment, disease-gene relevance analysis, and reproducibility utilities used in the accompanying manuscript.

---

## Overview

<p align="center">
  <img width="1300" height="1150" alt="gVAE workflow overview" src="https://github.com/user-attachments/assets/086ca913-5a1e-4477-bcd3-efc68a30ed2f" />
</p>

Genome-wide genotype matrices are high-dimensional, sparse, and often available for cohorts with limited sample sizes. The goal of this repository is to provide a reproducible implementation of a representation learning workflow that:

1. trains VAE/gVAE models on genotype data,
2. extracts stable latent representations,
3. identifies latent-variable-associated SNPs using attribution methods,
4. maps prioritized SNPs to genes,
5. evaluates disease relevance and drug-target support,
6. performs pathway enrichment analysis, and
7. supports downstream classification and regression analyses.


---

## Repository structure

The repository is organized around a small set of manuscript-facing scripts, shared model utilities, reproducibility files, and cluster execution templates.

### Core Python package

The main implementation lives in [`gvae/`](gvae/).

```text
gvae/
```

**Shared architecture and utilities**

* [`gvae/__init__.py`](gvae/__init__.py) — package initializer.
* [`gvae/model.py`](gvae/model.py) — shared gVAE, Vanilla VAE, and beta-VAE model definitions.
* [`gvae/metrics.py`](gvae/metrics.py) — shared reconstruction and prediction metric utilities.

**Model training and representation learning**

* [`gvae/gvae.py`](gvae/gvae.py) — main model-training entry point.
* [`gvae/latent_classification.py`](gvae/latent_classification.py) — downstream classification and regression from latent features.

**Interpretability and biological analysis**

* [`gvae/snp_prioritization.py`](gvae/snp_prioritization.py) — SHAP-based SNP prioritization from latent variables.
* [`gvae/gene-pathway_enrichment.py`](gvae/gene-pathway_enrichment.py) — SNP-to-gene mapping, pathway enrichment, and disease-gene relevance analysis.
* [`gvae/build_target_support_table.py`](gvae/build_target_support_table.py) — gene-level disease and drug-target support summaries.
* [`gvae/gwas-xai.R`](gvae/gwas-xai.R) — matched-budget comparison of GWAS-ranked and gVAE-XAI-prioritized signals.

**Cluster execution templates**

* [`gvae/gvae.slurm`](gvae/gvae.slurm) — SLURM template for model training. This is an example cluster submission script for model training.
* [`gvae/gene-pathway_enrichment.slurm`](gvae/gene-pathway_enrichment.slurm) — SLURM template for enrichment analysis. This is an example cluster submission script for gene/pathway enrichment.
* [`gvae/gwas-xai.slurm`](gvae/gwas-xai.slurm) — SLURM template for GWAS-XAI comparison. This is an example cluster submission script for GWAS-XAI comparison.

These files provide example cluster-job configurations and should be edited to match the local computing environment, data paths, memory limits, and runtime requirements.

### Repository-level files

* [`README.md`](README.md) — repository overview and usage guide.
* [`reproducibility.md`](reproducibility.md) — reproducibility notes and recommended workflow.
* [`requirements.txt`](requirements.txt) — Python package requirements.
* [`environment.yml`](environment.yml) — Conda environment specification.
* [`pyproject.toml`](pyproject.toml) — package metadata and editable-install configuration.
* [`Makefile`](Makefile) — helper commands for installation, cleanup, and checks.
* [`CITATION.cff`](CITATION.cff) — citation metadata.
* [`LICENSE`](LICENSE) — software license.

---

## Installation

The repository can be installed using either `conda` or `pip`.

### Option 1: Conda environment

```bash
conda env create -f environment.yml
conda activate gvae
```

### Option 2: Python requirements

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Option 3: Editable package installation

```bash
pip install -e .
```

After installation, the main Python workflows can be run from the repository root using module-style commands such as:

```bash
python -m gvae.gvae
python -m gvae.snp_prioritization
```

---

## Quick smoke test

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

## Expected data inputs

The scripts are designed for genotype and annotation files commonly used in genome-wide association studies and genomic representation learning.

Typical inputs include:

```text
Genotype matrix:
  <DISEASE>_filtered.csv

Phenotype file:
  <DISEASE>_origin.phen

Variant annotation files:
  <DISEASE>_origin.tped
  <DISEASE>.bim

GWAS association file:
  <DISEASE>_gwas.assoc

SNP-to-gene mapping file:
  cS2G or other SNP-to-gene mapping table

Pathway resources:
  Enrichr libraries or GMT files

Disease-gene resources:
  DisGeNET TSV file or API access
```

The exact file paths should be adjusted in the command-line arguments or SLURM scripts.

---

## Example workflows

### 1. Train gVAE model

```bash
python -m gvae.gvae \
  --disease T2D \
  --bed_prefix /path/to/plink/T2D \
  --latent_dim 100 \
  --num_sample 150 \
  --num_layer 4 \
  --epochs 50 \
  --batch_size 256
```

### 2. Prioritize SNPs using latent-variable attribution

```bash
python -m gvae.snp_prioritization \
  --disease T2D \
  --base_path /path/to/genotype/files \
  --latent_dim 100 \
  --num_samples 150 \
  --num_layers 4 \
  --shap_top_k 10 \
  --tped_file /path/to/T2D_origin.tped \
  --output_dir /path/to/xai_outputs
```

### 3. Run latent-space classification or regression

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

### 4. Run SNP-to-gene and pathway enrichment analysis

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

---

## Output structure

Depending on the script, outputs may include:

```text
Model outputs:
  trained model weights
  reconstruction summaries
  robustness summaries
  latent representations

XAI outputs:
  top SNPs per latent variable
  SNP attribution summaries
  q25/q75 latent feature files
  SHAP-weighted genotype matrices

Prediction outputs:
  classification or regression metrics
  training histories
  latent feature caches
  performance plots

Gene/pathway outputs:
  SNP-to-gene mapped tables
  pathway enrichment tables
  LV-by-pathway heatmaps
  LV bubble plots
  DisGeNET disease-gene relevance summaries
  target-support tables
```

---

## Reproducibility

The repository includes:

```text
reproducibility.md
environment.yml
requirements.txt
pyproject.toml
Makefile
```

These files document the computational environment and provide installation or workflow helpers. Paths in the example scripts and SLURM files should be adjusted to the local system.

For reviewer-facing reproducibility, the recommended workflow is:

1. create the documented environment,
2. prepare genotype, phenotype, GWAS, and annotation files,
3. run the model-training script,
4. run SNP attribution,
5. run gene/pathway enrichment,
6. run matched downstream analyses or support-table construction.

---

## Notes on terminology

Throughout the repository:

* `LV` denotes latent variable.
* `LD` denotes latent dimension in configuration names.
* `NS` denotes the number of posterior latent samples used for gVAE quantile aggregation.
* `L` denotes the number of encoder/decoder hidden layers.
* `shap_top_k` denotes the number of top SNPs retained per latent variable in attribution outputs.
* q25/q75 denote the posterior latent quantiles defining the reported gVAE feature representation.
* GWAS-top SNP filtering denotes structured filtering based on GWAS ranking, not random SNP downsampling.

---

## Citation

Please cite this repository using the metadata in:

```text
CITATION.cff
```

---

## License

This repository is released under the MIT License. See:

```text
LICENSE
```

---

## Contact

For questions about the manuscript code or reproducibility workflow, please contact the repository maintainer through GitHub.


