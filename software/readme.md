# gVAE Software Pipeline

<p align="center">
  <strong>Configuration-driven genomic representation learning, SNP prioritization, and biological interpretation.</strong>
</p>

<p align="center">
  <a href="gvae_pipeline.py">
    <img src="https://img.shields.io/badge/pipeline-gvae__pipeline.py-1f77b4?style=flat-square" alt="gVAE pipeline">
  </a>
  <a href="config_template.yaml">
    <img src="https://img.shields.io/badge/config-template-2ca02c?style=flat-square" alt="configuration template">
  </a>
  <a href="config_smoke_test.yaml">
    <img src="https://img.shields.io/badge/smoke-test-ff7f0e?style=flat-square" alt="smoke test">
  </a>
  <a href="../README.md">
    <img src="https://img.shields.io/badge/docs-main%20README-9467bd?style=flat-square" alt="main documentation">
  </a>
</p>

---

## Overview

The `software/` directory provides a user-facing, configuration-driven wrapper around the manuscript-facing gVAE implementation in [`../gvae/`](../gvae/).

Instead of manually calling each script, users can define their analysis once in a YAML configuration file and run selected parts of the gVAE workflow from a single command.

```text
gvae/        core model, XAI, prediction, enrichment, and manuscript-facing scripts
software/    user-facing configuration runner and analysis templates
```

The software pipeline is designed for users who want to run gVAE analyses on their own genotype datasets while keeping the scientific implementation synchronized with the manuscript code.

---

## What the pipeline does

The software pipeline supports the main analysis layers offered by gVAE:

* **Model training** using the shared gVAE architecture.
* **SHAP-based SNP prioritization** from learned latent variables.
* **Latent-space prediction** for classification or regression tasks.
* **SNP-to-gene and pathway enrichment** for biological interpretation.
* **GWAS-XAI matched-budget comparison** for benchmarking prioritized signals.
* **Smoke tests** for installation, imports, and command-line interfaces.

---

## Pipeline files

| File                                                                                                                                                                    | Role                                                                                                      | Description                                                                                                          |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| [![software/gvae\_pipeline.py](https://img.shields.io/badge/software%2Fgvae__pipeline.py-file-555555?style=flat-square)](software/gvae_pipeline.py)                     | ![workflow runner](https://img.shields.io/badge/workflow-runner-1f77b4?style=flat-square)                 | Configuration-driven pipeline runner for training, SNP attribution, prediction, enrichment, and GWAS-XAI comparison. |
| [![software/config\_template.yaml](https://img.shields.io/badge/software%2Fconfig__template.yaml-file-555555?style=flat-square)](software/config_template.yaml)         | ![full analysis template](https://img.shields.io/badge/full%20analysis-template-2ca02c?style=flat-square) | Full configuration template for running user-defined gVAE analyses.                                                  |
| [![software/config\_smoke\_test.yaml](https://img.shields.io/badge/software%2Fconfig__smoke__test.yaml-file-555555?style=flat-square)](software/config_smoke_test.yaml) | ![smoke test template](https://img.shields.io/badge/smoke%20test-template-ff7f0e?style=flat-square)       | Minimal configuration for checking installation, imports, and basic execution.                                       |
                                           |
                                           
---

## Quick start

From the repository root, install the package:

```bash
pip install -e .
```

Run a smoke test:

```bash
python software/gvae_pipeline.py \
  --config software/config_smoke_test.yaml \
  --steps smoke
```

Preview the full workflow without executing commands:

```bash
python software/gvae_pipeline.py \
  --config software/config_template.yaml \
  --steps all \
  --dry-run
```

Run selected analysis steps:

```bash
python software/gvae_pipeline.py \
  --config software/config_template.yaml \
  --steps smoke train xai
```

Run the full configured workflow:

```bash
python software/gvae_pipeline.py \
  --config software/config_template.yaml \
  --steps all
```

---

## Available pipeline steps

```text
smoke      check imports and command-line interfaces
train      train gVAE models
xai        run SHAP-based SNP prioritization
predict    run latent-space classification or regression
enrich     run SNP-to-gene, pathway, and disease-gene analysis
gwas_xai   run GWAS versus gVAE-XAI matched-budget comparison
all        run all configured steps in order
```

---

## Recommended workflow

A typical user workflow is:

1. Copy the template configuration file.
2. Update genotype, GWAS, SNP-to-gene, DisGeNET, and output paths.
3. Run the smoke test.
4. Preview commands using `--dry-run`.
5. Run the selected analysis steps.

Example:

```bash
cp software/config_template.yaml software/my_t2d_analysis.yaml

python software/gvae_pipeline.py \
  --config software/my_t2d_analysis.yaml \
  --steps all \
  --dry-run

python software/gvae_pipeline.py \
  --config software/my_t2d_analysis.yaml \
  --steps train xai enrich
```

---

## Configuration structure

The YAML configuration has three main sections:

```text
project    repository root and executable settings
analysis   disease label, model parameters, input paths, and output paths
steps      step-specific options and optional overrides
```

The most commonly edited fields are:

```yaml
disease: T2D
latent_dim: 100
num_samples: 150
num_layers: 4

bed_prefix: /path/to/plink/T2D
base_path: /path/to/genotype/files
tped_file: /path/to/T2D_origin.tped
bim_file: /path/to/T2D.bim
gwas_assoc_path: /path/to/T2D_gwas.assoc

s2g_path: /path/to/snp_to_gene.tsv
disgenet_tsv: /path/to/disgenet.tsv
```

For PLINK BED input, provide the prefix without the file extension. For example, if the files are:

```text
/path/to/plink/T2D.bed
/path/to/plink/T2D.bim
/path/to/plink/T2D.fam
```

then use:

```yaml
bed_prefix: /path/to/plink/T2D
```

---

## Input requirements

The full workflow may require the following local files, depending on which steps are selected:

```text
Genotype data:
  <DISEASE>.bed
  <DISEASE>.bim
  <DISEASE>.fam
  <DISEASE>_filtered.csv

Variant annotation:
  <DISEASE>_origin.tped
  <DISEASE>.bim

GWAS summary statistics:
  <DISEASE>_gwas.assoc

Biological resources:
  SNP-to-gene mapping table
  GMT pathway files or Enrichr libraries
  DisGeNET TSV file
```

Users can run only the steps relevant to the files they have. For example, `train` requires PLINK BED input, while `xai`, `predict`, and `enrich` require additional genotype, GWAS, or annotation files depending on the selected workflow.

---

## Output organization

The pipeline writes outputs to the directories defined in the YAML configuration. Typical outputs include:

```text
outputs/model/
  trained model weights
  reconstruction summaries
  robustness summaries
  latent representations

outputs/xai/
  top SNPs per latent variable
  SNP attribution summaries
  q25/q75 latent feature files
  SHAP-weighted genotype matrices

outputs/prediction/
  classification or regression metrics
  training histories
  latent feature caches
  performance plots

outputs/enrichment/
  SNP-to-gene mapped tables
  pathway enrichment tables
  LV-by-pathway heatmaps
  DisGeNET disease-gene relevance summaries
  target-support tables
```

---

## Design principle

The software pipeline is intentionally thin.

It coordinates the analysis, reads user configuration, logs the commands, and calls the manuscript-facing scripts. The scientific implementation remains in the [`../gvae/`](../gvae/) package so that the software pipeline and the manuscript code stay synchronized.

This design keeps the repository organized into two clear layers:

```text
Backend scientific implementation:
  ../gvae/

User-facing workflow interface:
  ./software/
```

---

## Minimal example

For a new disease or trait, a user can prepare a configuration file, then run:

```bash
python software/gvae_pipeline.py \
  --config software/my_analysis.yaml \
  --steps smoke train xai enrich
```

This provides a single entry point for moving from genotype input to latent representation learning, SNP prioritization, and biological interpretation.
