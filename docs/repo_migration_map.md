# Repository migration map for the manuscript codebase

This document shows how the original analysis scripts should be split when moving the full internal pipeline into the public repository.

## Original representation-learning script
Suggested public split:
- `gvae/models.py`
- `gvae/train.py`
- `scripts/train_representation.py`
- optional HPC version: `scripts/hpc/train_representation.slurm`

## Original downstream classification/regression script
Suggested public split:
- `gvae/downstream.py`
- `scripts/run_downstream_prediction.py`
- optional HPC version: `scripts/hpc/run_downstream_prediction.slurm`

## Original SHAP SNP-prioritization script
Suggested public split:
- `gvae/xai.py`
- `scripts/prioritize_snps.py`

## Original biology pipeline
Suggested public split:
- `gvae/biology.py`
- `scripts/run_gene_pathway_analysis.py`
- optional HPC version: `scripts/hpc/run_gene_pathway_analysis.slurm`

## Original figure scripts
Suggested public split:
- `figure_scripts/figure2_robustness_distributions.R`
- `figure_scripts/figure3_reconstruction_and_prediction.R`
- `figure_scripts/figure4_gene_pathway_relevance.R`
- `figure_scripts/figure5_dynamic_representation_vs_gwas.R`

## Streamlit app
Public entry point:
- `app/streamlit_app.py`
