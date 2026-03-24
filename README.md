# gVAE: Genomic Variational Autoencoder

Public, user-friendly repository scaffold for the gVAE manuscript.

<img width="1376" height="768" alt="image" src="https://github.com/user-attachments/assets/239fd223-b02d-49f1-88d9-f6e70664ce5f" />

This repository is organized in **two layers**:

1. **Demo layer**: fully runnable on synthetic genotype cohorts, with binary or quantitative traits, model comparison, SNP prioritization, mock biology summaries, and a Streamlit explorer.
2. **Controlled-data analysis layer**: the actual analysis and figure-generation scripts used for the manuscript workflow, adapted into a public repository structure. These scripts require user-supplied genotype/phenotype files and exported summary tables because real patient data cannot be shared publicly.

## What is included

## Exact code preservation

The exact scripts we run on real data analysis is reported at `archive/`. 


- a runnable Python package under `gvae/`
- a Streamlit app under `app/streamlit_app.py`
- synthetic data generation and end-to-end demo scripts under `scripts/`
- manuscript figure scripts under `archive/`
- the original analysis scripts and SLURM examples under `scripts/internal/` and `scripts/hpc/`

## Repository structure

```text
.
├── app/
│   └── streamlit_app.py
├── examples/
│   ├── configs/
│   ├── manuscript_summary_tables/
│   └── synthetic_data/
├── figure_scripts/
│   ├── figure2_robustness_distributions.R
│   ├── figure3_reconstruction_and_prediction.R
│   ├── figure4_gene_pathway_relevance.R
│   └── figure5_dynamic_representation_vs_gwas.R
├── gvae/
├── scripts/
│   ├── generate_synthetic_data.py
│   ├── run_demo_pipeline.py
│   ├── train_representation.py
│   ├── run_downstream_prediction.py
│   ├── prioritize_snps.py
│   ├── run_gene_pathway_analysis.py
│   ├── internal/
│   └── hpc/
└── README.md
```

## User quick start

### 1. Create an environment

```bash
conda env create -f environment.yml
conda activate gvae
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. Run the synthetic demo pipeline

Binary cohort:

```bash
python scripts/generate_synthetic_data.py \
  --task binary \
  --out_dir examples/synthetic_data/binary_demo \
  --n_samples 600 \
  --n_snps 2000 \
  --n_causal 40

python scripts/run_demo_pipeline.py \
  --task binary \
  --data_dir examples/synthetic_data/binary_demo \
  --out_dir reviewer_runs/binary_demo \
  --latent_dim 16 \
  --num_samples 10 \
  --depth 2 \
  --epochs 8
```

Quantitative cohort:

```bash
python scripts/generate_synthetic_data.py \
  --task quantitative \
  --out_dir examples/synthetic_data/quant_demo \
  --n_samples 600 \
  --n_snps 2000 \
  --n_causal 40

python scripts/run_demo_pipeline.py \
  --task quantitative \
  --data_dir examples/synthetic_data/quant_demo \
  --out_dir reviewer_runs/quant_demo \
  --latent_dim 16 \
  --num_samples 10 \
  --depth 2 \
  --epochs 8
```

This produces:

- representation metrics
- downstream prediction metrics
- top SNP summaries
- mock gene and pathway summaries
- PNG plots for reviewer inspection

The synthetic data generator now also exports a **PLINK-compatible bundle** so we can mimic the real internal workflow on synthetic cohorts:

- `demo.bed`, `demo.bim`, `demo.fam`
- `demo_origin.tped`, `demo_origin.tfam`, `demo_origin.phen`
- `demo_filtered.csv`
- `demo_gwas.assoc` for binary traits or `demo_gwas.qassoc` for quantitative traits

That means the public synthetic data can be used either with the lightweight user demo scripts or with the more pipeline-faithful internal scripts under `scripts/internal/`.

### 3. Launch the interactive Streamlit app

```bash
streamlit run app/streamlit_app.py
```

The app lets users choose:

- cohort type: binary or quantitative
- sample size
- number of SNPs
- number of causal SNPs
- latent dimension
- number of posterior samples for gVAE
- network depth
- training epochs

Then it runs a synthetic end-to-end analysis live in the browser.

## Controlled-data analysis layer

The files under `archive/` contain the actual manuscript analysis code. They are included for transparency and reproducibility, but they require:

- user-supplied genotype and phenotype files
- user-supplied GWAS summary statistics where applicable
- local or cluster-specific path edits
- controlled-data permissions

These scripts are not expected to run out of the box on GitHub-hosted public data because the real patient data cannot be distributed.

## Figure generation

The R scripts in `figure_scripts/` now contain real plotting code rather than placeholders.

- `figure2_robustness_distributions.R`
- `figure3_reconstruction_and_prediction.R`
- `figure4_gene_pathway_relevance.R`
- `figure5_dynamic_representation_vs_gwas.R`

Each script is written to support a public-repo workflow using relative paths or command-line arguments. Example summary tables for demonstration are stored in `examples/manuscript_summary_tables/`.

## Notes on privacy

This public repository does **not** include real genotype or phenotype data. The synthetic demo is intended to show how the software behaves and how the manuscript workflow is organized without exposing patient data.

## Citation

Please cite the manuscript and this repository if you use the code.
