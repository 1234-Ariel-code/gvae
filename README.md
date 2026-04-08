# gVAE: Genomic Variational Autoencoder

Public repository for **gVAE**, a representation-learning framework for high-dimensional genomic data with limited sample sizes.

gVAE extends the variational autoencoder framework by drawing multiple latent samples from the posterior distribution and integrating them through a quantile-guided aggregation step before reconstruction and downstream analysis. The method was developed to improve latent-space stability while preserving biological and disease-relevant structure for reconstruction, classification, and interpretation.

<img width="1376" height="768" alt="gVAE overview" src="https://github.com/user-attachments/assets/f4428d27-8fbd-4ee0-a365-c00bfb8c16a2" />

---

## Repository overview

This repository is organized into **two complementary layers**:

### 1. Public demo / package layer
This layer is intended for public use, lightweight testing, and software exploration. It includes:

- a runnable Python package under `gvae/`
- synthetic data generation under `scripts/`
- a lightweight end-to-end demo workflow
- a Streamlit app under `app/streamlit_app.py`

This layer exists because the real genotype datasets used in the manuscript cannot be redistributed publicly.

### 2. Controlled-data manuscript layer
This layer contains the real-data experiment logic used for the manuscript and is included for transparency and reproducibility. It includes:

- manuscript-oriented experiment scripts under `archive/`
- figure-generation scripts
- manuscript-specific evaluation and downstream analysis code
- SLURM job scripts and HPC-oriented workflows

These scripts generally require:
- user-supplied genotype and phenotype files,
- controlled-data permissions,
- local or cluster-specific path adaptation,
- substantial computational resources depending on the task.

---

## Important clarification: where should users start?

Different users should start in different places depending on their goal.

### If you want to explore the software/package
Start with:

- `gvae/`
- `scripts/`
- `app/streamlit_app.py`

### If you want to understand the manuscript experiment workflow
Start with:

- `archive/`

### If you want to reproduce manuscript figures
Start with:

- `archive/` and any figure-generation scripts associated with the manuscript workflow

---

## Repository structure

```text
gvae/               Python package and public-facing software layer
app/                Streamlit demo app
scripts/            Synthetic demo scripts and public lightweight workflows
archive/            Manuscript-oriented real-data analysis code
figures/            Publication figure-generation scripts (if present)
reproducibility.md  Notes on public vs controlled-data reproducibility
README.md           Repository landing page
```

---

## What is included

<img width="1376" height="768" alt="Repository structure" src="https://github.com/user-attachments/assets/d8b23e99-01d9-4e8f-a79f-d9510bc7370a" />

This public repository includes:

- the gVAE package implementation
- synthetic binary and quantitative genotype demos
- comparison workflows for gVAE, BaselineVAE, and BetaVAE
- downstream prediction and representation summaries
- mock SNP, gene, and pathway outputs for demonstration
- an interactive Streamlit interface
- manuscript-oriented scripts for controlled-data analyses
- reproducibility notes and supporting documentation

---

## Quick start

## 1. Create an environment

Using Conda:

```bash
conda env create -f environment.yml
conda activate gvae
```

Or using pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

---

## 2. Run the synthetic demo pipeline

### Binary cohort

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
  --out_dir user_runs/binary_demo \
  --latent_dim 16 \
  --num_samples 10 \
  --depth 2 \
  --epochs 8
```

### Quantitative cohort

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
  --out_dir user_runs/quant_demo \
  --latent_dim 16 \
  --num_samples 10 \
  --depth 2 \
  --epochs 8
```

These public demo workflows produce example outputs such as:

- representation metrics
- downstream prediction metrics
- top SNP summaries
- mock gene and pathway summaries
- PNG plots

---

## 3. Synthetic PLINK-compatible outputs

The synthetic generator also exports a **PLINK-compatible bundle** so that users can mimic parts of the internal workflow on public synthetic cohorts.

Typical outputs include:

- `demo.bed`, `demo.bim`, `demo.fam`
- `demo_origin.tped`, `demo_origin.tfam`, `demo_origin.phen`
- `demo_filtered.csv`
- `demo_gwas.assoc` for binary traits or `demo_gwas.qassoc` for quantitative traits

This means the public synthetic data can be used both for:

- lightweight public demos, and
- more pipeline-faithful testing of the manuscript-style workflow

without exposing real patient data.

---

## 4. Launch the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

The app allows users to interactively vary settings such as:

- cohort type
- sample size
- number of SNPs
- number of causal SNPs
- latent dimension
- number of posterior samples
- network depth
- training epochs

It is intended for **exploration and demonstration**, not for full manuscript-scale training on large controlled datasets.

---

## Controlled-data manuscript layer

The code under `archive/` contains the manuscript-oriented workflows used on real genomic datasets.

This layer is included for transparency, but it is not expected to run out of the box in a public GitHub setting because it depends on:

- controlled-access genotype data,
- protected phenotype files,
- local/HPC file paths,
- manuscript-specific outputs,
- user authorization for restricted datasets.

In other words:

> the public repository demonstrates the software and workflow structure, while the full paper-scale analyses require controlled data and appropriate compute infrastructure.

---

## Compute requirements

The repository contains both lightweight and heavy workflows.

### Lightweight / local usage
Usually suitable for:
- code inspection,
- synthetic demos,
- Streamlit exploration,
- small test runs.

### HPC / large-resource usage
Often required for:
- large genomic training runs,
- broad hyperparameter sweeps,
- manuscript-scale controlled-data analyses,
- some downstream interpretation pipelines.

Some SLURM scripts used in the manuscript workflow request high memory because of the dimensionality of the real genomic datasets and the scale of the analyses.

---

## Why simulation code is present

Some code in the public package or scripts may appear simulation-oriented. This is intentional.

Because the real patient-level genotype data used in the manuscript cannot be shared publicly, the repository includes simulation and synthetic-cohort workflows so that users can still:

- verify the software organization,
- run the pipeline end-to-end,
- inspect outputs,
- explore the method behavior.

---

## gVAE vs qgVAE naming

Some legacy code, scripts, or filenames may still refer to **qgVAE**.

In this repository:

- **gVAE** is the manuscript-facing name,
- **qgVAE** reflects an earlier/internal development naming convention.

They generally refer to the same methodological family unless explicitly distinguished.

---

## Documentation

For a clearer explanation of the repository layout and intended workflows, please see the project wiki.

The wiki explains:

- repository structure,
- entry points,
- compute requirements,
- simulation vs real-data usage,
- SLURM/HPC logic,
- naming conventions,
- reproducibility guidance.

---

## Data privacy

This public repository does **not** include real genotype or phenotype data.

The public demo layer is designed to show:
- how the method works,
- how the workflows are organized,
- how analyses are launched,

without exposing controlled human genomic data.

---

## Citation

Please cite the associated manuscript and this repository if you use this code.

See:

- `CITATION.cff`
- the accompanying manuscript
