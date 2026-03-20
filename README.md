# gVAE

**Genomic Variational Autoencoder for stable and interpretable representation learning in high-dimensional genomic data with small sample sizes**

<img width="1376" height="768" alt="canvas-image-1-1773162105249" src="https://github.com/user-attachments/assets/aedc1ad9-923d-437c-b02a-2efd193dc951" />


This repository accompanies the manuscript:

**Genomic Variational Autoencoder enables stable representation learning in high-dimensional genomic data with small sample sizes**

---

## Overview

Genomic data are challenging to model because they combine **extreme dimensionality** with **limited sample sizes**, often resulting in unstable representation learning and reduced biological interpretability. To address this, we introduce **gVAE (Genomic Variational Autoencoder)**, a **quantile-gated variational autoencoder** that draws multiple latent samples from the posterior distribution and integrates them through quantile-guided aggregation.

Unlike standard VAE implementations that typically rely on a single posterior draw, gVAE leverages multiple latent realizations to better use posterior uncertainty, improve representation stability, and preserve biologically meaningful variation. The framework is further coupled with **explainable artificial intelligence (XAI)**, **SNP-to-gene mapping**, **gene relevance analysis**, **pathway enrichment**, and **GWAS-vs-XAI comparison** to make the learned latent space biologically interpretable.

Across 18 genomic datasets, gVAE shows improved latent robustness, favorable reconstruction behavior, competitive downstream disease classification, and strong biological coherence.

---

## Key contributions

- Introduces **gVAE**, a quantile-gated VAE designed for **high-dimensional, small-sample genomic data**
- Uses **multiple posterior latent samples** rather than a single draw
- Aggregates latent samples through **quantile-guided gating**
- Improves **representation robustness** under genotype perturbation
- Maintains favorable **reconstruction quality** using metrics such as **R²** and **MSE**
- Supports downstream **disease classification** using learned latent representations
- Enables biological interpretation through:
  - **SHAP-based SNP attribution**
  - **SNP prioritization**
  - **SNP-to-gene mapping**
  - **gene relevance analysis**
  - **pathway enrichment**
- Compares latent-space biological discovery with **conventional GWAS-based prioritization**

---

## Repository structure

```text
gvae/
├── README.md
├── LICENSE
├── CITATION.cff
├── environment.yml
├── requirements.txt
├── .gitignore
├── reproducibility.md
├── Makefile
│
├── configs/
│   ├── datasets/
│   ├── models/
│   └── experiments/
│
├── data/
│   ├── README.md
│   ├── sample_data/
│   └── templates/
│
├── src/
│   └── gvae/
│       ├── data/
│       ├── models/
│       ├── training/
│       ├── evaluation/
│       ├── xai/
│       ├── biology/
│       ├── gwas/
│       ├── figures/
│       └── utils/
│
├── scripts/
│   ├── train_gvae.py
│   ├── train_baselines.py
│   ├── evaluate_reconstruction.py
│   ├── evaluate_robustness.py
│   ├── run_classification.py
│   ├── run_shap_prioritization.py
│   ├── run_snp_to_gene.py
│   ├── run_enrichment.py
│   ├── run_gwas_vs_xai.py
│   ├── build_main_figures.py
│   └── build_supplementary_outputs.py
│
├── slurm/
│   ├── preprocess_qc_ld.slurm
│   ├── train_gvae.slurm
│   ├── train_baselines.slurm
│   ├── run_shap_prioritization.slurm
│   ├── run_snp_to_gene.slurm
│   ├── run_enrichment.slurm
│   ├── run_gwas_vs_xai.slurm
│   ├── run_classification.slurm
│   └── build_figures.slurm
│
├── paper/
│   ├── manuscript/
│   ├── main_figures/
│   ├── supplementary_information/
│   ├── supplementary_methods/
│   └── source_data/
│
├── docs/
│   ├── reviewer_guide.md
│   ├── reproduction_guide.md
│   ├── methodology.md
│   └── output_index.md
│
└── tests/
