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

<img width="1376" height="768" alt="canvas-image-1-1773694468470" src="https://github.com/user-attachments/assets/4086bfe7-3c42-46ed-8ef6-30ea17f24b64" />

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
├── scripts/
│   ├── run_qc_and_gvae.py   
│   ├── run_xai_prioritization.py
│   ├── run_gene_pathways_relevance.py  
│   └── run_gwas_vs_xai.py 
│
├── slurm/
│   ├── run_qc_and_gvae.slurm
│   ├── run_xai_prioritization.slurm
│   ├── run_gene_pathways_relevance.slurm
│   └── run_gwas_vs_xai.slurm 
│
├── paper/
│   ├── manuscript/
│   ├── main_figures/
│   ├── supplementary_information/
│   └── supplementary_methods/ 
│
└── tests/
