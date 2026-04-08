# Reproducibility notes

This repository is split into two layers.

## 1. Public reproducibility layer

This is what reviewers and public users can run immediately:

- synthetic binary and quantitative genotype cohorts
- end-to-end gVAE/BaselineVAE/BetaVAE comparison
- reconstruction, robustness, and downstream prediction
- mock SNP prioritization and gene/pathway summaries
- Streamlit exploratory interface

## 2. Controlled-data manuscript layer

The paper results were generated on controlled-access genomic datasets that cannot be redistributed publicly. The public scripts are therefore written so that authorized users can point them to their own genotype matrices and phenotype files after obtaining access to those datasets.

## Public demo target

The public demo is not intended to reproduce every numeric value in the manuscript. Its purpose is to let users verify:

- how the software is organized;
- how gVAE is trained;
- how it is compared to baseline models;
- how downstream analyses are launched;
- how figures and tables are generated.

## Recommended user path

1. Create a synthetic cohort.
2. Run the demo pipeline.
3. Inspect the output metrics and plots.
4. Launch the Streamlit app.
5. Verify that changing sample size, SNP count, task type, and latent settings changes the outputs in a sensible way.
