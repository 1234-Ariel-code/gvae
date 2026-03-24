# User manual

## 1. Input data expectations

### Synthetic/public demo
The public demo expects:
- `genotypes.csv`: matrix of shape `(N, D)` with SNP dosages coded as 0, 1, 2
- `phenotype.csv`: one-column phenotype file
- `metadata.json`: task description

### Controlled-access user data
Authorized users can adapt the scripts to use:
- PLINK bed/bim/fam
- CSV dosage matrices
- binary or quantitative phenotype files

## 2. Main workflow

### Step A: train representation models
Run:
```bash
python scripts/train_representation.py --config examples/configs/binary_demo.yaml
```

### Step B: run downstream prediction
Run:
```bash
python scripts/run_downstream_prediction.py --task binary --input_dir outputs/binary_demo
```

### Step C: prioritize SNPs
Run:
```bash
python scripts/prioritize_snps.py --input_dir outputs/binary_demo --top_k 200
```

### Step D: gene/pathway analysis
Run:
```bash
python scripts/run_gene_pathway_analysis.py --input_dir outputs/binary_demo
```

### Step E: build summary figures
Run:
```bash
python scripts/build_paper_like_figures.py --input_dir outputs/binary_demo
```

## 3. Metrics by task

### Binary phenotypes
- AUC
- Accuracy

### Quantitative traits
- R2
- RMSE
- MAE
- Pearson correlation

AUC and accuracy should not be used for the raw quantitative trait itself unless the trait is intentionally converted into a classification problem.

## 4. Streamlit app
Launch:
```bash
streamlit run app/streamlit_app.py
```
