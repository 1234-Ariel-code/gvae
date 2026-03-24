# Reviewer quick start

## Lightweight demo

Generate a synthetic bundle:

```bash
python scripts/generate_synthetic_data.py   --task binary   --dataset_name demo   --out_dir examples/synthetic_data/binary_demo   --n_samples 600   --n_snps 2000   --n_causal 40
```

This writes both easy-to-read CSVs and pipeline-style files:

- `genotypes.csv`, `phenotype.csv`
- `demo_filtered.csv`
- `demo_origin.phen`, `demo_origin.tped`, `demo_origin.tfam`
- `demo.bed`, `demo.bim`, `demo.fam`
- `demo_gwas.assoc` or `demo_gwas.qassoc`

Run the reviewer demo:

```bash
python scripts/run_demo_pipeline.py --task binary --data_dir examples/synthetic_data/binary_demo --out_dir reviewer_runs/binary_demo --latent_dim 16 --num_samples 10 --depth 2 --epochs 8
```

## Pipeline-faithful synthetic run

You can also point the internal scripts to the synthetic exports so the workflow matches the controlled-data pipeline more closely. For example, after generating `demo.bed/.bim/.fam` and `demo_origin.phen`:

```bash
python scripts/internal/qgvae_bed.py   --disease demo   --num_sample 10   --latent_dim 16   --num_layer 2   --epochs 8   --batch_size 32   --bed_prefix examples/synthetic_data/binary_demo/demo   --beta_list 1.0,4.0
```

For the classification-style internal script, the synthetic bundle also includes the matching `demo_filtered.csv`, `demo_origin.phen`, `demo_origin.tped`, and `demo_gwas.assoc` files.
