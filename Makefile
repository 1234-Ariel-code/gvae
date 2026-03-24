PYTHON ?= python

install:
	$(PYTHON) -m pip install -r requirements.txt

demo-binary:
	$(PYTHON) scripts/generate_synthetic_data.py --task binary --n_samples 600 --n_snps 2000 --n_causal 40 --out_dir outputs/demo_binary_data
	$(PYTHON) scripts/run_demo_pipeline.py --task binary --data_dir outputs/demo_binary_data --out_dir outputs/demo_binary_run

demo-quant:
	$(PYTHON) scripts/generate_synthetic_data.py --task quantitative --n_samples 800 --n_snps 2500 --n_causal 50 --out_dir outputs/demo_quant_data
	$(PYTHON) scripts/run_demo_pipeline.py --task quantitative --data_dir outputs/demo_quant_data --out_dir outputs/demo_quant_run

app:
	streamlit run app/streamlit_app.py

clean:
	rm -rf outputs
