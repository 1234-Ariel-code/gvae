import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name != "app" else Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
print("Public demo note: downstream prediction is already executed inside scripts/run_demo_pipeline.py. Keep this file as the public CLI entry point placeholder.")
