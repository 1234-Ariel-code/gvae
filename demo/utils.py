from __future__ import annotations
import json, os, random
import numpy as np
import tensorflow as tf

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
