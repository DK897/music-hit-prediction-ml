import yaml, random, numpy as np, os

def read_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass

def ensure_dirs(paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)
