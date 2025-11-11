import yaml, os, random, numpy as np
def read_config(path): 
    with open(path,'r') as f: return yaml.safe_load(f)
def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass
