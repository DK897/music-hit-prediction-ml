#!/usr/bin/env python3
"""
Minimal runner placeholder. Use the full training.py provided in the repo later.
"""
import argparse, os
from src.utils import read_config, seed_everything
def main(config, out):
    cfg = read_config(config)
    seed_everything(cfg.get('dataset',{}).get('seed',42))
    print("Config loaded. This is a placeholder training runner.")
if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config'); p.add_argument('--out')
    args=p.parse_args(); main(args.config, args.out)
