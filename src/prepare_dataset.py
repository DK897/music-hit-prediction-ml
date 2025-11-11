#!/usr/bin/env python3
import pandas as pd, os, argparse
def main(features, labels, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(features):
        f = pd.read_csv(features, index_col=0)
    else:
        f = pd.DataFrame()
    if labels and os.path.exists(labels):
        l = pd.read_csv(labels, index_col=0)
        df = f.join(l, how='inner')
    else:
        df = f
    df.to_csv(out)
    print("Saved dataset to", out)
if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--features'); p.add_argument('--labels', default=None); p.add_argument('--out')
    args=p.parse_args(); main(args.features, args.labels, args.out)
