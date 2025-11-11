#!/usr/bin/env python3
"""
Simple feature extraction placeholder.
Replace with full librosa extraction when ready.
"""
import os, sys, argparse, pandas as pd
def main(wavs_dir, out_csv):
    # placeholder: create empty CSV if none
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df = pd.DataFrame()
    df.to_csv(out_csv)
    print("Wrote placeholder features to", out_csv)

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--wavs_dir'); p.add_argument('--out_csv')
    args=p.parse_args()
    main(args.wavs_dir, args.out_csv)
