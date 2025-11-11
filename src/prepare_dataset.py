import pandas as pd, argparse, os

def main(infile, out):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df = pd.read_csv(infile)

    # 1️⃣ Normalize columns
    df.columns = [c.strip() for c in df.columns]

    # 2️⃣ Drop non-numeric columns automatically
    non_numeric = df.select_dtypes(exclude=['number']).columns.tolist()
    if non_numeric:
        print(f"🧹 Dropping non-numeric columns: {non_numeric}")
        df = df.drop(columns=non_numeric)

    # 3️⃣ Rename label column to target if needed
    if 'target' not in df.columns:
        for col in df.columns:
            if col.lower() in ['label', 'hit', 'is_hit']:
                df.rename(columns={col: 'target'}, inplace=True)
                print(f"🪶 Renamed column '{col}' → 'target'")

    # 4️⃣ Drop NA values
    df = df.dropna()
    print(f"✅ Cleaned dataset shape: {df.shape}")

    # 5️⃣ Save cleaned dataset
    df.to_csv(out, index=False)
    print("💾 Saved cleaned dataset to", out)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--infile', default='data/processed/dataset.csv')
    p.add_argument('--out', default='data/processed/dataset.csv')
    args = p.parse_args()
    main(args.infile, args.out)
