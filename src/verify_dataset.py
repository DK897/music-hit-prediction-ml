import pandas as pd
import numpy as np
import sys, os

path = "data/processed/dataset.csv"

# 1️⃣ Check file existence
if not os.path.exists(path):
    sys.exit(f"❌ Missing dataset file: {path}. Please run prepare_dataset.py first.")

# 2️⃣ Load dataset
df = pd.read_csv(path)
print(f"📦 Loaded dataset: {df.shape[0]} samples, {df.shape[1]} columns")

# 3️⃣ Check for target/label column
if 'target' not in df.columns:
    sys.exit("❌ Dataset missing required 'target' column. Please ensure prepare_dataset.py renamed it correctly.")

# 4️⃣ Check all numeric
non_numeric_cols = [col for col, dtype in df.dtypes.items() if not np.issubdtype(dtype, np.number)]
if non_numeric_cols:
    sys.exit(f"❌ Non-numeric columns found: {non_numeric_cols}. Please re-run prepare_dataset.py to clean them.")

# 5️⃣ Summary print
print(f"✅ Verified dataset is numeric with '{len(df.columns)}' columns including 'target'.")
print("   First 5 columns:", list(df.columns[:5]))
