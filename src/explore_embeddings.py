import pandas as pd
from pathlib import Path

# 1. Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EMB_FILE = DATA_DIR / "embedding_query2product.parquet"

print("Loading:", EMB_FILE)

# 2. Load parquet
df = pd.read_parquet(EMB_FILE)

# 3. Basic info
print("Number of rows:", len(df))
print("Columns:", list(df.columns))
print("\nFirst 5 rows:")
print(df.head())

# 4. Create relevance label: 1 if clicked at least once, else 0
df["relevant"] = (df["UniqueSelects"] > 0).astype(int)

print("\nLabel counts (0 = not clicked, 1 = clicked):")
print(df["relevant"].value_counts())
