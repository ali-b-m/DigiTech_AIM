import numpy as np
import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_FILE = DATA_DIR / "embedding_query2product.parquet"

TRAIN_FILE = DATA_DIR / "train.parquet"
VAL_FILE = DATA_DIR / "val.parquet"
TEST_FILE = DATA_DIR / "test.parquet"


def main():
    print("Loading full dataset from:", RAW_FILE)
    df = pd.read_parquet(RAW_FILE)
    print("Total rows:", len(df))

    # ----- 1. Create relevance label -----
    # relevant = 1 if user selected the product at least once
    df["relevant"] = (df["UniqueSelects"] > 0).astype(int)

    # ----- 2. Build product_text (no prefixes yet) -----
    df["product_text"] = (
        df["ProductName"].fillna("") + " "
        + df["BrandName"].fillna("") + " "
        + df["Level4_ProductGroup"].fillna("") + " "
        + df["ProductType"].fillna("")
    ).str.strip()

    # Keep only the columns we really need for retrieval
    df = df[[
        "OriginalQuery",
        "PageLanguage",
        "product_text",
        "relevant",
    ]].copy()

    # Drop rows with missing query or product text
    df = df.dropna(subset=["OriginalQuery", "product_text"])
    print("Rows after dropping empty queries/products:", len(df))

    # ----- 3. Query-level split: each query only in one split -----
    queries = df["OriginalQuery"].unique()
    print("Unique queries:", len(queries))

    rng = np.random.default_rng(seed=42)
    rng.shuffle(queries)

    n = len(queries)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    # Remaining goes to test

    train_queries = set(queries[:n_train])
    val_queries = set(queries[n_train:n_train + n_val])
    test_queries = set(queries[n_train + n_val:])

    print(f"Train queries: {len(train_queries)}")
    print(f"Val queries:   {len(val_queries)}")
    print(f"Test queries:  {len(test_queries)}")

    # Assign split
    df["split"] = "test"  # default
    df.loc[df["OriginalQuery"].isin(train_queries), "split"] = "train"
    df.loc[df["OriginalQuery"].isin(val_queries), "split"] = "val"

    train_df = df[df["split"] == "train"].drop(columns=["split"])
    val_df = df[df["split"] == "val"].drop(columns=["split"])
    test_df = df[df["split"] == "test"].drop(columns=["split"])

    print("Train rows:", len(train_df))
    print("Val rows:  ", len(val_df))
    print("Test rows: ", len(test_df))

    # ----- 4. Save splits -----
    TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(TRAIN_FILE)
    val_df.to_parquet(VAL_FILE)
    test_df.to_parquet(TEST_FILE)

    print("\nSaved:")
    print("  Train ->", TRAIN_FILE)
    print("  Val   ->", VAL_FILE)
    print("  Test  ->", TEST_FILE)


if __name__ == "__main__":
    main()
