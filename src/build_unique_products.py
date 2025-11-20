import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_FILE = DATA_DIR / "embedding_query2product.parquet"
OUT_FILE = DATA_DIR / "unique_products.parquet"


def main():
    print("Loading full dataset:", RAW_FILE)
    df = pd.read_parquet(RAW_FILE)
    print("Rows in original file:", len(df))

    # Keep ONLY product-related fields
    keep_cols = [
        "ProductName",
        "BrandName",
        "ProductType",
        "Level4_ProductGroup",
        "PageLanguage",
    ]

    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in source data: {missing}")

    prod_df = df[keep_cols].copy()

    # Create a unique product key (name + brand + type)
    prod_df["unique_key"] = (
        prod_df["ProductName"].fillna("") + " / " +
        prod_df["BrandName"].fillna("") + " / " +
        prod_df["ProductType"].fillna("")
    )

    # Drop duplicates
    prod_df = prod_df.drop_duplicates(subset=["unique_key"]).reset_index(drop=True)
    print("Unique products:", len(prod_df))

    # Build product_text properly
    prod_df["product_text"] = (
        prod_df["ProductName"].fillna("") + " " +
        prod_df["BrandName"].fillna("") + " " +
        prod_df["ProductType"].fillna("") + " " +
        prod_df["Level4_ProductGroup"].fillna("")
    ).str.strip()

    # Save
    prod_df.to_parquet(OUT_FILE)
    print("Saved unique product file →", OUT_FILE)


if __name__ == "__main__":
    main()
