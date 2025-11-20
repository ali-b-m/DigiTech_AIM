import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FILE = DATA_DIR / "unique_products.parquet"   # or embedding_query2product.parquet

KEYWORDS = ["vibrator", "dildo", "erotik", "sextoy", "penispumpe", "buttplug"]

def main():
    print("Loading:", FILE)
    df = pd.read_parquet(FILE)
    print("Rows:", len(df))

    # Make a lower-case version for searching in text
    if "product_text" not in df.columns:
        df["product_text"] = (
            df["ProductName"].fillna("") + " " +
            df["BrandName"].fillna("") + " " +
            df["ProductType"].fillna("") + " " +
            df["Level4_ProductGroup"].fillna("")
        ).str.strip()

    df["product_text_lower"] = df["product_text"].str.lower()

    mask = False
    for kw in KEYWORDS:
        mask |= df["product_text_lower"].str.contains(kw)

    adult_df = df[mask]
    print("Number of 'adult' products found:", len(adult_df))

    # Show which ProductType / Level4 groups they are in
    if "ProductType" in adult_df.columns:
        print("\nProductType counts:")
        print(adult_df["ProductType"].value_counts().head(20))

    if "Level4_ProductGroup" in adult_df.columns:
        print("\nLevel4_ProductGroup counts:")
        print(adult_df["Level4_ProductGroup"].value_counts().head(20))

    # Optional: save them to inspect in VS Code
    out = DATA_DIR / "adult_products_sample.parquet"
    adult_df.to_parquet(out)
    print("\nSample saved to:", out)


if __name__ == "__main__":
    main()
