import pandas as pd
from pathlib import Path


# ----- 1. Paths -----
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EMB_FILE = DATA_DIR / "embedding_query2product.parquet"


def main():
    # ----- 2. Load parquet -----
    print("Loading:", EMB_FILE)
    df = pd.read_parquet(EMB_FILE)

    # ----- 3. Basic info -----
    print("Number of rows:", len(df))
    print("Columns:", list(df.columns))
    print("\nFirst 5 rows:")
    print(df.head())

    # ----- 4. Create relevance label -----
    # relevant = 1 if product was clicked at least once, else 0
    df["relevant"] = (df["UniqueSelects"] > 0).astype(int)

    print("\nLabel counts (0 = not clicked, 1 = clicked):")
    print(df["relevant"].value_counts())

    # ----- 5. Focus on German queries only -----
    df = df[df["PageLanguage"] == "de"].copy()
    print("\nAfter filtering to German (de):", len(df), "rows")

    # ----- 6. Build product_text -----
    # This is the text description we will later feed into an embedding model
    df["product_text"] = (
        df["ProductName"].fillna("") + " " +
        df["BrandName"].fillna("") + " " +
        df["Level4_ProductGroup"].fillna("") + " " +
        df["ProductType"].fillna("")
    )

    # ----- 7. Show some examples -----
    print("\nExample query / product_text pairs:")
    for i in range(5):
        row = df.iloc[i]
        print(f"\nRow {i}")
        print("  Query:   ", row["OriginalQuery"])
        print("  Product: ", row["product_text"])
        print("  Clicked? relevant =", row["relevant"])
        print("  ----------")

    # ----- 8. Save a smaller sample for experiments -----
    sample_size = 200_000  # you can change this later
    if len(df) < sample_size:
        sample_size = len(df)

    sample = df.sample(n=sample_size, random_state=42)
    out_file = DATA_DIR / "embedding_de_sample.parquet"
    sample.to_parquet(out_file)
    print(f"\nSaved a sample of {len(sample)} rows to {out_file}")


if __name__ == "__main__":
    main()
