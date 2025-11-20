import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List


# ----- 1. Paths -----
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

SAMPLE_FILE = DATA_DIR / "embedding_de_sample.parquet"
EMB_FILE = DATA_DIR / "product_embeddings.npz"   # we will create this file


# ----- 2. Helper functions -----
def load_data() -> pd.DataFrame:
    """
    Load the German sample parquet we created earlier.
    """
    if not SAMPLE_FILE.exists():
        raise FileNotFoundError(
            f"{SAMPLE_FILE} not found. Run explore_embeddings.py first to create it."
        )

    print(f"Loading sample data from: {SAMPLE_FILE}")
    df = pd.read_parquet(SAMPLE_FILE)

    # We only need these columns for now
    df = df[["OriginalQuery", "product_text", "relevant"]].copy()

    print("Rows in sample:", len(df))
    return df


def build_or_load_product_embeddings(
    df: pd.DataFrame,
    model: SentenceTransformer,
) -> tuple[np.ndarray, List[str]]:
    """
    If embeddings already exist on disk, load them.
    Otherwise, compute and save them.
    """
    if EMB_FILE.exists():
        print(f"Loading precomputed embeddings from: {EMB_FILE}")
        data = np.load(EMB_FILE, allow_pickle=True)
        product_embeddings = data["embeddings"]
        product_texts = list(data["texts"])
        return product_embeddings, product_texts

    # If we reach here, we have to compute them.
    print("No saved embeddings found, computing them now...")

    # To avoid making it too heavy, we can deduplicate product_texts
    # so the same product is not embedded many times.
    unique_product_texts = df["product_text"].dropna().unique().tolist()
    print("Unique product texts:", len(unique_product_texts))

    # Encode with the model
    product_embeddings = model.encode(
        unique_product_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # makes cosine similarity just dot product
    )

    # Save to disk for later runs
    np.savez_compressed(
        EMB_FILE,
        embeddings=product_embeddings,
        texts=np.array(unique_product_texts, dtype=object),
    )
    print(f"Saved embeddings to: {EMB_FILE}")

    return product_embeddings, unique_product_texts


def search_products(
    query: str,
    model: SentenceTransformer,
    product_embeddings: np.ndarray,
    product_texts: List[str],
    top_k: int = 5,
):
    """
    Encode the query, compute cosine similarity with all product embeddings,
    and print the top_k most similar products.
    """
    # Encode query
    query_emb = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]  # shape: (dim,)

    # Cosine similarity = dot product because we normalized
    scores = product_embeddings @ query_emb  # shape: (num_products,)

    # Get indices of top_k scores
    top_idx = np.argsort(-scores)[:top_k]

    print("\nTop matches:")
    for rank, idx in enumerate(top_idx, start=1):
        print(f"\n#{rank}  (score = {scores[idx]:.3f})")
        print("Product text:", product_texts[idx])


def main():
    # ----- 3. Load data -----
    df = load_data()

    # ----- 4. Load embedding model -----
    # Small and fast sentence embedding model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"\nLoading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    # ----- 5. Build or load embeddings for all products -----
    product_embeddings, product_texts = build_or_load_product_embeddings(df, model)

    # ----- 6. Simple interactive search -----
    print("\nYou can now type queries and see the most similar products.")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Enter a search query: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        search_products(query, model, product_embeddings, product_texts, top_k=5)


if __name__ == "__main__":
    main()
