import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive search using a bi-encoder model on product_text."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help=(
            "HuggingFace model name or local path to fine-tuned model.\n"
            "Examples:\n"
            "  intfloat/multilingual-e5-base\n"
            "  BAAI/bge-base-en-v1.5\n"
            "  models/e5_multi_base_finetuned\n"
            "  models/gte_multi_base_finetuned"
        ),
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=str(DATA_DIR / "test.parquet"),
        help="Parquet file with products (must have at least 'product_text' column).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="How many top products to show for each query.",
    )
    parser.add_argument(
        "--max_products",
        type=int,
        default=100_000,
        help=(
            "Optional: limit the number of products for demo speed. "
            "If None, use all rows in data_file."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: 'cuda' or 'cpu'. Default: auto-detect.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help=(
            "If set, pass trust_remote_code=True to SentenceTransformer "
            "(needed for some models like Alibaba-NLP/gte-multilingual-base)."
        ),
    )
    args = parser.parse_args()
    return args


def encode_texts(model, texts: List[str]) -> np.ndarray:
    """Encode text with L2-normalized embeddings."""
    return model.encode(
        texts,
        batch_size=128,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def main():
    args = parse_args()

    print("===== INTERACTIVE SEARCH CONFIG =====")
    print(f"Model:       {args.model_name_or_path}")
    print(f"Data file:   {args.data_file}")
    print(f"Top-K:       {args.top_k}")
    print(f"Max products:{args.max_products}")
    print(f"trust_remote_code: {args.trust_remote_code}")
    print("=====================================\n")

    data_path = Path(args.data_file)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print(f"Loading data from: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"Total rows in file: {len(df)}")

    # Ensure we have a product_text column
    if "product_text" not in df.columns:
        # Build a reasonable product_text from ProductName, BrandName, Level4_ProductGroup / ProductType
        print("Column 'product_text' not found. Building it from other columns...")
        def build_product_text(row):
            parts = []
            if "ProductName" in row and pd.notna(row["ProductName"]):
                parts.append(str(row["ProductName"]))
            if "BrandName" in row and pd.notna(row["BrandName"]):
                parts.append(str(row["BrandName"]))
            if "Level4_ProductGroup" in row and pd.notna(row["Level4_ProductGroup"]):
                parts.append(str(row["Level4_ProductGroup"]))
            elif "ProductType" in row and pd.notna(row["ProductType"]):
                parts.append(str(row["ProductType"]))
            # Fallback to something
            if not parts:
                return str(row.get("ProductName", "")) or "UNKNOWN_PRODUCT"
            return " ".join(parts)

        df["product_text"] = df.apply(build_product_text, axis=1)

    # Optionally limit number of products for speed
    if args.max_products is not None and len(df) > args.max_products:
        print(f"Sampling {args.max_products} products for the demo (out of {len(df)}).")
        df = df.sample(n=args.max_products, random_state=42).reset_index(drop=True)

    print(f"Using {len(df)} products in this interactive demo.\n")

    # For printing, keep some extra columns if they exist
    has_product_name = "ProductName" in df.columns
    has_brand_name = "BrandName" in df.columns

    # Build prefixed product text (E5/BGE-style)
    df["product_text_prefixed"] = "passage: " + df["product_text"].astype(str)

    # Device
    if args.device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Loading model on device: {device}")
    model = SentenceTransformer(
        args.model_name_or_path,
        device=device,
        trust_remote_code=args.trust_remote_code,
    )

    # Precompute embeddings for all products
    print("\nEncoding all products (this is done once)...")
    product_texts = df["product_text_prefixed"].tolist()
    product_embs = []

    for i in tqdm(range(0, len(product_texts), 1024), desc="Encoding products"):
        batch = product_texts[i : i + 1024]
        emb = encode_texts(model, batch)
        product_embs.append(emb)

    product_embs = np.vstack(product_embs)  # shape: (N, d)
    print(f"Product embeddings shape: {product_embs.shape}\n")

    # Interactive loop
    print("You can now type queries and see the most similar products.")
    print("Type 'exit' or just press Enter on empty line to quit.\n")

    while True:
        try:
            query = input("Enter a search query: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if query == "" or query.lower() == "exit":
            print("Goodbye.")
            break

        # Build query text with prefix
        query_text = "query: " + query

        # Encode query
        q_emb = encode_texts(model, [query_text])[0]  # shape: (d,)

        # Cosine similarity because embeddings are normalized -> dot product
        scores = product_embs @ q_emb  # shape: (N,)

        # Get top-K indices
        k = min(args.top_k, len(df))
        top_idx = np.argsort(-scores)[:k]

        print("\nTop matches:\n")
        for rank, idx in enumerate(top_idx, start=1):
            row = df.iloc[idx]
            score = float(scores[idx])
            # Build a nice display string
            if has_product_name:
                name = str(row["ProductName"])
            else:
                # fallback to product_text
                name = str(row["product_text"])

            brand = str(row["BrandName"]) if has_brand_name and pd.notna(row["BrandName"]) else ""
            text_snippet = row["product_text"]

            print(f"#{rank:2d}  (score = {score:.3f})")
            if brand:
                print(f"    Name:   {name}  [{brand}]")
            else:
                print(f"    Name:   {name}")
            print(f"    Text:   {text_snippet}")
            print("")

        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
