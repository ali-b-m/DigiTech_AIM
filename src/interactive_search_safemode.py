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
            "HF model id or local fine-tuned model path.\n"
            "Examples:\n"
            "  intfloat/multilingual-e5-base\n"
            "  BAAI/bge-base-en-v1.5\n"
            "  models/e5_finetuned\n"
            "  models/gte_multi_base_finetuned"
        ),
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=str(DATA_DIR / "test.parquet"),
        help="Parquet file with products.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="How many top products to save for each query.",
    )
    parser.add_argument(
        "--max_products",
        type=int,
        default=100_000,
        help="Optional limit on number of products for speed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: 'cuda' or 'cpu'. Default: auto.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Needed for some models like Alibaba-NLP/gte-multilingual-base.",
    )
    parser.add_argument(
        "--block_text_keywords",
        type=str,
        default="",
        help=(
            "Comma-separated keywords. If any is contained in product_text "
            "the product will be excluded from results. Example: "
            "--block_text_keywords vibrator,dildo,erotik"
        ),
    )
    parser.add_argument(
        "--block_type_keywords",
        type=str,
        default="",
        help=(
            "Comma-separated keywords. If any is contained in ProductType or "
            "Level4_ProductGroup, the product will be excluded."
        ),
    )
    return parser.parse_args()


def encode_texts(model, texts: List[str]) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=128,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def main():
    args = parse_args()

    data_path = Path(args.data_file)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_parquet(data_path)

    # Ensure product_text exists
    if "product_text" not in df.columns:
        def build_text(row):
            parts = []
            if "ProductName" in row and pd.notna(row["ProductName"]):
                parts.append(str(row["ProductName"]))
            if "BrandName" in row and pd.notna(row["BrandName"]):
                parts.append(str(row["BrandName"]))
            if "ProductType" in row and pd.notna(row["ProductType"]):
                parts.append(str(row["ProductType"]))
            if "Level4_ProductGroup" in row and pd.notna(row["Level4_ProductGroup"]):
                parts.append(str(row["Level4_ProductGroup"]))
            return " ".join(parts) if parts else "UNKNOWN_PRODUCT"

        df["product_text"] = df.apply(build_text, axis=1)

    # Maybe subsample for speed
    if args.max_products is not None and len(df) > args.max_products:
        df = df.sample(n=args.max_products, random_state=42).reset_index(drop=True)

    # Prefixed text for the encoder
    df["product_text_prefixed"] = "passage: " + df["product_text"].astype(str)

    # Prepare blocking lists
    block_text_kw = [k.strip().lower() for k in args.block_text_keywords.split(",") if k.strip()]
    block_type_kw = [k.strip().lower() for k in args.block_type_keywords.split(",") if k.strip()]

    # Add lowercase helper columns
    df["product_text_lower"] = df["product_text"].str.lower()
    if "ProductType" in df.columns:
        df["ProductType_lower"] = df["ProductType"].fillna("").str.lower()
    else:
        df["ProductType_lower"] = ""
    if "Level4_ProductGroup" in df.columns:
        df["Level4_lower"] = df["Level4_ProductGroup"].fillna("").str.lower()
    else:
        df["Level4_lower"] = ""

    # Device
    if args.device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    model = SentenceTransformer(
        args.model_name_or_path,
        device=device,
        trust_remote_code=args.trust_remote_code,
    )

    # Precompute embeddings
    product_texts = df["product_text_prefixed"].tolist()
    product_embs = []
    for i in tqdm(range(0, len(product_texts), 1024), desc="Encoding products"):
        emb = encode_texts(model, product_texts[i:i + 1024])
        product_embs.append(emb)
    product_embs = np.vstack(product_embs)

    # Output directory
    model_clean = args.model_name_or_path.replace("\\", "/").split("/")[-1]
    save_dir = ROOT / "interactive_outputs" / model_clean
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\nInteractive search ready.")
    print("Type a query, or 'exit' to quit.\n")

    while True:
        try:
            query = input("Query: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if query == "" or query.lower() == "exit":
            print("Goodbye.")
            break

        # Encode query
        q_emb = encode_texts(model, ["query: " + query])[0]
        scores = product_embs @ q_emb

        # Initial ranking
        ranked_idx = np.argsort(-scores)

        # Apply blocking rules
        allowed_idx = []
        for idx in ranked_idx:
            row = df.iloc[idx]

            blocked = False

            # Block by text keywords
            if block_text_kw:
                text = row["product_text_lower"]
                if any(kw in text for kw in block_text_kw):
                    blocked = True

            # Block by type/group keywords
            if not blocked and block_type_kw:
                t = row["ProductType_lower"]
                g = row["Level4_lower"]
                if any(kw in t or kw in g for kw in block_type_kw):
                    blocked = True

            if not blocked:
                allowed_idx.append(idx)
            if len(allowed_idx) >= args.top_k:
                break

        # If everything got blocked, fall back to top_k without filtering
        if not allowed_idx:
            allowed_idx = ranked_idx[:args.top_k]

        # Save to file
        query_clean = "".join(c for c in query if c.isalnum() or c in "_-")
        if not query_clean:
            query_clean = "query"
        save_path = save_dir / f"{query_clean}.txt"

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(f"Query: {query}\n")
            f.write(f"Model: {args.model_name_or_path}\n")
            f.write(f"Top {len(allowed_idx)} results\n")
            f.write(f"Blocked text keywords: {block_text_kw}\n")
            f.write(f"Blocked type keywords: {block_type_kw}\n\n")

            for rank, idx in enumerate(allowed_idx, start=1):
                row = df.iloc[idx]
                score = float(scores[idx])
                name = row.get("ProductName", row["product_text"])
                brand = row.get("BrandName", "")
                ptype = row.get("ProductType", "")
                lvl4 = row.get("Level4_ProductGroup", "")
                text = row["product_text"]

                f.write(f"#{rank} (score={score:.4f})\n")
                if brand:
                    f.write(f"  Name: {name} [{brand}]\n")
                else:
                    f.write(f"  Name: {name}\n")
                if ptype or lvl4:
                    f.write(f"  Type: {ptype} | Group: {lvl4}\n")
                f.write(f"  Text: {text}\n\n")

        print(f"Results saved to: {save_path}\n")


if __name__ == "__main__":
    main()
