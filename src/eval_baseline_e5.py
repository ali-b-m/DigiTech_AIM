import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List
import random


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TEST_FILE = DATA_DIR / "test.parquet"


def encode_texts(model, texts: List[str]) -> np.ndarray:
    """Encode text with L2-normalized embeddings."""
    return model.encode(
        texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def ndcg_at_k(relevances: np.ndarray, k: int = 10) -> float:
    """Compute NDCG@k for a ranking."""
    relevances = relevances[:k]
    if relevances.sum() == 0:
        return 0.0
    dcg = 0.0
    for i, rel in enumerate(relevances):
        dcg += rel / np.log2(i + 2)
    ideal = np.sort(relevances)[::-1]
    idcg = 0.0
    for i, rel in enumerate(ideal):
        idcg += rel / np.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0


def main():
    print("Loading test data from:", TEST_FILE)
    df = pd.read_parquet(TEST_FILE)
    print("Total test rows:", len(df))

    # Dataset summary
    print("\n========== DATASET SUMMARY ==========")
    unique_queries = df["OriginalQuery"].nunique()
    print("Unique queries:", unique_queries)

    positives_per_query = df.groupby("OriginalQuery")["relevant"].sum()
    queries_with_positive = (positives_per_query > 0).sum()
    print("Queries with ≥1 positive:", queries_with_positive)

    products_per_query = df.groupby("OriginalQuery").size()
    print("Avg products per query:", products_per_query.mean())

    lang_stats = df["PageLanguage"].value_counts(normalize=True)
    print("Languages distribution:")
    print(lang_stats)
    print("=====================================\n")

    # Build prefix texts
    df["query_text"] = "query: " + df["OriginalQuery"].astype(str)
    df["product_text_prefixed"] = "passage: " + df["product_text"].astype(str)

    # Load model
    model_name = "intfloat/multilingual-e5-base"
    print("Loading model:", model_name)
    model = SentenceTransformer(model_name)

    grouped = df.groupby("OriginalQuery")

    # Metrics
    hit1 = 0
    hit5 = 0
    hit10 = 0
    mrr10 = 0
    ndcg10 = 0

    total_used = 0
    example_queries = []

    print("\nEvaluating... (this may take some minutes)")

    for q, group in grouped:
        if group["relevant"].sum() == 0:
            continue

        total_used += 1
        if len(example_queries) < 3:
            example_queries.append((q, group.copy()))

        query_text = group["query_text"].iloc[0]
        product_texts = group["product_text_prefixed"].tolist()
        labels = group["relevant"].to_numpy()

        query_emb = encode_texts(model, [query_text])[0]
        product_embs = encode_texts(model, product_texts)
        scores = product_embs @ query_emb

        ranked_idx = np.argsort(-scores)
        ranked_labels = labels[ranked_idx]

        # Hit@K
        hit1 += int(ranked_labels[:1].max() == 1)
        hit5 += int(ranked_labels[:5].max() == 1)
        hit10 += int(ranked_labels[:10].max() == 1)

        # MRR@10
        pos_idx = np.where(ranked_labels == 1)[0]
        if len(pos_idx) > 0:
            rank = pos_idx[0] + 1
            if rank <= 10:
                mrr10 += 1.0 / rank

        # NDCG@10
        ndcg10 += ndcg_at_k(ranked_labels, k=10)

    # Normalize metrics
    hit1 /= total_used
    hit5 /= total_used
    hit10 /= total_used
    mrr10 /= total_used
    ndcg10 /= total_used

    print("\n========== BASELINE RESULTS ==========")
    print("Model:", model_name)
    print("Queries evaluated:", total_used)
    print(f"Hit@1:   {hit1:.4f}")
    print(f"Hit@5:   {hit5:.4f}")
    print(f"Hit@10:  {hit10:.4f}")
    print(f"MRR@10:  {mrr10:.4f}")
    print(f"NDCG@10: {ndcg10:.4f}")
    print("======================================\n")

    # Show example predictions
    print("========== EXAMPLE QUERIES ==========")

    for (query, group) in example_queries:
        print(f"\nQuery: {query}")
        query_text = "query: " + query
        product_texts = group["product_text_prefixed"].tolist()
        labels = group["relevant"].to_numpy()

        query_emb = encode_texts(model, [query_text])[0]
        product_embs = encode_texts(model, product_texts)
        scores = product_embs @ query_emb
        ranked_idx = np.argsort(-scores)

        for rank in range(min(5, len(product_texts))):
            idx = ranked_idx[rank]
            print(
                f"  #{rank + 1}: {product_texts[idx][:60]:60} | "
                f"score={scores[idx]:.3f} | relevant={labels[idx]}"
            )

    print("\n======================================\n")


if __name__ == "__main__":
    main()
