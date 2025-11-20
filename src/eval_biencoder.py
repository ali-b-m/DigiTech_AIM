import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a bi-encoder model on query-product relevance."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Model name (HF) or local path (e.g. models/e5_finetuned_v1).",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help=(
            "Experiment name for saving results. "
            "If not set, derived from last part of model path/name."
        ),
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=str(DATA_DIR / "test.parquet"),
        help="Path to test parquet file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on: 'cuda' or 'cpu'. Default: auto-detect.",
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=None,
        help="Optional limit on number of queries to evaluate (for quick debug).",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help=(
            "If set, pass trust_remote_code=True to SentenceTransformer. "
            "Needed for some models like Alibaba-NLP/gte-multilingual-base."
        ),
    )
    args = parser.parse_args()

    if args.exp_name is None:
        base = args.model_name_or_path.rstrip("/").split("/")[-1]
        args.exp_name = base

    return args


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
    """Compute NDCG@k."""
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
    args = parse_args()

    print("===== EVALUATION CONFIG =====")
    print(f"Model:      {args.model_name_or_path}")
    print(f"Experiment: {args.exp_name}")
    print(f"Test file:  {args.test_file}")
    print(f"Max queries:{args.max_queries}")
    print(f"trust_remote_code: {args.trust_remote_code}")
    print("=============================\n")

    results_dir = RESULTS_DIR / args.exp_name
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Loading test data from:", args.test_file)
    df = pd.read_parquet(args.test_file)
    print("Total test rows:", len(df))

    # Dataset summary
    unique_queries = df["OriginalQuery"].nunique()
    positives_per_query = df.groupby("OriginalQuery")["relevant"].sum()
    queries_with_positive = (positives_per_query > 0).sum()
    products_per_query = df.groupby("OriginalQuery").size()
    lang_stats = df["PageLanguage"].value_counts(normalize=True)

    print("\n========== DATASET SUMMARY ==========")
    print("Unique queries:", unique_queries)
    print("Queries with ≥1 positive:", queries_with_positive)
    print("Avg products per query:", products_per_query.mean())
    print("Languages distribution:")
    print(lang_stats)
    print("=====================================\n")

    # Build prefixed texts
    df["query_text"] = "query: " + df["OriginalQuery"].astype(str)
    df["product_text_prefixed"] = "passage: " + df["product_text"].astype(str)

    # Device
    if args.device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("Loading model on device:", device)
    model = SentenceTransformer(
        args.model_name_or_path,
        device=device,
        trust_remote_code=args.trust_remote_code,
    )

    grouped = df.groupby("OriginalQuery")
    group_items = list(grouped)
    if args.max_queries is not None:
        group_items = group_items[: args.max_queries]
    total_groups = len(group_items)

    # Aggregate metrics
    hit1_sum = 0
    hit5_sum = 0
    hit10_sum = 0
    mrr10_sum = 0
    ndcg10_sum = 0
    total_used = 0

    # Per-query metrics for CSV
    per_query_records = []
    example_queries = []

    print("\nEvaluating model with progress bar...\n")

    for q, group in tqdm(group_items, total=total_groups, desc="Processing queries"):
        if group["relevant"].sum() == 0:
            # Skip queries with no positive labels
            continue

        total_used += 1
        if len(example_queries) < 3:
            example_queries.append((q, group.copy()))

        query_text = group["query_text"].iloc[0]
        product_texts = group["product_text_prefixed"].tolist()
        labels = group["relevant"].to_numpy()

        # Encode
        query_emb = encode_texts(model, [query_text])[0]
        product_embs = encode_texts(model, product_texts)
        scores = product_embs @ query_emb

        ranked_idx = np.argsort(-scores)
        ranked_labels = labels[ranked_idx]

        # Hit@K
        h1 = int(ranked_labels[:1].max() == 1)
        h5 = int(ranked_labels[:5].max() == 1)
        h10 = int(ranked_labels[:10].max() == 1)

        hit1_sum += h1
        hit5_sum += h5
        hit10_sum += h10

        # MRR@10
        pos_idx = np.where(ranked_labels == 1)[0]
        mrr_val = 0.0
        if len(pos_idx) > 0:
            rank = pos_idx[0] + 1
            if rank <= 10:
                mrr_val = 1.0 / rank
                mrr10_sum += mrr_val

        # NDCG@10
        ndcg_val = ndcg_at_k(ranked_labels, k=10)
        ndcg10_sum += ndcg_val

        per_query_records.append(
            {
                "query": q,
                "hit@1": h1,
                "hit@5": h5,
                "hit@10": h10,
                "mrr@10": mrr_val,
                "ndcg@10": ndcg_val,
                "num_products": len(product_texts),
                "num_positives": int(labels.sum()),
            }
        )

    # Normalize metrics
    hit1 = hit1_sum / total_used
    hit5 = hit5_sum / total_used
    hit10 = hit10_sum / total_used
    mrr10 = mrr10_sum / total_used
    ndcg10 = ndcg10_sum / total_used

    print("\n========== EVALUATION RESULTS ==========")
    print("Model:", args.model_name_or_path)
    print("Experiment:", args.exp_name)
    print("Queries evaluated:", total_used)
    print(f"Hit@1:   {hit1:.4f}")
    print(f"Hit@5:   {hit5:.4f}")
    print(f"Hit@10:  {hit10:.4f}")
    print(f"MRR@10:  {mrr10:.4f}")
    print(f"NDCG@10: {ndcg10:.4f}")
    print("========================================\n")

    # Save metrics JSON
    metrics = {
        "model_name_or_path": args.model_name_or_path,
        "exp_name": args.exp_name,
        "test_file": args.test_file,
        "timestamp": datetime.now().isoformat(),
        "num_queries_total": int(unique_queries),
        "num_queries_with_positive": int(queries_with_positive),
        "avg_products_per_query": float(products_per_query.mean()),
        "language_distribution": lang_stats.to_dict(),
        "queries_evaluated": int(total_used),
        "hit@1": float(hit1),
        "hit@5": float(hit5),
        "hit@10": float(hit10),
        "mrr@10": float(mrr10),
        "ndcg@10": float(ndcg10),
    }
    with open(results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save metrics CSV
    metrics_df = pd.DataFrame(
        [
            {"metric": "hit@1", "value": hit1},
            {"metric": "hit@5", "value": hit5},
            {"metric": "hit@10", "value": hit10},
            {"metric": "mrr@10", "value": mrr10},
            {"metric": "ndcg@10", "value": ndcg10},
        ]
    )
    metrics_df.to_csv(results_dir / "metrics.csv", index=False)

    # Save per-query metrics
    per_query_df = pd.DataFrame(per_query_records)
    per_query_df.to_csv(results_dir / "per_query_metrics.csv", index=False)

    # Save example predictions
    example_output = []
    for (query, group) in example_queries:
        record = {"query": query, "products": []}
        query_text = "query: " + query
        product_texts = group["product_text_prefixed"].tolist()
        labels = group["relevant"].to_numpy()
        query_emb = encode_texts(model, [query_text])[0]
        product_embs = encode_texts(model, product_texts)
        scores = product_embs @ query_emb
        ranked_idx = np.argsort(-scores)

        for rank in range(min(10, len(product_texts))):
            idx = ranked_idx[rank]
            record["products"].append(
                {
                    "rank": int(rank + 1),
                    "product_text": product_texts[idx],
                    "score": float(scores[idx]),
                    "relevant": int(labels[idx]),
                }
            )
        example_output.append(record)

    with open(results_dir / "examples.json", "w", encoding="utf-8") as f:
        json.dump(example_output, f, indent=2)

    # Plot simple bar chart of metrics
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        names = ["hit@1", "hit@5", "hit@10", "mrr@10", "ndcg@10"]
        values = [hit1, hit5, hit10, mrr10, ndcg10]
        ax.bar(names, values)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"Metrics Overview - {args.exp_name}")
        ax.set_ylabel("Score")
        plt.tight_layout()
        fig.savefig(results_dir / "metrics_overview.png")
        plt.close(fig)

    except ImportError:
        print(
            "matplotlib not installed, skipping plots. "
            "Install with: pip install matplotlib"
        )

    print("Results and artifacts saved to:", results_dir)


if __name__ == "__main__":
    main()
