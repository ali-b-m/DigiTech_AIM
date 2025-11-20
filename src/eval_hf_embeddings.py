# eval_hf_embeddings.py
# Evaluation for HuggingFace / SentenceTransformer models (local)

import os
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer


# ============================
# METRIC FUNCTIONS
# ============================
def hit_at_k(ranks, k):
    return float(np.mean([1.0 if r < k else 0.0 for r in ranks]))


def mrr_at_k(ranks, k):
    vals = []
    for r in ranks:
        if r < k:
            vals.append(1.0 / (r + 1))
        else:
            vals.append(0.0)
    return float(np.mean(vals))


def ndcg_at_k(all_relevances, k):
    ndcgs = []
    for rel in all_relevances:
        rel_k = rel[:k]
        # DCG
        dcg = 0.0
        for i, r in enumerate(rel_k):
            dcg += (2**r - 1) / np.log2(i + 2)
        # IDCG
        ideal = sorted(rel_k, reverse=True)
        idcg = 0.0
        for i, r in enumerate(ideal):
            idcg += (2**r - 1) / np.log2(i + 2)
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(ndcgs))


# ============================
# PLOTTING
# ============================
def make_plots(
    out_dir,
    ranks,
    all_relevances,
    pos_sims,
    neg_sims,
    products_per_query,
    metrics,
):
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Hit@K, MRR@K, NDCG@K bar plots
    ks = [1, 5, 10]
    hit_values = [metrics["hit@1"], metrics["hit@5"], metrics["hit@10"]]
    mrr_values = [metrics["mrr@1"], metrics["mrr@5"], metrics["mrr@10"]]
    ndcg_values = [metrics["ndcg@1"], metrics["ndcg@5"], metrics["ndcg@10"]]

    # Hit@K
    plt.figure()
    plt.bar([str(k) for k in ks], hit_values)
    plt.xlabel("K")
    plt.ylabel("Hit@K")
    plt.title("Hit@K")
    plt.savefig(plots_dir / "hit_at_k.png", bbox_inches="tight")
    plt.close()

    # MRR@K
    plt.figure()
    plt.bar([str(k) for k in ks], mrr_values)
    plt.xlabel("K")
    plt.ylabel("MRR@K")
    plt.title("MRR@K")
    plt.savefig(plots_dir / "mrr_at_k.png", bbox_inches="tight")
    plt.close()

    # NDCG@K
    plt.figure()
    plt.bar([str(k) for k in ks], ndcg_values)
    plt.xlabel("K")
    plt.ylabel("NDCG@K")
    plt.title("NDCG@K")
    plt.savefig(plots_dir / "ndcg_at_k.png", bbox_inches="tight")
    plt.close()

    # Similarity histogram (pos vs neg)
    if len(pos_sims) > 0 and len(neg_sims) > 0:
        plt.figure()
        plt.hist(pos_sims, bins=50, alpha=0.5, label="Positive pairs")
        plt.hist(neg_sims, bins=50, alpha=0.5, label="Negative pairs")
        plt.xlabel("Similarity")
        plt.ylabel("Count")
        plt.legend()
        plt.title("Similarity distribution (positive vs negative)")
        plt.savefig(plots_dir / "similarity_hist.png", bbox_inches="tight")
        plt.close()

    # Products per query histogram
    if len(products_per_query) > 0:
        plt.figure()
        plt.hist(products_per_query, bins=50)
        plt.xlabel("Products per query")
        plt.ylabel("Count of queries")
        plt.title("Distribution of products per query")
        plt.savefig(plots_dir / "products_per_query_hist.png", bbox_inches="tight")
        plt.close()


# ============================
# MAIN
# ============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="HF / SentenceTransformer model name or local path",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        required=True,
        help="Experiment name (used as folder name under results/).",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="data/test.parquet",
        help="Path to test parquet file.",
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=None,
        help="Max number of queries to evaluate (for speed). None = all.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device, e.g. 'cuda' or 'cpu'. If None, auto-detect.",
    )
    args = parser.parse_args()

    start_time = time.time()

    print("===== HF EVALUATION CONFIG =====")
    print(f"Model:      {args.model_name_or_path}")
    print(f"Experiment: {args.exp_name}")
    print(f"Test file:  {args.test_path}")
    print(f"Max queries:{args.max_queries}")
    print("================================\n")

    # ---------------------------
    # Load data
    # ---------------------------
    df = pd.read_parquet(args.test_path)
    total_rows = len(df)

    # product_text should exist from your split script
    if "product_text" not in df.columns:
        raise ValueError("Column 'product_text' not found in test parquet.")

    grouped = df.groupby("OriginalQuery")

    # queries with at least 1 positive
    queries = []
    for q, g in grouped:
        if g["relevant"].sum() > 0:
            queries.append(q)

    if args.max_queries is not None:
        queries = queries[: args.max_queries]

    num_queries = len(queries)
    num_pos_queries = sum(
        1 for q in queries if grouped.get_group(q)["relevant"].sum() > 0
    )
    avg_products = total_rows / max(num_queries, 1)

    print("========== DATASET SUMMARY ==========")
    print(f"Total rows: {total_rows}")
    print(f"Unique queries (in this run): {num_queries}")
    print(f"Queries with ≥1 positive:     {num_pos_queries}")
    print(f"Avg products per query:       {avg_products}")
    if "PageLanguage" in df.columns:
        lang_dist = df["PageLanguage"].value_counts(normalize=True)
        print("Languages distribution:")
        print(lang_dist)
    print("=====================================\n")

    # ---------------------------
    # Load model
    # ---------------------------
    print("Loading model...")
    model_kwargs = {}
    if args.device is not None:
        model_kwargs["device"] = args.device
    model = SentenceTransformer(args.model_name_or_path, **model_kwargs)

    # ---------------------------
    # Evaluation loop
    # ---------------------------
    ranks = []
    all_relevances = []
    pos_sims = []
    neg_sims = []
    products_per_query = []
    example_outputs = []

    for qi, query in enumerate(tqdm(queries, desc="Processing queries")):
        g = grouped.get_group(query)
        products_per_query.append(len(g))

        # Build texts
        q_text = "query: " + str(query)
        p_texts = ["passage: " + t for t in g["product_text"].tolist()]

        # Encode
        q_emb = model.encode(q_text, convert_to_numpy=True, normalize_embeddings=True)
        p_embs = model.encode(
            p_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # Similarities
        sims = np.dot(p_embs, q_emb)  # since normalized, dot == cosine
        order = np.argsort(-sims)  # descending

        relev = g["relevant"].values
        sorted_relev = relev[order]

        # collect sims for histogram
        for sim_val, rel_val in zip(sims, relev):
            if rel_val == 1:
                pos_sims.append(float(sim_val))
            else:
                neg_sims.append(float(sim_val))

        if 1 not in sorted_relev:
            # Should not really happen but is safe to skip
            continue

        best_rank = int(np.where(sorted_relev == 1)[0][0])
        ranks.append(best_rank)
        all_relevances.append(sorted_relev.tolist())

        # Save example for first few queries
        if qi < 3:
            example_outputs.append(
                {
                    "query": query,
                    "top5": [
                        {
                            "product_text": p_texts[i],
                            "score": float(sims[i]),
                            "relevant": int(relev[i]),
                        }
                        for i in order[:5]
                    ],
                }
            )

    # ---------------------------
    # Metrics
    # ---------------------------
    if len(ranks) == 0:
        raise RuntimeError("No queries with positives were evaluated.")

    hit1 = hit_at_k(ranks, 1)
    hit5 = hit_at_k(ranks, 5)
    hit10 = hit_at_k(ranks, 10)

    mrr1 = mrr_at_k(ranks, 1)
    mrr5 = mrr_at_k(ranks, 5)
    mrr10 = mrr_at_k(ranks, 10)

    ndcg1 = ndcg_at_k(all_relevances, 1)
    ndcg5 = ndcg_at_k(all_relevances, 5)
    ndcg10 = ndcg_at_k(all_relevances, 10)

    end_time = time.time()
    total_time = end_time - start_time

    print("========== HF EVALUATION RESULTS ==========")
    print(f"Model:          {args.model_name_or_path}")
    print(f"Experiment:     {args.exp_name}")
    print(f"Queries evaled: {len(ranks)}")
    print(f"Hit@1:   {hit1:.4f}")
    print(f"Hit@5:   {hit5:.4f}")
    print(f"Hit@10:  {hit10:.4f}")
    print(f"MRR@1:   {mrr1:.4f}")
    print(f"MRR@5:   {mrr5:.4f}")
    print(f"MRR@10:  {mrr10:.4f}")
    print(f"NDCG@1:  {ndcg1:.4f}")
    print(f"NDCG@5:  {ndcg5:.4f}")
    print(f"NDCG@10: {ndcg10:.4f}")
    print(f"Total time (s): {total_time:.1f}")
    print("===========================================\n")

    # ---------------------------
    # Save results
    # ---------------------------
    out_dir = Path("results") / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "model": args.model_name_or_path,
        "exp_name": args.exp_name,
        "num_queries": len(ranks),
        "hit@1": hit1,
        "hit@5": hit5,
        "hit@10": hit10,
        "mrr@1": mrr1,
        "mrr@5": mrr5,
        "mrr@10": mrr10,
        "ndcg@1": ndcg1,
        "ndcg@5": ndcg5,
        "ndcg@10": ndcg10,
        "total_time_sec": total_time,
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(out_dir / "examples.json", "w") as f:
        json.dump(example_outputs, f, indent=2)

    np.save(out_dir / "ranks.npy", np.array(ranks, dtype=np.int32))

    with open(out_dir / "relevances.json", "w") as f:
        json.dump(all_relevances, f)

    # Console summary for later
    console_summary = [
        f"Model: {args.model_name_or_path}",
        f"Experiment: {args.exp_name}",
        f"Queries evaluated: {len(ranks)}",
        f"Hit@1: {hit1:.4f}",
        f"Hit@5: {hit5:.4f}",
        f"Hit@10: {hit10:.4f}",
        f"MRR@1: {mrr1:.4f}",
        f"MRR@5: {mrr5:.4f}",
        f"MRR@10: {mrr10:.4f}",
        f"NDCG@1: {ndcg1:.4f}",
        f"NDCG@5: {ndcg5:.4f}",
        f"NDCG@10: {ndcg10:.4f}",
        f"Total time (sec): {total_time:.1f}",
    ]
    with open(out_dir / "console_output.txt", "w") as f:
        f.write("\n".join(console_summary))

    # Plots
    make_plots(
        out_dir,
        ranks,
        all_relevances,
        pos_sims,
        neg_sims,
        products_per_query,
        metrics,
    )

    print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
