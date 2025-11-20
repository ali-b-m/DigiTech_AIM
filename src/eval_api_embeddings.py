# eval_api_embeddings.py
# Evaluation for API-based embedding models (OpenAI, Google)

import os
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from openai import OpenAI
import google.generativeai as genai


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
        dcg = 0.0
        for i, r in enumerate(rel_k):
            dcg += (2**r - 1) / np.log2(i + 2)
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

    # Similarity histogram
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
# API WRAPPERS
# ============================
def openai_embed(text_list, model_name):
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(model=model_name, input=text_list)
    vectors = [d.embedding for d in response.data]
    return np.array(vectors)


def google_embed(text_list, model_name):
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key is None:
        raise RuntimeError("GOOGLE_API_KEY environment variable is not set.")

    genai.configure(api_key=api_key)
    vectors = []
    for txt in text_list:
        r = genai.embed_content(model=model_name, content=txt)
        vectors.append(r["embedding"])
    return np.array(vectors)


# ============================
# MAIN
# ============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider",
        type=str,
        required=True,
        choices=["openai", "google"],
        help="Which API provider to use.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Embedding model name, e.g. 'text-embedding-3-large' or 'models/text-embedding-004'.",
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
        default=200,
        help="Max number of queries to evaluate (limit for cost).",
    )
    args = parser.parse_args()

    start_time = time.time()

    print("===== API EVALUATION =====")
    print(args)
    print("==========================\n")

    # ---------------------------
    # Load data
    # ---------------------------
    df = pd.read_parquet(args.test_path)
    total_rows_all = len(df)

    if "product_text" not in df.columns:
        raise ValueError("Column 'product_text' not found in test parquet.")

    grouped = df.groupby("OriginalQuery")

    # Collect only queries with at least one positive
    all_queries_with_pos = []
    for q, g in grouped:
        if g["relevant"].sum() > 0:
            all_queries_with_pos.append(q)

    # 🔀 Random, reproducible sampling of queries
    rng = np.random.default_rng(seed=42)
    rng.shuffle(all_queries_with_pos)

    if args.max_queries is not None and args.max_queries > 0:
        queries = all_queries_with_pos[: args.max_queries]
    else:
        queries = all_queries_with_pos

    num_queries = len(queries)

    # Restrict df to only those queries for the summary stats
    df_sel = df[df["OriginalQuery"].isin(queries)].copy()
    total_rows_sel = len(df_sel)
    avg_products = total_rows_sel / max(num_queries, 1)

    num_pos_queries = num_queries  # by construction, all have ≥1 positive

    print("========== DATASET SUMMARY ==========")
    print(f"Total test rows (full file): {total_rows_all}")
    print(f"Rows used in this run:       {total_rows_sel}")
    print(f"Unique queries (in this run): {num_queries}")
    print(f"Queries with ≥1 positive:     {num_pos_queries}")
    print(f"Avg products per query:       {avg_products}")
    if "PageLanguage" in df_sel.columns:
        lang_dist = df_sel["PageLanguage"].value_counts(normalize=True)
        print("Languages distribution (selected subset):")
        print(lang_dist)
    print("=====================================\n")

    # ---------------------------
    # Choose embedding function
    # ---------------------------
    if args.provider == "openai":
        embed_fn = lambda texts: openai_embed(texts, args.model_name)
    else:
        embed_fn = lambda texts: google_embed(texts, args.model_name)

    # ---------------------------
    # Evaluation loop
    # ---------------------------
    out_dir = Path("results") / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    ranks = []
    all_relevances = []
    pos_sims = []
    neg_sims = []
    products_per_query = []
    example_outputs = []

    total_api_calls = 0
    total_texts_embedded = 0

    print(f"Queries to evaluate: {len(queries)} \n")
    print("Evaluating...\n")

    for qi, query in enumerate(tqdm(queries, desc="Processing queries")):
        g = grouped.get_group(query)
        products_per_query.append(len(g))

        q_text = "query: " + str(query)
        p_texts = ["passage: " + t for t in g["product_text"].tolist()]

        # 1) Query embedding (one call)
        q_emb = embed_fn([q_text])[0]
        total_api_calls += 1
        total_texts_embedded += 1

        # 2) Product embeddings (one call per query)
        p_embs = embed_fn(p_texts)
        total_api_calls += 1
        total_texts_embedded += len(p_texts)

        # Convert to numpy
        q_emb = np.array(q_emb, dtype=np.float32)
        p_embs = np.array(p_embs, dtype=np.float32)

        # Similarity (dot product; assume embeddings approx normalized)
        sims = np.dot(p_embs, q_emb)
        order = np.argsort(-sims)

        relev = g["relevant"].values
        sorted_relev = relev[order]

        # collect similarity scores
        for sim_val, rel_val in zip(sims, relev):
            if rel_val == 1:
                pos_sims.append(float(sim_val))
            else:
                neg_sims.append(float(sim_val))

        if 1 not in sorted_relev:
            continue

        best_rank = int(np.where(sorted_relev == 1)[0][0])
        ranks.append(best_rank)
        all_relevances.append(sorted_relev.tolist())

        # Save first 3 example queries
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

    print("========== API EVALUATION RESULTS ==========")
    print(f"Provider:       {args.provider}")
    print(f"Model:          {args.model_name}")
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
    print(f"Total API calls:        {total_api_calls}")
    print(f"Total texts embedded:   {total_texts_embedded}")
    print(f"Total time (seconds):   {total_time:.1f}")
    print("============================================\n")

    metrics = {
        "provider": args.provider,
        "model": args.model_name,
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
        "total_api_calls": total_api_calls,
        "total_texts_embedded": total_texts_embedded,
        "total_time_sec": total_time,
    }

    # Save metrics and artifacts
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(out_dir / "examples.json", "w") as f:
        json.dump(example_outputs, f, indent=2)

    np.save(out_dir / "ranks.npy", np.array(ranks, dtype=np.int32))

    with open(out_dir / "relevances.json", "w") as f:
        json.dump(all_relevances, f)

    # Console summary
    console_summary = [
        f"Provider: {args.provider}",
        f"Model: {args.model_name}",
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
        f"Total API calls: {total_api_calls}",
        f"Total texts embedded: {total_texts_embedded}",
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
