# src/analyse_models.py
# Analyse speed + architecture of models that already have results.
# This DOES NOT re-run any evaluations.

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import csv

try:
    from transformers import AutoConfig, AutoModel
except ImportError:
    AutoConfig = None
    AutoModel = None


# ------------------------------
# 1. Configure your experiments
# ------------------------------

@dataclass
class Experiment:
    exp_name: str          # folder name inside results/
    kind: str              # "hf" or "api"
    model_id: str          # HF model id or API model name
    provider: Optional[str] = None  # "openai"/"google" for API, else None
    note: str = ""         # description e.g. "baseline", "finetuned"


EXPERIMENTS: List[Experiment] = [
    # Hugging Face / local models
    Experiment(
        exp_name="e5_base_baseline",
        kind="hf",
        model_id="intfloat/multilingual-e5-base",
        note="HF baseline",
    ),
    Experiment(
        exp_name="e5_finetuned_v1",
        kind="hf",
        # architecture is still the same as the base model
        model_id="intfloat/multilingual-e5-base",
        note="HF finetuned on ETHAIM data",
    ),
    Experiment(
        exp_name="bge_base_en_v15_baseline",
        kind="hf",
        model_id="BAAI/bge-base-en-v1.5",
        note="HF baseline",
    ),
    Experiment(
        exp_name="gte_multi_baseline",
        kind="hf",
        model_id="Alibaba-NLP/gte-multilingual-base",
        note="HF baseline",
    ),

    # API models (OpenAI)
    Experiment(
        exp_name="openai_te3_large_baseline",
        kind="api",
        model_id="text-embedding-3-large",
        provider="openai",
        note="OpenAI API",
    ),
    Experiment(
        exp_name="openai_te3_small",
        kind="api",
        model_id="text-embedding-3-small",
        provider="openai",
        note="OpenAI API",
    ),
]

RESULTS_DIR = Path("results")
ANALYSIS_DIR = Path("model_analysis")
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------
# 2. Helpers
# ------------------------------

def load_metrics(exp: Experiment) -> dict:
    metrics_path = RESULTS_DIR / exp.exp_name / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found for {exp.exp_name} at {metrics_path}")
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


_param_cache = {}


def get_arch_info(model_id: str):
    """
    Return (params_millions, embedding_dim, num_layers) for a HF model.
    If transformers is not available, or something fails, returns (None, None, None).
    """
    if AutoConfig is None or AutoModel is None:
        return None, None, None

    if model_id in _param_cache:
        return _param_cache[model_id]

    try:
        # some models (gte) need trust_remote_code=True
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True)

        total_params = sum(p.numel() for p in model.parameters())
        params_m = total_params / 1e6

        # try to guess embedding dimension and number of layers
        hidden_size = getattr(cfg, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(cfg, "d_model", None)

        num_layers = getattr(cfg, "num_hidden_layers", None)
        if num_layers is None:
            num_layers = getattr(cfg, "encoder_layers", None)

        del model

        info = (params_m, hidden_size, num_layers)
        _param_cache[model_id] = info
        return info

    except Exception as e:
        print(f"[WARN] Could not read architecture info for {model_id}: {e}")
        return None, None, None


def safe_get(d: dict, key: str, default=None):
    return d[key] if key in d else default


def fmt(x, width=10, digits=4):
    if x is None:
        return " " * width
    if isinstance(x, float):
        return f"{x:.{digits}f}".rjust(width)
    return str(x).rjust(width)


# ------------------------------
# 3. Main analysis
# ------------------------------

def main():
    rows = []

    for exp in EXPERIMENTS:
        metrics = load_metrics(exp)

        num_queries = safe_get(metrics, "num_queries", None)
        total_time = safe_get(metrics, "total_time_sec", None)

        # --- speed per query ---
        if total_time and total_time > 0 and num_queries:
            qps = num_queries / total_time
            qpm = qps * 60.0
        else:
            qps = None
            qpm = None

        # --- for API models: texts per second ---
        total_texts = safe_get(metrics, "total_texts_embedded", None)
        if total_texts is not None and total_time and total_time > 0:
            tps = total_texts / total_time
            tpm = tps * 60.0
        else:
            tps = None
            tpm = None

        # --- architecture for HF models ---
        if exp.kind == "hf":
            params_m, emb_dim, num_layers = get_arch_info(exp.model_id)
        else:
            params_m = emb_dim = num_layers = None

        row = {
            "exp_name": exp.exp_name,
            "kind": exp.kind,
            "provider": exp.provider if exp.provider else "local",
            "model_id": exp.model_id,
            "note": exp.note,
            "hit@1": safe_get(metrics, "hit@1", None),
            "hit@5": safe_get(metrics, "hit@5", None),
            "hit@10": safe_get(metrics, "hit@10", None),
            "mrr@10": safe_get(metrics, "mrr@10", None),
            "ndcg@10": safe_get(metrics, "ndcg@10", None),
            "num_queries": num_queries,
            "total_time_sec": total_time,
            "queries_per_sec": qps,
            "queries_per_min": qpm,
            "total_texts_embedded": total_texts,
            "texts_per_sec": tps,
            "texts_per_min": tpm,
            "params_millions": params_m,
            "embedding_dim": emb_dim,
            "num_layers": num_layers,
        }
        rows.append(row)

    # --------- print nice table ----------
    header = [
        "exp_name", "kind", "provider", "model_id", "note",
        "hit@1", "hit@5", "hit@10", "mrr@10", "ndcg@10",
        "num_queries", "total_time_sec", "queries_per_sec", "queries_per_min",
        "total_texts_embedded", "texts_per_sec", "texts_per_min",
        "params_millions", "embedding_dim", "num_layers",
    ]

    print("\n================= MODEL ANALYSIS (NO RE-EVAL) =================")
    print(" | ".join(h.ljust(20) for h in header))
    print("-" * 220)

    for r in rows:
        line = " | ".join([
            str(r["exp_name"]).ljust(22),
            str(r["kind"]).ljust(5),
            str(r["provider"]).ljust(8),
            str(r["model_id"]).ljust(30),
            str(r["note"]).ljust(26),
            fmt(r["hit@1"]),
            fmt(r["hit@5"]),
            fmt(r["hit@10"]),
            fmt(r["mrr@10"]),
            fmt(r["ndcg@10"]),
            fmt(r["num_queries"], width=8, digits=0),
            fmt(r["total_time_sec"], width=10, digits=1),
            fmt(r["queries_per_sec"], width=10, digits=3),
            fmt(r["queries_per_min"], width=10, digits=1),
            fmt(r["total_texts_embedded"], width=8, digits=0),
            fmt(r["texts_per_sec"], width=10, digits=2),
            fmt(r["texts_per_min"], width=10, digits=1),
            fmt(r["params_millions"], width=10, digits=1),
            fmt(r["embedding_dim"], width=10, digits=0),
            fmt(r["num_layers"], width=8, digits=0),
        ])
        print(line)

    # --------- save CSV for slides/report ----------
    csv_path = ANALYSIS_DIR / "model_speed_and_architecture.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\n[INFO] CSV summary written to: {csv_path}")
    print("================================================================\n")


if __name__ == "__main__":
    main()
