# 📦 ETHAIM — Product Search Relevance Benchmark & Bi-Encoder Fine-Tuning Pipeline

This repository implements a **full end-to-end pipeline** for benchmarking, training, and comparing embedding-based models for **product search relevance** on the ETHAIM dataset.

It supports:
- Data preparation & splitting  
- Baseline evaluation (HuggingFace + API models)  
- Fine-tuning bi-encoders with MultipleNegativesRankingLoss  
- Full offline ranking evaluation  
- Model comparison by accuracy, speed, and architecture  
- Real-time interactive search demo  

---

# 📁 Project Structure

```
project/
│
├── data/
│   ├── train.parquet
│   ├── val.parquet
│   ├── test.parquet
│   └── raw_embedding_file.parquet
│
├── models/
│   └── <exp_name>/                  # saved fine-tuned models
│
├── results/
│   └── <exp_name>/                  # metrics, plots, examples
│
├── model_analysis/
│   └── model_speed_and_architecture.csv
│
├── src/
│   ├── split_dataset.py
│   ├── train_biencoder.py
│   ├── eval_biencoder.py
│   ├── eval_api_embeddings.py
│   ├── eval_hf_embeddings.py
│   ├── analyse_models.py
│   ├── interactive_search.py
│   ├── baseline_embeddings.py
│   ├── explore_embeddings.py
│   └── check_gpu.py
│
└── README.md
```

---

# ⚙️ Installation

```
python -m venv env
env/Scripts/activate        # Windows
pip install -r requirements.txt
```

Check GPU:
```
python src/check_gpu.py
```

---

# 🧹 1. Data Preparation

Split data into train/val/test:

```
python src/split_dataset.py
```

Ensures:
- Query-level split to avoid leakage  
- Balanced languages  
- Balanced frequencies  

Output stored in `data/`.

---

# 🧪 2. Baseline Evaluation (HuggingFace Models)

Evaluate any HF embedding model:

```
python src/eval_hf_embeddings.py ^
  --model_name sentence-transformers/all-MiniLM-L6-v2 ^
  --exp_name minilm_baseline
```

Output in:

```
results/minilm_baseline/
```

Includes:
- metrics.json  
- metrics.csv  
- examples.json  
- per_query_metrics.csv  
- metrics_overview.png  

---

# ☁️ 3. API Model Evaluation (OpenAI / Google)

OpenAI:

```
python src/eval_api_embeddings.py ^
  --provider openai ^
  --model text-embedding-3-large ^
  --exp_name openai_large
```

Set key:

```
setx OPENAI_API_KEY "xxxx"
```

Google:

```
python src/eval_api_embeddings.py ^
  --provider google ^
  --model models/embedding-001 ^
  --exp_name google_emb01
```

---

# 🎓 4. Fine-Tuning Bi-Encoders

Train a HuggingFace model:

```
python src/train_biencoder.py ^
  --model_name intfloat/multilingual-e5-base ^
  --exp_name e5_finetuned ^
  --epochs 1 ^
  --batch_size 256
```

For models needing remote code:

```
python src/train_biencoder.py ^
  --model_name Alibaba-NLP/gte-multilingual-base ^
  --exp_name gte_finetuned ^
  --trust_remote_code ^
  --epochs 1
```

Models saved to:

```
models/<exp_name>/
```

---

# 🧾 5. Evaluate Fine-Tuned Models

```
python src/eval_biencoder.py ^
  --model_name_or_path models/e5_finetuned ^
  --exp_name e5_finetuned_eval
```

---

# 📊 6. Model Analysis (Speed + Architecture + Metrics)

Runs once after all evaluations:

```
python src/analyse_models.py
```

Produces:

```
model_analysis/model_speed_and_architecture.csv
```

Contains:
- hit@k  
- mrr  
- ndcg  
- parameters  
- embedding size  
- layers  
- throughput (queries/sec)  

---

# 🔍 7. Real-Time Interactive Search Demo

Use any HF or fine-tuned model:

```
python src/interactive_search.py ^
  --model_name_or_path models/e5_finetuned ^
  --data_file data/test.parquet ^
  --top_k 5
```

Example:

```
Enter a search query: schlafsack
→ shows top-5 products
```

---

# 🧠 Metrics Explained

| Metric | Meaning |
|--------|---------|
| **Hit@1** | correct item is rank 1 |
| **Hit@5** | correct item is in top 5 |
| **MRR@10** | earlier relevant = higher score |
| **NDCG@10** | full-quality ranking metric |

---

# 🛠 Add New Model (Example)

Train BGE:

```
python src/train_biencoder.py ^
  --model_name BAAI/bge-base-en-v1.5 ^
  --exp_name bge_finetuned
```

Evaluate:

```
python src/eval_biencoder.py ^
  --model_name_or_path models/bge_finetuned ^
  --exp_name bge_finetuned_eval
```

---

# ✔️ Summary

This project provides:

- Complete dataset preparation  
- Baseline model benchmarks  
- Fine-tuning pipeline  
- Evaluation tooling  
- Model comparison  
- Real-time product search demo  

Perfect for production experiments or research.

