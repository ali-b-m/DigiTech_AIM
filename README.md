
# 📦 ETHAIM — Product Search Relevance Benchmark & Bi-Encoder Fine-Tuning Pipeline  
*(Updated: now includes Interactive Search + LLM-as-Judge Evaluation)*

This repository implements a **full end-to-end pipeline** for benchmarking, training, evaluating, and comparing embedding-based models for **e-commerce product search** on the ETHAIM dataset.

It contains tooling for:
- Data preparation  
- Baseline model evaluation (HF + API models)  
- Fine-tuning bi-encoders (E5, BGE, GTE, etc.)  
- Ranking evaluation (Hit@K, MRR, NDCG)  
- Model speed & architecture analysis  
- **Interactive search demo**  
- **LLM-as-Judge model-vs-model relevance comparison**  

---

# 📁 Project Structure (Updated)

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
│   └── <exp_name>/                 # fine-tuned models
│
├── results/
│   └── <exp_name>/                 # evaluation output
│
├── interactive_outputs/
│   └── <model_name>/
│         <query>.txt               # search results stored here
│
├── model_analysis/
│   └── model_speed_and_architecture.csv
│
└── src/
    ├── split_dataset.py
    ├── train_biencoder.py
    ├── eval_biencoder.py
    ├── eval_hf_embeddings.py
    ├── eval_api_embeddings.py
    ├── analyse_models.py
    ├── interactive_search.py       # NEW
    ├── eval_with_llm_judge.py      # NEW
    ├── baseline_embeddings.py
    ├── explore_embeddings.py
    └── check_gpu.py
```

---

# ⚙️ Installation

```
python -m venv env
env/Scripts/activate
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

---

# 🧪 2. Baseline Evaluation (HuggingFace Models)

```
python src/eval_hf_embeddings.py ^
  --model_name intfloat/multilingual-e5-base ^
  --exp_name e5_base_baseline
```

---

# ☁️ 3. API Model Evaluation (OpenAI / Google)

OpenAI:

```
python src/eval_api_embeddings.py ^
  --provider openai ^
  --model text-embedding-3-large ^
  --exp_name openai_te3_large
```

Google:

```
python src/eval_api_embeddings.py ^
  --provider google ^
  --model models/embedding-001 ^
  --exp_name google_embed01
```

---

# 🎓 4. Fine-Tuning Bi-Encoders

```
python src/train_biencoder.py ^
  --model_name intfloat/multilingual-e5-base ^
  --exp_name e5_finetuned ^
  --epochs 1 ^
  --batch_size 256
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

# 📊 6. Model Analysis (Speed + Architecture)

```
python src/analyse_models.py
```

Outputs:

```
model_analysis/model_speed_and_architecture.csv
```

---

# 🔍 7. Interactive Real-Time Search Demo (NEW)

```
python src/interactive_search.py ^
  --model_name_or_path models/e5_finetuned ^
  --data_file data/unique_products.parquet ^
  --top_k 100
```

Outputs saved to:

```
interactive_outputs/<model_name>/<query>.txt
```

---

# 🤖 8. LLM-as-Judge — Model-vs-Model Comparison (NEW)

```
python src/eval_with_llm_judge.py ^
  --file_a "interactive_outputs/e5_finetuned/blackdress.txt" ^
  --file_b "interactive_outputs/bge-base-en-v1.5/blackdress.txt" ^
  --label_a e5 ^
  --label_b bge ^
  --query "black dress"
```

Produces:

```
results/llm_judgements/judge_blackdress.json
```

---

# ✔️ Summary

This project now includes:

- Dataset preparation  
- Baseline evaluations  
- Fine-tuning pipeline  
- Detailed evaluation  
- Interactive search  
- LLM-as-Judge comparison  
- Speed + architecture analysis  

Perfect for research & production experimentation.
