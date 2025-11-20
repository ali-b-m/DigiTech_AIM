import math
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a bi-encoder (query/product) with MultipleNegativesRankingLoss."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="intfloat/multilingual-e5-base",
        help="Base model name or path (HuggingFace or local).",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help=(
            "Experiment name used for saving model and results. "
            "If not set, derived from model_name."
        ),
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=str(DATA_DIR / "train.parquet"),
        help="Path to train parquet file.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--max_train_rows",
        type=int,
        default=None,
        help="Optional limit on number of positive rows used for training.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate for optimizer.",
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

    # Derive experiment name if not given
    if args.exp_name is None:
        base = args.model_name.rstrip("/").split("/")[-1]
        args.exp_name = base

    return args


def prepare_training_examples(df: pd.DataFrame, max_rows=None):
    """
    Convert training dataframe into InputExample objects for
    MultipleNegativesRankingLoss (query, positive_product).
    """
    df_pos = df[df["relevant"] == 1].copy()
    print("Positive rows in train:", len(df_pos))

    if max_rows is not None and len(df_pos) > max_rows:
        df_pos = df_pos.sample(n=max_rows, random_state=42)
        print(f"Sampled {len(df_pos)} positive rows for training.")

    # Build E5-style prefixed texts
    df_pos["query_text"] = "query: " + df_pos["OriginalQuery"].astype(str)
    df_pos["product_text"] = "passage: " + df_pos["product_text"].astype(str)

    examples = []
    print("Converting rows to InputExample objects...")
    for _, row in tqdm(df_pos.iterrows(), total=len(df_pos), desc="Building training examples"):
        ex = InputExample(
            texts=[row["query_text"], row["product_text"]],
        )
        examples.append(ex)

    print("Total training examples:", len(examples))
    return examples


def main():
    args = parse_args()

    print("===== TRAINING CONFIG =====")
    print(f"Model name:     {args.model_name}")
    print(f"Experiment:     {args.exp_name}")
    print(f"Train file:     {args.train_file}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Epochs:         {args.epochs}")
    print(f"Max train rows: {args.max_train_rows}")
    print(f"Learning rate:  {args.lr}")
    print(f"trust_remote_code: {args.trust_remote_code}")
    print("===========================\n")

    models_dir = ROOT / "models" / args.exp_name
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Checking CUDA availability...")
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Training will run on CPU (slow).")
        device = "cpu"
    else:
        device = "cuda"
        print("CUDA is available – using GPU.")

    print("\nLoading train data from:", args.train_file)
    df = pd.read_parquet(args.train_file)
    print("Train rows:", len(df))

    positives = (df["relevant"] == 1).sum()
    negatives = (df["relevant"] == 0).sum()
    print(f"Positives: {positives} | Negatives: {negatives}")

    # Prepare examples
    train_examples = prepare_training_examples(df, max_rows=args.max_train_rows)

    train_dataloader = DataLoader(
        train_examples, shuffle=True, batch_size=args.batch_size
    )

    print("\nLoading base model:", args.model_name)
    model = SentenceTransformer(
        args.model_name,
        device=device,
        trust_remote_code=args.trust_remote_code,
    )

    train_loss = losses.MultipleNegativesRankingLoss(model)

    num_training_steps = len(train_dataloader) * args.epochs
    warmup_steps = math.ceil(num_training_steps * 0.1)
    print(f"Training steps: {num_training_steps}, warmup steps: {warmup_steps}")

    # Save training config
    train_config = {
        "model_name": args.model_name,
        "exp_name": args.exp_name,
        "train_file": args.train_file,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "max_train_rows": args.max_train_rows,
        "learning_rate": args.lr,
        "num_training_steps": num_training_steps,
        "warmup_steps": warmup_steps,
        "trust_remote_code": args.trust_remote_code,
    }
    with open(models_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(train_config, f, indent=2)

    print("\nStarting training...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        output_path=str(models_dir),
        use_amp=True,
        show_progress_bar=True,
    )

    print("\nTraining finished.")
    print("Fine-tuned model saved to:", models_dir)


if __name__ == "__main__":
    main()
