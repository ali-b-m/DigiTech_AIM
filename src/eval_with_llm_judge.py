# src/eval_with_llm_judge.py
#
# Use an OpenAI model as a judge to compare two result files.
# Each file is just plain text (e.g. top-k results for some query).

import argparse
import json
import os
import re
from pathlib import Path

from openai import OpenAI


def parse_args():
    parser = argparse.ArgumentParser(
        description="Use an OpenAI LLM as a judge to compare two model outputs."
    )
    parser.add_argument(
        "--file_a",
        type=str,
        required=True,
        help="Path to first result file (e.g. fine-tuned model output).",
    )
    parser.add_argument(
        "--file_b",
        type=str,
        required=True,
        help="Path to second result file (e.g. baseline model output).",
    )
    parser.add_argument(
        "--label_a",
        type=str,
        default="Model_A",
        help="Short label for first model (shown to the judge).",
    )
    parser.add_argument(
        "--label_b",
        type=str,
        default="Model_B",
        help="Short label for second model (shown to the judge).",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="(Optional) The user query used for these results.",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default="gpt-4.1-mini",
        help="OpenAI model used as judge (e.g. gpt-4.1-mini, gpt-4.1).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/llm_judgements",
        help="Where to save the JSON judgement.",
    )
    return parser.parse_args()


def load_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return p.read_text(encoding="utf-8")


def build_prompt(query, label_a, text_a, label_b, text_b):
    """Build the prompt sent to the judge model."""
    if query is None:
        query_str = "UNKNOWN (no explicit query provided)."
    else:
        query_str = query

    prompt = f"""
You are an impartial judge evaluating e-commerce search results.

The user query is:
\"\"\"{query_str}\"\"\"


You are given two ranked result lists produced by two different models.
Each list shows products and their ranking for the same query.

--------------------
Results from {label_a}:
--------------------
{text_a}

--------------------
Results from {label_b}:
--------------------
{text_b}

Please compare these two result lists. Consider:

1. Relevance: How well do the products match the user query?
2. Diversity: Are the results varied or are they near-duplicates?
3. Usefulness: Would this list help a real shopper find the right product quickly?
4. Safety / Suitability: Are there obviously inappropriate results given a generic shopper?

Return your answer as valid JSON with the following structure:

{{
  "query": "...",
  "model_a_label": "{label_a}",
  "model_b_label": "{label_b}",
  "scores": {{
    "model_a": {{
      "relevance": <number 0-10>,
      "diversity": <number 0-10>,
      "usefulness": <number 0-10>,
      "safety": <number 0-10>
    }},
    "model_b": {{
      "relevance": <number 0-10>,
      "diversity": <number 0-10>,
      "usefulness": <number 0-10>,
      "safety": <number 0-10>
    }}
  }},
  "overall_better": "<\"model_a\" or \"model_b\" or \"tie\">",
  "explanation": "Short natural-language explanation comparing the two."
}}

Only output JSON, no extra text.
"""
    return prompt


def strip_code_fences(text: str) -> str:
    """
    If the model returns ```json ... ``` or ``` ... ```, strip the fences
    and keep only the inner JSON.
    """
    t = text.strip()
    if t.startswith("```"):
        # Try to capture the content between the first ```... and the last ```
        m = re.search(r"```(?:json)?\s*(.*)```", t, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return t


def main():
    args = parse_args()

    # 1. Read the two result files
    text_a = load_text(args.file_a)
    text_b = load_text(args.file_b)

    # 2. Build prompt
    prompt = build_prompt(
        query=args.query,
        label_a=args.label_a,
        text_a=text_a,
        label_b=args.label_b,
        text_b=text_b,
    )

    # 3. Init OpenAI client (expects OPENAI_API_KEY in env)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set. "
            "Set it before running this script."
        )

    client = OpenAI(api_key=api_key)

    print("Calling OpenAI judge model:", args.judge_model)

    response = client.chat.completions.create(
        model=args.judge_model,
        messages=[
            {"role": "system", "content": "You are a strict JSON-only judge."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )

    raw_text = response.choices[0].message.content or ""
    cleaned = strip_code_fences(raw_text)

    # 4. Parse JSON (if parsing fails, keep raw text)
    try:
        judgement = json.loads(cleaned)
        parsed_ok = True
    except json.JSONDecodeError:
        judgement = {"raw_text": raw_text}
        parsed_ok = False

    # 5. Save to file
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_a = Path(args.file_a).stem
    base_b = Path(args.file_b).stem

    if args.query:
        safe_query = "".join(c for c in args.query if c.isalnum() or c in "-_")[:40]
    else:
        safe_query = "no_query"

    out_name = f"judge_{safe_query}__{base_a}__vs__{base_b}.json"
    out_path = out_dir / out_name

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(judgement, f, indent=2, ensure_ascii=False)

    print("\nLLM judgement saved to:")
    print("  ", out_path)

    # 6. Nice preview
    print("\nPreview:\n")
    if parsed_ok:
        print(json.dumps(judgement, indent=2, ensure_ascii=False))

        # Optional: small summary
        try:
            a = judgement["scores"]["model_a"]
            b = judgement["scores"]["model_b"]
            overall = judgement.get("overall_better", "unknown")
            print("\nSummary:")
            print(f"  {args.label_a} scores: {a}")
            print(f"  {args.label_b} scores: {b}")
            print(f"  Overall better: {overall}")
        except Exception:
            pass
    else:
        print("(Could not parse JSON, raw model output below)")
        print(raw_text)


if __name__ == "__main__":
    main()
