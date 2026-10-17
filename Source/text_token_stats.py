"""
Token-length audit for text-generation dataset.

Compute source (prompt) and target (provider text) token length distributions
using the FLAN-T5 tokenizer. Output: Source/text_token_stats.json

Usage:
  python Source/text_token_stats.py --train text_generation_train.jsonl \
      --model google/flan-t5-small --output Source/text_token_stats.json
"""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

sys_path = str(Path(__file__).parent)
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)
from text_gen_common import load_schema, format_input, format_target, read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="text_generation_train.jsonl")
    ap.add_argument("--model", default="google/flan-t5-small")
    ap.add_argument("--output", default="Source/text_token_stats.json")
    args = ap.parse_args()

    print(f"Loading tokenizer {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    trait_names, cat_of = load_schema()

    src_lens = []
    tgt_lens = []
    for rec in read_jsonl(args.train):
        prompt = format_input(rec["trait_probs"], trait_names, cat_of)
        target = format_target(rec["predictions"])
        src_lens.append(len(tok.encode(prompt)))
        tgt_lens.append(len(tok.encode(target)))

    import numpy as np
    arr_s = np.array(src_lens, dtype=float)
    arr_t = np.array(tgt_lens, dtype=float)

    def stats(arr, name):
        return {
            "name": name,
            "n": int(len(arr)),
            "min": int(arr.min()),
            "mean": round(float(arr.mean()), 1),
            "median": round(float(np.median(arr)), 1),
            "p90": round(float(np.percentile(arr, 90)), 1),
            "p95": round(float(np.percentile(arr, 95)), 1),
            "p99": round(float(np.percentile(arr, 99)), 1),
            "max": int(arr.max()),
        }

    # Choose limits from p99 with small headroom
    max_src = int(np.percentile(arr_s, 99)) + 10
    max_tgt = int(np.percentile(arr_t, 99)) + 20
    src_trunc = int((arr_s > max_src).sum())
    tgt_trunc = int((arr_t > max_tgt).sum())

    report = {
        "tokenizer_model": args.model,
        "vocab_size": tok.vocab_size,
        "source_stats": stats(arr_s, "source_prompt"),
        "target_stats": stats(arr_t, "target_text"),
        "recommended_max_source_length": max_src,
        "recommended_max_target_length": max_tgt,
        "source_truncation_count": src_trunc,
        "source_truncation_rate": round(src_trunc / len(arr_s), 6) if len(arr_s) else 0,
        "target_truncation_count": tgt_trunc,
        "target_truncation_rate": round(tgt_trunc / len(arr_t), 6) if len(arr_t) else 0,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()