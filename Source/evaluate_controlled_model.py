# -*- coding: utf-8 -*-
"""
Evaluate fine-tuned or baseline Seq2Seq models on the Content-Plan Test Split.

Inputs:
  - Source/text_generation_controlled_train.jsonl (test split rows)
  - Model directory (--model)

Outputs:
  - Evaluation JSON report (--output) with:
    * structure_valid_rate
    * plan_precision_mean
    * plan_recall_mean
    * plan_f1_mean
    * genuine_unsupported_rate
    * acceptance_rate (fraction passing validate_against_plan without fallback)
    * per-category precision / recall
    * sample predictions

Usage:
  # Fast evaluation on 24 test rows:
  python Source/evaluate_controlled_model.py --limit 24

  # Full evaluation on all 702 test rows:
  python Source/evaluate_controlled_model.py --model Source/models_text/flan_t5_small_controlled
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from text_gen_common import (
    CATEGORY_ORDER,
    CATEGORY_TITLES,
    parse_sections,
    validate_structure,
)
from generate_text_controlled import (
    extract_concepts,
    lexicon_collisions,
    plan_faithfulness,
    validate_against_plan,
)

DEFAULT_DATASET = SRC / "text_generation_controlled_train.jsonl"
DEFAULT_MODEL = SRC / "models_text" / "flan_t5_small"
DEFAULT_OUT = SRC / "controlled_model_eval_report.json"
MAX_SOURCE_LEN = 384
MAX_NEW_TOKENS = 256


def load_test_split(dataset_path: Path, limit: int | None = None) -> list[dict]:
    test_rows = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("split") == "test":
                test_rows.append(rec)
    if limit is not None and limit > 0:
        test_rows = test_rows[:limit]
    return test_rows


def evaluate_model(
    model_dir: Path,
    dataset_path: Path = DEFAULT_DATASET,
    output_path: Path = DEFAULT_OUT,
    limit: int | None = None,
    batch_size: int = 8,
    verbose: bool = True,
) -> dict:
    t0 = time.time()
    test_rows = load_test_split(dataset_path, limit=limit)
    if not test_rows:
        raise ValueError(f"No test split records found in {dataset_path}")

    if verbose:
        print(f"Loaded {len(test_rows)} test split records (limit={limit})")
        print(f"Loading model from {model_dir}...")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    prompts = [r["prompt"] for r in test_rows]
    generated_texts = []

    if verbose:
        print(f"Generating completions on {device} (greedy decoding)...")

    t_gen0 = time.time()
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SOURCE_LEN,
        ).to(device)
        with torch.no_grad():
            gen_tokens = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
            )
        for g in gen_tokens:
            text = tokenizer.decode(g, skip_special_tokens=True)
            generated_texts.append(text)
    gen_duration = time.time() - t_gen0

    # Evaluate outputs
    valid_structures = 0
    accepted_by_validator = 0
    precisions = []
    recalls = []
    f1s = []
    genuine_unsupported_counts = []
    category_metrics = {cat: {"precision": [], "recall": []} for cat in CATEGORY_ORDER}

    sample_comparisons = []

    for idx, (rec, cand_text) in enumerate(zip(test_rows, generated_texts)):
        plan = rec["plan"]
        v_struct = validate_structure(cand_text)
        if v_struct["valid"]:
            valid_structures += 1

        ok, reasons, expressed = validate_against_plan(cand_text, plan)
        if ok:
            accepted_by_validator += 1

        f_metrics = plan_faithfulness(cand_text, plan)
        precisions.append(f_metrics["precision"])
        recalls.append(f_metrics["recall"])
        f1s.append(f_metrics["f1"])
        genuine_unsupported_counts.append(len(f_metrics["genuine_unsupported"]))

        # Per category
        sections, _ = parse_sections(cand_text)
        for cat in CATEGORY_ORDER:
            title = CATEGORY_TITLES[cat]
            body = sections.get(title, "")
            sel = set(plan["categories"][cat]["selected"])
            exp = extract_concepts(body)
            tp = len(exp & sel)
            p = tp / len(exp) if exp else (1.0 if not sel else 0.0)
            r = tp / len(sel) if sel else 1.0
            category_metrics[cat]["precision"].append(p)
            category_metrics[cat]["recall"].append(r)

        if idx < 3:
            sample_comparisons.append({
                "image_id": rec["image_id"],
                "target_text": rec["target_text"],
                "generated_text": cand_text,
                "validator_ok": ok,
                "reasons": reasons,
                "faithfulness": f_metrics,
            })

    n = len(test_rows)
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_dir": str(model_dir),
        "dataset": str(dataset_path),
        "n_samples": n,
        "device": device,
        "gen_duration_seconds": round(gen_duration, 2),
        "seconds_per_sample": round(gen_duration / n, 3) if n else 0.0,
        "structure_valid_rate": round(valid_structures / n, 4),
        "acceptance_rate": round(accepted_by_validator / n, 4),
        "plan_precision_mean": round(float(sum(precisions) / n), 4),
        "plan_recall_mean": round(float(sum(recalls) / n), 4),
        "plan_f1_mean": round(float(sum(f1s) / n), 4),
        "mean_genuine_unsupported_per_sample": round(float(sum(genuine_unsupported_counts) / n), 4),
        "per_category": {
            cat: {
                "precision_mean": round(float(sum(category_metrics[cat]["precision"]) / n), 4),
                "recall_mean": round(float(sum(category_metrics[cat]["recall"]) / n), 4),
            }
            for cat in CATEGORY_ORDER
        },
        "sample_comparisons": sample_comparisons,
        "total_elapsed_seconds": round(time.time() - t0, 2),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if verbose:
        print("\n=== Evaluation Results ===")
        print(f"Structure Valid Rate:   {report['structure_valid_rate']*100:.1f}%")
        print(f"Validator Acceptance:  {report['acceptance_rate']*100:.1f}%")
        print(f"Plan Precision Mean:   {report['plan_precision_mean']*100:.1f}%")
        print(f"Plan Recall Mean:      {report['plan_recall_mean']*100:.1f}%")
        print(f"Plan F1 Mean:          {report['plan_f1_mean']*100:.1f}%")
        print(f"Unsupported per sample: {report['mean_genuine_unsupported_per_sample']:.2f}")
        print(f"Speed:                 {report['seconds_per_sample']}s/sample")
        print(f"Report saved to:       {output_path}")

    return report


def main():
    ap = argparse.ArgumentParser(description="Evaluate text generator model on content plan split")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to model directory")
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Path to text_generation_controlled_train.jsonl")
    ap.add_argument("--output", default=str(DEFAULT_OUT), help="Path to write evaluation report json")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of test samples (e.g. 24)")
    ap.add_argument("--batch", type=int, default=8, help="Batch size")
    args = ap.parse_args()

    evaluate_model(
        model_dir=Path(args.model),
        dataset_path=Path(args.dataset),
        output_path=Path(args.output),
        limit=args.limit,
        batch_size=args.batch,
        verbose=True,
    )


if __name__ == "__main__":
    main()
