# -*- coding: utf-8 -*-
"""
Build controlled content-plan training dataset from OOF semantic probabilities
and calibrated trait thresholds.

Inputs:
  - oof_semantic_probs.csv (6983 rows x 46 cols: image_id + 45 trait probabilities)
  - Source/models_semantic/trait_thresholds.json (calibrated per-trait thresholds)
  - Source/text_generation_splits.json (group-isolated train/val/test split)

Outputs:
  - Source/text_generation_controlled_train.jsonl (6983 rows: image_id, split, prompt, target_text, plan)
  - Source/controlled_text_dataset_report.json (verification metrics and summary statistics)

Usage:
  python Source/build_controlled_text_dataset.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import pandas as pd

from text_gen_common import (
    CATEGORY_ORDER,
    CATEGORY_TITLES,
    load_schema,
    parse_sections,
    validate_structure,
)
from generate_text_template import load_thresholds, DEFAULT_THRESHOLD
from generate_text_controlled import (
    build_content_plan,
    realize_deterministic,
    build_plan_prompt,
    validate_against_plan,
    plan_faithfulness,
)

DEFAULT_OOF = ROOT / "oof_semantic_probs.csv"
DEFAULT_THRESHOLDS = SRC / "models_semantic" / "trait_thresholds.json"
DEFAULT_SPLITS = SRC / "text_generation_splits.json"
DEFAULT_OUTPUT = SRC / "text_generation_controlled_train.jsonl"
DEFAULT_REPORT = SRC / "controlled_text_dataset_report.json"


def norm_name(v: str) -> str:
    s = os.path.basename(str(v).strip())
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\)\s+\(", ")(", s)
    return s.lower()


def build_dataset(
    oof_path: Path = DEFAULT_OOF,
    thresholds_path: Path = DEFAULT_THRESHOLDS,
    splits_path: Path = DEFAULT_SPLITS,
    output_path: Path = DEFAULT_OUTPUT,
    report_path: Path = DEFAULT_REPORT,
    verbose: bool = True,
) -> dict:
    t0 = time.time()
    if verbose:
        print(f"Loading OOF probabilities from {oof_path}...")
    oof = pd.read_csv(oof_path, encoding="utf-8-sig")
    if "image_id" not in oof.columns:
        raise ValueError(f"{oof_path} missing 'image_id' column")

    schema_traits, _ = load_schema()
    missing_traits = [t for t in schema_traits if t not in oof.columns]
    if missing_traits:
        raise ValueError(f"OOF missing required schema traits: {missing_traits}")

    if verbose:
        print(f"Loading thresholds from {thresholds_path}...")
    tmap, strategy = load_thresholds(str(thresholds_path))
    if tmap is None:
        raise FileNotFoundError(f"Thresholds file not found: {thresholds_path}")

    if verbose:
        print(f"Loading splits from {splits_path}...")
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    train_set = {norm_name(x) for x in splits["train_images"]}
    val_set = {norm_name(x) for x in splits["val_images"]}
    test_set = {norm_name(x) for x in splits["test_images"]}

    rows = []
    split_counts = {"train": 0, "val": 0, "test": 0, "unknown": 0}
    empty_sections_count = {cat: 0 for cat in CATEGORY_ORDER}
    validation_failures = 0
    all_selected_traits = []

    prompt_lens = []
    target_lens = []

    for idx, row in oof.iterrows():
        img_id = str(row["image_id"])
        k = norm_name(img_id)

        if k in train_set:
            split = "train"
        elif k in val_set:
            split = "val"
        elif k in test_set:
            split = "test"
        else:
            split = "unknown"
        split_counts[split] += 1

        probs = {t: float(row[t]) for t in schema_traits}
        plan = build_content_plan(probs, thresholds=tmap, threshold=DEFAULT_THRESHOLD)
        prompt = build_plan_prompt(plan)
        target_text = realize_deterministic(plan)

        # Validate ground truth
        ok, reasons, _ = validate_against_plan(target_text, plan)
        if not ok:
            validation_failures += 1

        v_struct = validate_structure(target_text)
        if not v_struct["valid"]:
            validation_failures += 1

        for cat in CATEGORY_ORDER:
            if plan["categories"][cat]["empty"]:
                empty_sections_count[cat] += 1

        all_selected_traits.extend(plan["selected_all"])
        prompt_lens.append(len(prompt))
        target_lens.append(len(target_text))

        rec = {
            "image_id": img_id,
            "split": split,
            "prompt": prompt,
            "target_text": target_text,
            "plan": {
                "selected_all": plan["selected_all"],
                "omitted_all": plan["omitted_all"],
                "n_selected": plan["n_selected"],
                "categories": {
                    cat: {
                        "selected": plan["categories"][cat]["selected"],
                        "omitted": plan["categories"][cat]["omitted"],
                        "empty": plan["categories"][cat]["empty"],
                    }
                    for cat in CATEGORY_ORDER
                },
            },
        }
        rows.append(rec)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_records": len(rows),
        "split_counts": split_counts,
        "validation_failures": validation_failures,
        "validation_pass_rate": 1.0 - (validation_failures / len(rows) if rows else 0),
        "avg_prompt_chars": float(sum(prompt_lens) / len(prompt_lens)) if prompt_lens else 0.0,
        "avg_target_chars": float(sum(target_lens) / len(target_lens)) if target_lens else 0.0,
        "empty_sections_by_category": empty_sections_count,
        "total_selected_concepts": len(all_selected_traits),
        "avg_selected_per_row": float(len(all_selected_traits) / len(rows)) if rows else 0.0,
        "elapsed_seconds": round(time.time() - t0, 2),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if verbose:
        print(f"Generated {len(rows)} records in {report['elapsed_seconds']}s")
        print(f"Splits: {split_counts}")
        print(f"Validation pass rate: {report['validation_pass_rate']*100:.2f}%")
        print(f"Output saved to: {output_path}")
        print(f"Report saved to: {report_path}")

    return report


def main():
    ap = argparse.ArgumentParser(description="Build controlled content-plan training dataset")
    ap.add_argument("--oof", default=str(DEFAULT_OOF), help="Path to oof_semantic_probs.csv")
    ap.add_argument("--thresholds", default=str(DEFAULT_THRESHOLDS), help="Path to trait_thresholds.json")
    ap.add_argument("--splits", default=str(DEFAULT_SPLITS), help="Path to text_generation_splits.json")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output jsonl path")
    ap.add_argument("--report", default=str(DEFAULT_REPORT), help="Output report json path")
    args = ap.parse_args()

    build_dataset(
        oof_path=Path(args.oof),
        thresholds_path=Path(args.thresholds),
        splits_path=Path(args.splits),
        output_path=Path(args.output),
        report_path=Path(args.report),
        verbose=True,
    )


if __name__ == "__main__":
    main()
