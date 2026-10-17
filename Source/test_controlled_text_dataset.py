# -*- coding: utf-8 -*-
"""Unit and regression tests for Phase 4 controlled content-plan dataset & modeling.

Run:
  python -u -B Source/test_controlled_text_dataset.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import build_controlled_text_dataset as B
import train_controlled_text_generator as T
import evaluate_controlled_model as E
from generate_text_controlled import build_content_plan, realize_deterministic, build_plan_prompt, validate_against_plan


PASS = []
FAIL = []


def check(name: str, cond: bool, extra: str = ""):
    if cond:
        PASS.append(name)
        print(f"PASS {name}")
    else:
        FAIL.append(name)
        print(f"FAIL {name}: {extra}", file=sys.stderr)


class TestControlledDataset(unittest.TestCase):
    def test_1_norm_name(self):
        self.assertEqual(B.norm_name("1 (1492) (1).jpg"), "1 (1492)(1).jpg")
        self.assertEqual(B.norm_name("Humans/1 (1).JPG"), "1 (1).jpg")
        check("norm_name handles spaces and extensions", True)

    def test_2_dataset_artifacts_exist(self):
        check("controlled dataset jsonl exists", B.DEFAULT_OUTPUT.exists())
        check("controlled dataset report exists", B.DEFAULT_REPORT.exists())
        report = json.loads(B.DEFAULT_REPORT.read_text(encoding="utf-8"))
        check("report total records 6983", report["total_records"] == 6983)
        check("report validation pass rate 1.0", report["validation_pass_rate"] == 1.0)
        check("report splits match expected", report["split_counts"]["train"] == 5591 and
              report["split_counts"]["val"] == 690 and report["split_counts"]["test"] == 702)

    def test_3_dataset_record_schema(self):
        with open(B.DEFAULT_OUTPUT, encoding="utf-8") as f:
            first_line = json.loads(f.readline())
        required_keys = {"image_id", "split", "prompt", "target_text", "plan"}
        check("record schema has all required keys", required_keys <= set(first_line.keys()))
        check("plan has categories", "categories" in first_line["plan"])
        check("target text has 5 sections", first_line["target_text"].count(":") >= 5)

    def test_4_no_target_leakage_in_prompt(self):
        with open(B.DEFAULT_OUTPUT, encoding="utf-8") as f:
            for _ in range(10):
                line = json.loads(f.readline())
                target = line["target_text"]
                prompt = line["prompt"]
                # The full rendered sentences should not be present verbatim in the prompt
                first_sentence = target.split(".")[0]
                check("no verbatim sentence leakage", first_sentence not in prompt)

    def test_5_cannot_overwrite_frozen_flan(self):
        frozen_dir = SRC / "models_text" / "flan_t5_small"
        with self.assertRaises(ValueError):
            with patch("sys.argv", ["train_controlled_text_generator.py", "--output", str(frozen_dir)]):
                T.main()
        check("cannot overwrite frozen flan baseline", True)

    def test_6_load_dataset_records(self):
        buckets = T.load_dataset_records(B.DEFAULT_OUTPUT, max_samples=20)
        check("load_dataset_records respects max_samples", len(buckets["train"]) == 20)
        check("val records loaded", len(buckets["val"]) > 0)
        check("test records loaded", len(buckets["test"]) == 702)

    def test_7_evaluate_model_filters_test_split(self):
        records = E.load_test_split(B.DEFAULT_OUTPUT, limit=5)
        check("load_test_split loads only test records", len(records) == 5 and all(r["split"] == "test" for r in records))


def main():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestControlledDataset)
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    print(f"\n{len(PASS)} PASS / {len(FAIL)} FAIL")
    return 0 if result.wasSuccessful() and not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
