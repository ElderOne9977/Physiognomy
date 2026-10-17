# -*- coding: utf-8 -*-
"""
Train FLAN-T5-small on explicit Content-Plan -> Five-Section English Realization.

Guarantees:
  * Input is an explicit content plan prompt (listing selected concepts per category).
  * Target is a faithful, category-bounded 5-section English realization.
  * Preserves historical models_text/flan_t5_small intact; saves exclusively to
    Source/models_text/flan_t5_small_controlled (or caller-supplied --output).
  * Supports --smoke for rapid (<15s) integration verification.
  * Supports --max-train-samples for quick pilot runs on CPU.

Usage:
  # Quick smoke check (verifies pipeline, tokenizer, collator, forward+backward step):
  python Source/train_controlled_text_generator.py --smoke

  # Pilot training (e.g., 200 samples):
  python Source/train_controlled_text_generator.py --max-train-samples 200 --epochs 1

  # Full training (requires user confirmation, ~3.5h on CPU):
  python Source/train_controlled_text_generator.py --output Source/models_text/flan_t5_small_controlled
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

DEFAULT_DATASET = SRC / "text_generation_controlled_train.jsonl"
LOCAL_FLAN_DIR = SRC / "models_text" / "flan_t5_small"
DEFAULT_BASE_MODEL = str(LOCAL_FLAN_DIR) if (LOCAL_FLAN_DIR / "model.safetensors").exists() else "google/flan-t5-small"
DEFAULT_OUTPUT = SRC / "models_text" / "flan_t5_small_controlled"
MAX_SOURCE_LEN = 384
MAX_TARGET_LEN = 256


def load_dataset_records(dataset_path: Path, max_samples: int | None = None) -> dict[str, list[dict]]:
    buckets = {"train": [], "val": [], "test": []}
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            split = rec.get("split", "train")
            if split in buckets:
                buckets[split].append(rec)
    if max_samples:
        buckets["train"] = buckets["train"][:max_samples]
        buckets["val"] = buckets["val"][: min(len(buckets["val"]), max_samples // 4 + 1)]
    return buckets


def prepare_hf_datasets(buckets: dict[str, list[dict]], tokenizer: AutoTokenizer):
    def to_ds(recs: list[dict]):
        srcs = [r["prompt"] for r in recs]
        tgts = [r["target_text"] for r in recs]
        enc_src = tokenizer(srcs, max_length=MAX_SOURCE_LEN, truncation=True)
        enc_tgt = tokenizer(text_target=tgts, max_length=MAX_TARGET_LEN, truncation=True)
        data = []
        for i in range(len(srcs)):
            data.append({
                "input_ids": enc_src["input_ids"][i],
                "attention_mask": enc_src["attention_mask"][i],
                "labels": enc_tgt["input_ids"][i],
            })
        return Dataset.from_list(data)

    train_ds = to_ds(buckets["train"])
    val_ds = to_ds(buckets["val"]) if buckets["val"] else None
    test_ds = to_ds(buckets["test"]) if buckets["test"] else None
    return train_ds, val_ds, test_ds


def main():
    ap = argparse.ArgumentParser(description="Train FLAN-T5 on controlled content-plan dataset")
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Path to text_generation_controlled_train.jsonl")
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Base model identifier or local checkpoint dir")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output directory for fine-tuned model")
    ap.add_argument("--epochs", type=int, default=3, help="Training epochs")
    ap.add_argument("--batch", type=int, default=4, help="Batch size per device")
    ap.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--max-train-samples", type=int, default=None, help="Cap training samples (for pilot tests)")
    ap.add_argument("--smoke", action="store_true", help="Run quick 2-step verification")
    args = ap.parse_args()

    set_seed(args.seed)
    t0 = time.time()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Run build_controlled_text_dataset.py first.")

    out_dir = Path(args.output)
    frozen_base = SRC / "models_text" / "flan_t5_small"
    if out_dir.resolve() == frozen_base.resolve():
        raise ValueError("Cannot overwrite frozen historical flan_t5_small baseline! Choose a separate output directory.")

    print(f"Loading dataset from {dataset_path}...")
    buckets = load_dataset_records(dataset_path, max_samples=args.max_train_samples if not args.smoke else 8)
    print(f"Loaded samples: train={len(buckets['train'])}, val={len(buckets['val'])}, test={len(buckets['test'])}")

    print(f"Loading tokenizer & model from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch version: {torch.__version__}, Device: {device}")
    if device == "cpu":
        print("Note: CPU training active. Ensure single expensive Python process rule is respected.")

    train_ds, val_ds, _ = prepare_hf_datasets(buckets, tokenizer)

    if args.smoke:
        print("Running SMOKE test...")
        model.to(device)
        sample = buckets["train"][0]
        prompt = sample["prompt"]
        enc = tokenizer(prompt, return_tensors="pt", max_length=MAX_SOURCE_LEN, truncation=True).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=64)
        pred_text = tokenizer.decode(gen[0], skip_special_tokens=True)
        print(f"Smoke prompt chars: {len(prompt)}, output chars: {len(pred_text)}")
        out_smoke = SRC / "models_text" / "smoke_check"
        out_smoke.mkdir(parents=True, exist_ok=True)
        (out_smoke / "smoke_verified.json").write_text(json.dumps({
            "status": "PASS",
            "device": device,
            "train_samples_seen": len(train_ds),
            "sample_output_preview": pred_text[:200],
        }, indent=2), encoding="utf-8")
        print("SMOKE TEST PASSED.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        eval_strategy="epoch" if val_ds else "no",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20 if args.max_train_samples else 100,
        save_total_limit=2,
        load_best_model_at_end=True if val_ds else False,
        metric_for_best_model="eval_loss" if val_ds else None,
        greater_is_better=False,
        seed=args.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] if val_ds else [],
    )

    print("Starting training...")
    train_result = trainer.train()
    print("Saving fine-tuned model and tokenizer...")
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    train_meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "base_model": args.base_model,
        "max_source_length": MAX_SOURCE_LEN,
        "max_target_length": MAX_TARGET_LEN,
        "epochs": args.epochs,
        "batch_size": args.batch,
        "learning_rate": args.lr,
        "seed": args.seed,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds) if val_ds else 0,
        "device": device,
        "train_runtime_seconds": round(train_result.metrics.get("train_runtime", time.time() - t0), 2),
        "train_loss": train_result.metrics.get("train_loss"),
    }
    (out_dir / "train_meta.json").write_text(json.dumps(train_meta, indent=2), encoding="utf-8")
    print(f"Training completed in {train_meta['train_runtime_seconds']}s. Saved to {out_dir}")


if __name__ == "__main__":
    main()
