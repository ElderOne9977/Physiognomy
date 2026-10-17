"""
Train FLAN-T5-small on frozen OOF probs -> provider text (group-isolated splits).
Usage: python Source/train_text_generator.py --representation bins --output Source/models_text/flan_t5_small_bins
"""
import argparse, json, os, re
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq, EarlyStoppingCallback
from datasets import Dataset
import sys
sys.path.insert(0, str(Path(__file__).parent))
from text_gen_common import load_schema, format_input, format_target, prob_to_bin

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = ROOT / "text_generation_train.jsonl"
DEFAULT_SPLITS = Path(__file__).parent / "text_generation_splits.json"
DEFAULT_OUT = Path(__file__).parent / "models_text" / "flan_t5_small"
MODEL_NAME = "google/flan-t5-small"
MAX_SOURCE_LEN = 320
MAX_TARGET_LEN = 256

def norm_name(v):
    import re as _re, os as _os
    s = _os.path.basename(str(v).strip())
    s = _re.sub(r"\s+", " ", s)
    s = _re.sub(r"\)\s+\(", ")(", s)
    return s.lower()

def load_rows(train_path):
    rows=[]
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: rows.append(json.loads(line))
    return rows

def build_datasets(rows, splits_path, tok, trait_names, cat_of, representation="continuous"):
    sp=json.loads(Path(splits_path).read_text(encoding="utf-8"))
    train_set=set(sp["train_images"])
    val_set=set(sp["val_images"])
    test_set=set(sp["test_images"])
    def bucket():
        b={"train":[],"val":[],"test":[],"other":[]}
        for r in rows:
            k=norm_name(r["image_id"])
            rec={"image_id": r["image_id"], "probs": r["trait_probs"], "pred": r["predictions"], "target_text": r["target_text"]}
            if k in train_set: b["train"].append(rec)
            elif k in val_set: b["val"].append(rec)
            elif k in test_set: b["test"].append(rec)
            else: b["other"].append(rec)
        return b
    b=bucket()
    assert len(b["train"])+len(b["val"])+len(b["test"]) == len(rows), f"split mismatch {len(b['train'])} {len(b['val'])} {len(b['test'])} vs {len(rows)} other {len(b['other'])}"
    print(f"split counts train {len(b['train'])} val {len(b['val'])} test {len(b['test'])}")
    def fmt(split_rows):
        srcs, tgts = [], []
        for r in split_rows:
            srcs.append(format_input(r["probs"], trait_names, cat_of, representation=representation))
            tgts.append(r["target_text"])
        return srcs, tgts
    def to_ds(split_rows):
        srcs, tgts = fmt(split_rows)
        enc_src = tok(srcs, max_length=MAX_SOURCE_LEN, truncation=True)
        enc_tgt = tok(text_target=tgts, max_length=MAX_TARGET_LEN, truncation=True)
        data=[]
        for i in range(len(srcs)):
            data.append({"input_ids": enc_src["input_ids"][i], "attention_mask": enc_src["attention_mask"][i], "labels": enc_tgt["input_ids"][i]})
        return Dataset.from_list(data), srcs, tgts
    train_ds, _, _ = to_ds(b["train"])
    val_ds, _, _ = to_ds(b["val"])
    test_ds, _, _ = to_ds(b["test"])
    return train_ds, val_ds, test_ds, b

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train", default=str(DEFAULT_TRAIN))
    ap.add_argument("--splits", default=str(DEFAULT_SPLITS))
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--representation", choices=["continuous","bins"], default="continuous")
    ap.add_argument("--smoke", action="store_true")
    args=ap.parse_args()
    from transformers import set_seed
    set_seed(args.seed)
    trait_names, cat_of = load_schema()
    assert len(trait_names)==45, f"dim {len(trait_names)}"
    if args.representation=="bins":
        assert prob_to_bin(0.0)=="P0"
        assert prob_to_bin(1.0)=="P9"
        assert prob_to_bin(0.09)=="P0"
        assert prob_to_bin(0.10)=="P1"
        assert prob_to_bin(0.99)=="P9"
        for v in [0.0,0.15,0.29,0.5,0.82,1.0]:
            assert prob_to_bin(v) in [f"P{i}" for i in range(10)]
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    rows = load_rows(args.train)
    train_ds, val_ds, test_ds, buckets = build_datasets(rows, args.splits, tok, trait_names, cat_of, representation=args.representation)
    assert len(buckets["test"])==702, f"test {len(buckets['test'])}"
    assert len(trait_names)==45
    probe_src = format_input(buckets["train"][0]["probs"], trait_names, cat_of, representation=args.representation)
    assert buckets["train"][0]["target_text"] not in probe_src, "target leakage into input"
    sp=json.loads(Path(args.splits).read_text(encoding="utf-8"))
    train_set=set(norm_name(x) for x in sp["train_images"])
    test_set2=set(norm_name(x) for x in sp["test_images"])
    assert not (train_set & test_set2), "train/test group overlap"
    print(f"device check: torch {torch.__version__} cuda={torch.cuda.is_available()} device={'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        try:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
    else:
        print("WARNING: CUDA unavailable, CPU training will be slow (~2-4h for 5591 rows x 3 epochs). Using same config.")
    print(f"representation={args.representation}")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    if args.smoke:
        model.eval()
        print("smoke: train rows", len(train_ds), "val", len(val_ds))
        sample_probs = buckets["train"][0]["probs"]
        prompt = format_input(sample_probs, trait_names, cat_of, representation=args.representation)
        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_SOURCE_LEN)["input_ids"]
        import torch as _t
        with _t.no_grad():
            out = model.generate(ids, max_new_tokens=MAX_TARGET_LEN, do_sample=False, num_beams=1)
        print(tok.decode(out[0], skip_special_tokens=True)[:800])
        print("smoke ok")
        return
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    collator = DataCollatorForSeq2Seq(tok, model=model, padding=True)
    targs = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        report_to="none",
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds, processing_class=tok, data_collator=collator, callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
    trainer.train()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    meta={"model":args.model,"max_source_length":MAX_SOURCE_LEN,"max_target_length":MAX_TARGET_LEN,"epochs":args.epochs,"batch":args.batch,"lr":args.lr,"seed":args.seed,"train_n":len(train_ds),"val_n":len(val_ds),"test_n":len(test_ds),"representation":args.representation}
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved to {out_dir}")

if __name__=="__main__":
    main()