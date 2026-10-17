"""
Generate text with trained FLAN-T5 on frozen test split (greedy).
Supports --representation continuous|bins (must match training).
"""
import argparse, json, re, os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from text_gen_common import load_schema, format_input, parse_sections
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_TRAIN=ROOT/"text_generation_train.jsonl"
DEFAULT_SPLITS=Path(__file__).parent/"text_generation_splits.json"
DEFAULT_MODEL=Path(__file__).parent/"models_text"/"flan_t5_small"
MAX_SOURCE_LEN=320
MAX_TARGET_LEN=256

def norm_name(v):
    import re as _re, os as _os
    s=_os.path.basename(str(v).strip()); s=_re.sub(r"\s+"," ",s); s=_re.sub(r"\)\s+\(",")(",s); return s.lower()

def load_rows(p):
    rows=[]
    with open(p,encoding="utf-8") as f:
        for l in f:
            l=l.strip()
            if l: rows.append(json.loads(l))
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--splits", default=str(DEFAULT_SPLITS))
    ap.add_argument("--train", default=str(DEFAULT_TRAIN))
    ap.add_argument("--output", default=str(Path(__file__).parent/"generated_text_model_test.jsonl"))
    ap.add_argument("--max-source", type=int, default=MAX_SOURCE_LEN)
    ap.add_argument("--max-target", type=int, default=MAX_TARGET_LEN)
    ap.add_argument("--representation", choices=["continuous","bins"], default=None)
    args=ap.parse_args()
    trait_names, cat_of = load_schema()
    # infer representation from train_meta if not supplied
    rep=args.representation
    if rep is None:
        meta_path=Path(args.model)/"train_meta.json"
        if meta_path.exists():
            try:
                m=json.loads(meta_path.read_text(encoding="utf-8"))
                rep=m.get("representation","continuous")
            except Exception:
                rep="continuous"
        else:
            rep="continuous"
    print(f"representation={rep}")
    tok=AutoTokenizer.from_pretrained(args.model)
    model=AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model.eval()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    rows=load_rows(args.train)
    splits=json.loads(Path(args.splits).read_text(encoding="utf-8"))
    test_set=set(splits["test_images"])
    test_rows=[r for r in rows if norm_name(r["image_id"]) in test_set]
    # sort to match evaluate_text_model / audit order (norm_name sort)
    test_rows=sorted(test_rows, key=lambda x: norm_name(x["image_id"]))
    print(f"generating {len(test_rows)} test rows with {args.model} on {device}")
    out=[]
    prompts=[format_input(r["trait_probs"], trait_names, cat_of, representation=rep) for r in test_rows]
    BS=16
    with torch.no_grad():
        for i in range(0, len(prompts), BS):
            chunk=prompts[i:i+BS]
            enc=tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=args.max_source)
            gen=model.generate(**{k: v.to(device) for k, v in enc.items()},
                               max_new_tokens=args.max_target, do_sample=False, num_beams=1)
            texts=tok.batch_decode(gen, skip_special_tokens=True)
            for r, t in zip(test_rows[i:i+BS], texts):
                out.append({"image_id": r["image_id"], "generated_text": t, "probs": r["trait_probs"]})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output,"w",encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False)+"\n")
    print(f"Wrote {len(out)} to {args.output}")
    from text_gen_common import validate_structure
    valid=sum(1 for r in out if validate_structure(r["generated_text"])["valid"])
    print(f"valid 5-section {valid}/{len(out)}")
    assert len(out)==702, f"expected 702 got {len(out)}"

if __name__=="__main__":
    main()