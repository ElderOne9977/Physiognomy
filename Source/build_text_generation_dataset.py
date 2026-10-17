"""
Build text-generation datasets from OOF semantic probs + provider text.

Inputs:
  - oof_semantic_probs.csv  (image_id, 45 trait probs, row order == .npy)
  - labels_enriched.jsonl   (ground-truth provider text per image)
  - semantic_schema.json    (trait order + category)

Outputs:
  - text_generation_train.jsonl    : {image_id, trait_probs, traits_ordered, target_text, predictions}
  - text_generation_instruct.jsonl : {prompt, completion, image_id, trait_probs}

Deterministic: trait order = schema order; prompt lists traits in that order
with 2-decimal probs; no global fit; same norm_name as upstream.

Usage:
  python Source/build_text_generation_dataset.py
"""
import json
import os
import re
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import pandas as pd
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
OOF_CSV = ROOT / "oof_semantic_probs.csv"
JSONL = ROOT / "labels_enriched.jsonl"
SCHEMA = ROOT / "semantic_schema.json"
OUT_TRAIN = ROOT / "text_generation_train.jsonl"
OUT_INSTRUCT = ROOT / "text_generation_instruct.jsonl"

PRED_CATS = ["personality", "career", "relationships", "future_destiny", "wealth"]


def norm_name(v):
    s = os.path.basename(str(v).strip())
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\)\s+\(", ")(", s)
    return s.lower()


def predictions_to_text(pred):
    lines = []
    for cat in PRED_CATS:
        vals = pred.get(cat) or []
        vals = [x.strip() for x in vals if isinstance(x, str) and x.strip()]
        if vals:
            lines.append(f"{cat.replace('_',' ').title()}: " + " ".join(vals))
    return "\n".join(lines)


def main():
    import pandas as pd
    oof = pd.read_csv(OOF_CSV, encoding="utf-8-sig")
    if "image_id" not in oof.columns:
        raise ValueError("oof_semantic_probs.csv missing image_id")
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    trait_names = [t["name"] for t in schema["traits"]]
    # verify columns
    missing = [t for t in trait_names if t not in oof.columns]
    if missing:
        raise ValueError(f"OOF missing traits: {missing}")
    # load provider text map
    mp = {}
    with open(JSONL, encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            rec=json.loads(line)
            k=norm_name(rec.get("image",""))
            if k:
                mp[k]=rec
    n_matched=n_skipped=0
    out_train=[]
    out_inst=[]
    for _, row in oof.iterrows():
        img=row["image_id"]
        k=norm_name(img)
        rec=mp.get(k)
        if rec is None:
            n_skipped+=1
            continue
        n_matched+=1
        probs={t: float(row[t]) for t in trait_names}
        # deterministic prompt: traits in schema order, 2 decimals
        prob_str=", ".join(f"{t}={probs[t]:.2f}" for t in trait_names)
        pred=rec.get("predictions") or {}
        target=predictions_to_text(pred)
        if not target:
            continue
        # train record
        out_train.append({
            "image_id": str(img),
            "trait_probs": probs,
            "traits_ordered": trait_names,
            "target_text": target,
            "predictions": pred,
        })
        # instruct record: deterministic instruction format
        prompt=(
            "You are a physiognomy interpreter. Given semantic trait probabilities "
            "(0-1, ordered as in schema), generate a hedged physiognomy reading "
            "with five sections: Personality, Career, Relationships, Future & Destiny, Wealth. "
            "Use hedged language (may suggest, could indicate). "
            f"Traits: {prob_str}"
        )
        out_inst.append({
            "prompt": prompt,
            "completion": target,
            "image_id": str(img),
            "trait_probs": probs,
        })
    # write
    with open(OUT_TRAIN, "w", encoding="utf-8") as f:
        for r in out_train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(OUT_INSTRUCT, "w", encoding="utf-8") as f:
        for r in out_inst:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Matched {n_matched}, skipped {n_skipped} (no provider text)")
    print(f"Wrote {OUT_TRAIN} ({len(out_train)} rows)")
    print(f"Wrote {OUT_INSTRUCT} ({len(out_inst)} rows)")
    # sanity
    assert len(out_train)==len(out_inst), "train/instruct row mismatch"
    assert len(out_train)==n_matched, "count mismatch"
    # ponytail: OOF only, no leakage; upgrade: also emit non-OOF split for final-model training


if __name__=="__main__":
    main()