"""
Evaluate template-generated physiognomy text.

Metrics:
  - ROUGE-L (LCS-based, per-row P/R/F then macro)
  - BLEU-4 + BLEU-1 (corpus, low-cost pure-Python)
  - Semantic precision/recall/F1 (predicted active traits vs true binary labels)
  - Unsupported concept count (predicted trait not in schema)
  - Hallucination rate (1 - semantic precision)

Inputs default to the pipeline artefacts so you can run with no args:
  python Source/evaluate_text_generation.py
  python Source/evaluate_text_generation.py --generated out.jsonl --oof oof_semantic_probs.csv

If --generated is not given, the script generates text on-the-fly from OOF
via generate_text_template.generate_text (threshold 0.5) and evaluates that.
Reference text comes from labels_enriched.jsonl (provider text) joined on
norm_name.
"""
import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OOF = ROOT / "oof_semantic_probs.csv"
DEFAULT_LABELS_CSV = ROOT / "semantic_labels.csv"
DEFAULT_JSONL = ROOT / "labels_enriched.jsonl"
DEFAULT_SCHEMA = ROOT / "semantic_schema.json"
DEFAULT_TRAIN = ROOT / "text_generation_train.jsonl"

PRED_CATS = ["personality", "career", "relationships", "future_destiny", "wealth"]
THRESHOLD = 0.5


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


def lcs_len(a_tokens, b_tokens):
    # DP with O(min(n,m)) space, pure python; n,m ~ 200-400 so fine
    if not a_tokens or not b_tokens:
        return 0
    # ensure a is shorter
    if len(a_tokens) > len(b_tokens):
        a_tokens, b_tokens = b_tokens, a_tokens
    prev = [0] * (len(a_tokens) + 1)
    for b in b_tokens:
        cur = [0] * (len(a_tokens) + 1)
        for i, a in enumerate(a_tokens, 1):
            if a == b:
                cur[i] = prev[i-1] + 1
            else:
                cur[i] = prev[i] if prev[i] >= cur[i-1] else cur[i-1]
        prev = cur
    return prev[len(a_tokens)]


def rouge_l_one(hyp: str, ref: str):
    ht = hyp.lower().split()
    rt = ref.lower().split()
    if not ht or not rt:
        return 0.0, 0.0, 0.0
    l = lcs_len(ht, rt)
    p = l / len(ht) if ht else 0.0
    r = l / len(rt) if rt else 0.0
    f = (2*p*r/(p+r)) if (p+r) else 0.0
    return p, r, f


def bleu_corpus(hyps, refs, max_n=4):
    # Simple corpus BLEU without smoothing beyond replace-zero-with-eps
    # hyps, refs are lists of token lists
    precisions = []
    for n in range(1, max_n+1):
        matches = 0
        total = 0
        for ht, rt in zip(hyps, refs):
            if len(ht) < n:
                continue
            hc = Counter(tuple(ht[i:i+n]) for i in range(len(ht)-n+1))
            rc = Counter(tuple(rt[i:i+n]) for i in range(len(rt)-n+1))
            overlap = sum(min(hc[g], rc[g]) for g in hc)
            matches += overlap
            total += max(0, len(ht)-n+1)
        if total == 0:
            precisions.append(0.0)
        else:
            precisions.append(matches / total if total else 0.0)
    # brevity penalty
    hyp_len = sum(len(h) for h in hyps)
    ref_len = sum(len(r) for r in refs)
    if hyp_len == 0:
        bp = 0.0
    elif hyp_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / hyp_len) if hyp_len else 0.0
    # geometric mean, with epsilon for zero precisions (common cheap smoothing)
    if any(p == 0 for p in precisions):
        # use smoothing: replace 0 with 1/total-like epsilon
        # but keep BLEU low to signal poor overlap; report 0 for strict
        bleu = 0.0
        # also compute smoothed version for display
        sm = [p if p>0 else 1e-9 for p in precisions]
        bleu_sm = bp * math.exp(sum(math.log(p) for p in sm) / max_n)
    else:
        bleu = bp * math.exp(sum(math.log(p) for p in precisions) / max_n)
        bleu_sm = bleu
    bleu1 = precisions[0] if precisions else 0.0
    return bleu, bleu_sm, bleu1, precisions, bp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated", default=None, help="jsonl with {image, generated_text, probs}; if omitted, generate from OOF")
    ap.add_argument("--oof", default=str(DEFAULT_OOF))
    ap.add_argument("--labels-csv", default=str(DEFAULT_LABELS_CSV))
    ap.add_argument("--jsonl", default=str(DEFAULT_JSONL))
    ap.add_argument("--schema", default=str(DEFAULT_SCHEMA))
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--out", default=str(Path(__file__).parent / "text_generation_eval.json"))
    args = ap.parse_args()

    import pandas as pd
    # schema
    schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
    trait_names = [t["name"] for t in schema["traits"]]
    trait_set = set(trait_names)

    # true binary labels per image (semantic_labels.csv)
    lab_df = pd.read_csv(args.labels_csv, encoding="utf-8-sig")
    c0 = lab_df.columns[0]
    if c0.strip().lower().lstrip("\ufeff") == "image" and c0 != "image":
        lab_df = lab_df.rename(columns={c0: "image"})
    lab_df["norm_key"] = lab_df["image"].apply(norm_name)
    true_map = {}
    for _, r in lab_df.iterrows():
        k = r["norm_key"]
        true_set = {t for t in trait_names if int(r[t]) == 1}
        true_map[k] = true_set

    # provider text map (for ROUGE/BLEU ref)
    ref_map = {}
    with open(args.jsonl, encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            rec=json.loads(line)
            k=norm_name(rec.get("image",""))
            if k:
                ref_map[k] = predictions_to_text(rec.get("predictions") or {})

    # generated texts
    gen_items = []  # list of (norm_key, image, generated_text, active_set)
    if args.generated:
        with open(args.generated, encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line:
                    continue
                rec=json.loads(line)
                img = rec.get("image") or rec.get("image_id") or ""
                k = norm_name(img)
                text = rec.get("generated_text") or rec.get("text") or ""
                active = set(rec.get("active_traits") or [])
                # if active missing but probs present, threshold
                if not active and "probs" in rec:
                    active = {t for t,p in rec["probs"].items() if float(p) >= args.threshold and t in trait_set}
                gen_items.append((k, img, text, active))
    else:
        # generate on the fly from OOF
        from generate_text_template import generate_text
        oof = pd.read_csv(args.oof, encoding="utf-8-sig")
        for _, row in oof.iterrows():
            img = str(row["image_id"])
            k = norm_name(img)
            probs = {t: float(row[t]) for t in trait_names if t in oof.columns}
            text, active_list = generate_text(probs, threshold=args.threshold)
            active = set(active_list)
            gen_items.append((k, img, text, active))

    # filter to those with both true and ref (should be nearly all)
    rows = []
    for k, img, text, active in gen_items:
        if k not in true_map or k not in ref_map:
            continue
        rows.append((k, img, text, active, true_map[k], ref_map[k]))

    print(f"Evaluating {len(rows)} rows (gen {len(gen_items)}, true {len(true_map)}, ref {len(ref_map)})")
    if not rows:
        raise SystemExit("no overlapping rows for evaluation")

    # ROUGE-L
    rouge_ps, rouge_rs, rouge_fs = [], [], []
    hyps_tokens, refs_tokens = [], []
    for _, _, hyp, _, _, ref in rows:
        p, r, f = rouge_l_one(hyp, ref)
        rouge_ps.append(p); rouge_rs.append(r); rouge_fs.append(f)
        hyps_tokens.append(hyp.lower().split())
        refs_tokens.append(ref.lower().split())

    bleu, bleu_sm, bleu1, precisions, bp = bleu_corpus(hyps_tokens, refs_tokens, max_n=4)

    # semantic
    sem_ps, sem_rs, sem_fs = [], [], []
    unsupported_total = 0
    hallucinated_rows = 0
    for _, _, _, pred, true, _ in rows:
        # unsupported = pred not in schema (template guarantees 0, but check)
        unsupp = len(pred - trait_set)
        unsupported_total += unsupp
        tp = len(pred & true)
        fp = len(pred - true)
        fn = len(true - pred)
        prec = tp / (tp + fp) if (tp + fp) else 1.0 if not pred else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 1.0
        f1 = (2*prec*rec/(prec+rec)) if (prec+rec) else 0.0
        sem_ps.append(prec); sem_rs.append(rec); sem_fs.append(f1)
        if fp > 0:
            hallucinated_rows += 1

    report = {
        "n_eval": len(rows),
        "n_gen_total": len(gen_items),
        "threshold": args.threshold,
        "rougeL_precision_mean": float(sum(rouge_ps)/len(rouge_ps)) if rouge_ps else 0.0,
        "rougeL_recall_mean": float(sum(rouge_rs)/len(rouge_rs)) if rouge_rs else 0.0,
        "rougeL_f1_mean": float(sum(rouge_fs)/len(rouge_fs)) if rouge_fs else 0.0,
        "bleu4": float(bleu),
        "bleu4_smoothed": float(bleu_sm),
        "bleu1": float(bleu1),
        "bleu_precisions_1_4": [float(x) for x in precisions],
        "bleu_brevity_penalty": float(bp),
        "semantic_precision_mean": float(sum(sem_ps)/len(sem_ps)) if sem_ps else 0.0,
        "semantic_recall_mean": float(sum(sem_rs)/len(sem_rs)) if sem_rs else 0.0,
        "semantic_f1_mean": float(sum(sem_fs)/len(sem_fs)) if sem_fs else 0.0,
        "unsupported_concept_count": int(unsupported_total),
        "hallucinated_rows": int(hallucinated_rows),
        "hallucination_rate": float(hallucinated_rows / len(rows)) if rows else 0.0,
        "hallucination_rate_semantic": float(1 - sum(sem_ps)/len(sem_ps)) if sem_ps else 0.0,
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote {out_path}")
    # ponytail: ROUGE/BLEU are token-overlap only on formulaic provider text;
    # upgrade to BERTScore when provider text diversifies


if __name__ == "__main__":
    main()