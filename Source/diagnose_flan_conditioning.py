# -*- coding: utf-8 -*-
"""Focused counterfactual conditioning diagnosis for the FROZEN FLAN-T5 checkpoint.

Question: does the trained checkpoint actually use the 45-d semantic vector, and
can it follow an explicit content plan? Deterministic greedy decoding only.

Conditions (frozen prompt format from text_gen_common.format_input unless stated):
  ORIGINAL        : the row's own 45-d vector
  SHUFFLED        : another row's vector (seeded derangement, no self)
  CONSTANT_MEAN   : train-split mean vector for every row
  SINGLE_TRAIT    : flip one trait low->high (0.05 -> 0.95) on top of the row
  CATEGORY_MASK   : replace one category's traits with the train-split mean
  PLAN            : explicit content-plan prompt (selected/empty per category)

Metrics (documented, no tuning on outcomes):
  changed_section_rate    : per-section body differs from ORIGINAL
  mean_rougeL_vs_original : output similarity to ORIGINAL
  semantic_precision      : |expressed & supported| / |expressed|
  semantic_recall         : |expressed & supported| / |supported|
  unsupported_rate        : |expressed - supported| / |expressed|
  trait_responsiveness    : SINGLE_TRAIT own-section changed fraction
  category_responsiveness : CATEGORY_MASK own-section changed fraction
  plan_precision/recall   : expressed concepts vs planned concepts (PLAN cond)

Frozen, never modified by this script: model dir, tokenizer, splits, prompt
instruction, greedy decoding (max_new_tokens=256, do_sample=False, num_beams=1),
schema, OOF probabilities. No training, no retraining.

Usage:
  python Source/diagnose_flan_conditioning.py --n 12
  python Source/diagnose_flan_conditioning.py --n 12 --skip-plan
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from text_gen_common import (  # noqa: E402
    CATEGORY_ORDER,
    CATEGORY_TITLES,
    format_input,
    load_schema,
    parse_sections,
    validate_structure,
)
from evaluate_text_model import norm_name, rouge_l_one  # noqa: E402
from build_semantic_labels import TRAIT_PATTERNS  # noqa: E402

DEFAULT_MODEL = SRC / "models_text" / "flan_t5_small"
DEFAULT_TRAIN = ROOT / "text_generation_train.jsonl"
DEFAULT_SPLITS = SRC / "text_generation_splits.json"
DEFAULT_OUT = SRC / "flan_conditioning_counterfactual.json"
DEFAULT_PLAN_OUT = SRC / "flan_content_plan_probe.json"

SEED = 42
BS = 16
MAX_SOURCE_LEN = 320
MAX_NEW_TOKENS = 256

# Representative single traits: one per category (schema names).
SINGLE_TRAIT_PROBES = ["warm", "leadership", "loyal", "steady_growth", "financial_stability"]
FLIP_LOW = 0.05
FLIP_HIGH = 0.95

# Deterministic interpretation thresholds (stored in the report).
THRESH = {
    "responsive_section_changed": 0.50,
    "weak_section_changed": 0.10,
    "responsive_rouge_drop": 0.05,
    "plan_precision_usable": 0.80,
    "plan_recall_usable": 0.50,
}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def mean(xs):
    return float(sum(xs) / len(xs)) if xs else 0.0


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_concepts(text):
    """Canonical lexicon extraction (same regexes that built semantic_labels.csv)."""
    return {t for t, (_cat, pats) in TRAIT_PATTERNS.items()
            if any(re.search(p, text, re.I) for p in pats)}


def load_frozen(n):
    """First n frozen test rows in canonical (norm_name) order."""
    trait_names, cat_of = load_schema()
    splits = json.loads(DEFAULT_SPLITS.read_text(encoding="utf-8"))
    test_set = set(splits["test_images"])
    train_set = set(splits["train_images"])
    assert not (train_set & test_set), "train/test overlap"
    rows = load_jsonl(DEFAULT_TRAIN)
    test_rows = [r for r in rows if norm_name(r["image_id"]) in test_set]
    test_rows = sorted(test_rows, key=lambda r: norm_name(r["image_id"]))
    assert len(test_rows) == 702, f"expected 702 test rows got {len(test_rows)}"
    for r in test_rows:
        assert len(r["trait_probs"]) == 45
        assert list(r["trait_probs"].keys()) == trait_names
    return test_rows[:n], trait_names, cat_of, rows, train_set


def train_mean(rows, train_set, trait_names):
    tr = [r for r in rows if norm_name(r["image_id"]) in train_set]
    mat = np.array([[float(r["trait_probs"][t]) for t in trait_names] for r in tr])
    return mat.mean(axis=0)


def vec_of(row, trait_names):
    return [float(row["trait_probs"][t]) for t in trait_names]


def probs_of(vec, trait_names):
    return {t: float(v) for t, v in zip(trait_names, vec)}


def supported_set(probs, trait_names, thr=0.5):
    return {t for t in trait_names if probs[t] >= thr}


def prf(pred, true):
    tp, fp, fn = len(pred & true), len(pred - true), len(true - pred)
    p = tp / (tp + fp) if (tp + fp) else 1.0
    r = tp / (tp + fn) if (tp + fn) else 1.0
    return p, r


def section_body(text, title):
    return parse_sections(text)[0].get(title, "")


def gen_texts(model, tok, device, prompts):
    import torch
    texts = []
    with torch.no_grad():
        for i in range(0, len(prompts), BS):
            chunk = prompts[i:i + BS]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=MAX_SOURCE_LEN)
            gen = model.generate(**{k: v.to(device) for k, v in enc.items()},
                                 max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1)
            texts.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return texts


def build_prompts(vecs, trait_names, cat_of):
    return [format_input(probs_of(v, trait_names), trait_names, cat_of,
                         representation="continuous") for v in vecs]


# ------------------------------------------------------------------ plan probe

PLAN_HEADER = (
    "Realize the following content plan as a five-section English face-reading "
    "description. Express ONLY the listed concepts in each section. If a section "
    "has no listed concepts, write a short neutral sentence."
)


def build_plan_prompt(plan, cat_of):
    lines = [PLAN_HEADER, ""]
    for cat in CATEGORY_ORDER:
        concepts = plan[cat]["selected"]
        lines.append(f"{CATEGORY_TITLES[cat]}: " + (", ".join(concepts) if concepts else "(none)"))
    lines.append("")
    lines.append("Return exactly these five sections:")
    lines.extend(CATEGORY_TITLES[c] for c in CATEGORY_ORDER)
    return "\n".join(lines)


def build_plan(vec, trait_names, cat_of, thresholds=None, k=3):
    """Deterministic content plan: top-k supported concepts per category by margin."""
    probs = probs_of(vec, trait_names)
    plan = {}
    for cat in CATEGORY_ORDER:
        members = [t for t in trait_names if cat_of[t] == cat]
        active = [t for t in members
                  if probs[t] >= (thresholds.get(t, 0.5) if thresholds else 0.5)]
        active.sort(key=lambda t: (-(probs[t] - (thresholds.get(t, 0.5) if thresholds else 0.5)),
                                   -probs[t], t))
        selected = active[:k]
        plan[cat] = {
            "selected": selected,
            "omitted": [t for t in active if t not in selected],
            "empty": len(selected) == 0,
        }
    return plan


# --------------------------------------------------------------------- runner

def run(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    t0 = time.time()
    test_rows, trait_names, cat_of, all_rows, train_set = load_frozen(args.n)
    tmean = train_mean(all_rows, train_set, trait_names)
    log(f"loaded {len(test_rows)} frozen test rows; mean vector from train split")

    base_vecs = [vec_of(r, trait_names) for r in test_rows]

    # seeded derangement for SHUFFLED (no row keeps its own vector)
    rng = np.random.RandomState(SEED)
    idx = np.arange(len(test_rows))
    perm = None
    for _ in range(10000):
        cand = rng.permutation(len(test_rows))
        if not (cand == idx).any():
            perm = cand
            break
    shuffled_vecs = [base_vecs[int(perm[i])] for i in range(len(test_rows))]

    conditions = {
        "ORIGINAL": base_vecs,
        "SHUFFLED": shuffled_vecs,
        "CONSTANT_MEAN": [[float(v) for v in tmean] for _ in test_rows],
    }
    for t in SINGLE_TRAIT_PROBES:
        j = trait_names.index(t)
        vecs = []
        for v in base_vecs:
            nv = list(v)
            nv[j] = FLIP_HIGH
            vecs.append(nv)
        conditions[f"SINGLE_TRAIT::{t}"] = vecs
    for cat in CATEGORY_ORDER:
        cols = [j for j, t in enumerate(trait_names) if cat_of[t] == cat]
        vecs = []
        for v in base_vecs:
            nv = list(v)
            for j in cols:
                nv[j] = float(tmean[j])
            vecs.append(nv)
        conditions[f"CATEGORY_MASK::{cat}"] = vecs

    log(f"loading model {args.model}")
    tok = AutoTokenizer.from_pretrained(str(args.model))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(args.model))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    log(f"cuda_available={torch.cuda.is_available()} device={device}")

    texts = {}
    for name, vecs in conditions.items():
        prompts = build_prompts(vecs, trait_names, cat_of)
        tt = time.time()
        texts[name] = gen_texts(model, tok, device, prompts)
        log(f"  {name}: {len(prompts)} prompts in {time.time() - tt:.1f}s")

    plan_result = None
    if not args.skip_plan:
        plans = [build_plan(v, trait_names, cat_of) for v in base_vecs]
        plan_prompts = [build_plan_prompt(p, cat_of) for p in plans]
        tt = time.time()
        plan_texts = gen_texts(model, tok, device, plan_prompts)
        log(f"  PLAN: {len(plan_prompts)} prompts in {time.time() - tt:.1f}s")
        plan_result = evaluate_plan(plans, plan_texts)

    report = evaluate(texts, test_rows, base_vecs, trait_names, cat_of)
    report["meta"] = {
        "model_dir": str(args.model),
        "n_rows": len(test_rows),
        "seed": SEED,
        "batch_size": BS,
        "max_source_len": MAX_SOURCE_LEN,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "num_beams": 1,
        "cuda_available": bool(torch.cuda.is_available()),
        "single_trait_probes": SINGLE_TRAIT_PROBES,
        "flip_low": FLIP_LOW,
        "flip_high": FLIP_HIGH,
        "thresholds": THRESH,
        "elapsed_s": round(time.time() - t0, 1),
    }
    if plan_result is not None:
        report["content_plan_probe"] = plan_result

    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"wrote {args.out}")
    if plan_result is not None:
        Path(args.plan_out).write_text(
            json.dumps({"meta": report["meta"], **plan_result}, indent=2, ensure_ascii=False),
            encoding="utf-8")
        log(f"wrote {args.plan_out}")
    print_summary(report)
    return report


def evaluate(texts, test_rows, base_vecs, trait_names, cat_of):
    """Compute conditioning metrics for every non-ORIGINAL condition."""
    orig = texts["ORIGINAL"]
    n = len(orig)
    sections = [CATEGORY_TITLES[c] for c in CATEGORY_ORDER]

    orig_concepts = [extract_concepts(t) for t in orig]
    orig_supported = [supported_set(probs_of(v, trait_names), trait_names) for v in base_vecs]
    base_prec, base_rec, base_unsup = [], [], []
    for c, s in zip(orig_concepts, orig_supported):
        p, r = prf(c, s)
        base_prec.append(p)
        base_rec.append(r)
        base_unsup.append(len(c - s) / len(c) if c else 0.0)

    valid = sum(1 for t in orig if validate_structure(t)["valid"])
    uniq = len({t for t in orig})

    out = {
        "original": {
            "n": n,
            "n_unique_texts": uniq,
            "structure_valid_rate": valid / n,
            "semantic_precision_mean": mean(base_prec),
            "semantic_recall_mean": mean(base_rec),
            "unsupported_rate_mean": mean(base_unsup),
        },
        "conditions": {},
    }

    for name, ts in texts.items():
        if name == "ORIGINAL":
            continue
        sec_changed = {s: 0 for s in sections}
        rouge = []
        for a, b in zip(orig, ts):
            _, _, f = rouge_l_one(b, a)
            rouge.append(f)
            for s in sections:
                if section_body(a, s) != section_body(b, s):
                    sec_changed[s] += 1
        prec, rec, unsup = [], [], []
        for c, s in zip((extract_concepts(t) for t in ts), orig_supported):
            p, r = prf(c, s)
            prec.append(p)
            rec.append(r)
            unsup.append(len(c - s) / len(c) if c else 0.0)
        out["conditions"][name] = {
            "changed_section_rate": {s: sec_changed[s] / n for s in sections},
            "changed_any_section_rate": mean([1.0 if any(section_body(a, s) != section_body(b, s)
                                                        for s in sections) else 0.0
                                              for a, b in zip(orig, ts)]),
            "mean_rougeL_f1_vs_original": mean(rouge),
            "semantic_precision_mean": mean(prec),
            "semantic_recall_mean": mean(rec),
            "unsupported_rate_mean": mean(unsup),
            "n_unique_texts": len({t for t in ts}),
        }

    trait_resp = {}
    for name in list(out["conditions"]):
        if not name.startswith("SINGLE_TRAIT::"):
            continue
        trait = name.split("::", 1)[1]
        cat = next(c for c in CATEGORY_ORDER if trait in [t for t in trait_names if cat_of[t] == c])
        title = CATEGORY_TITLES[cat]
        cond = out["conditions"][name]
        trait_resp[trait] = {
            "category": cat,
            "own_section": title,
            "own_section_changed_rate": cond["changed_section_rate"][title],
            "mean_rougeL_f1_vs_original": cond["mean_rougeL_f1_vs_original"],
            "responsive": cond["changed_section_rate"][title] >= THRESH["responsive_section_changed"],
        }
    cat_resp = {}
    for name in list(out["conditions"]):
        if not name.startswith("CATEGORY_MASK::"):
            continue
        cat = name.split("::", 1)[1]
        title = CATEGORY_TITLES[cat]
        cond = out["conditions"][name]
        cat_resp[cat] = {
            "own_section": title,
            "own_section_changed_rate": cond["changed_section_rate"][title],
            "mean_rougeL_f1_vs_original": cond["mean_rougeL_f1_vs_original"],
            "responsive": cond["changed_section_rate"][title] >= THRESH["responsive_section_changed"],
        }
    out["trait_responsiveness"] = trait_resp
    out["category_responsiveness"] = cat_resp
    out["summary"] = {
        "n_traits_responsive": sum(1 for v in trait_resp.values() if v["responsive"]),
        "n_traits_tested": len(trait_resp),
        "n_categories_responsive": sum(1 for v in cat_resp.values() if v["responsive"]),
        "n_categories_tested": len(cat_resp),
        "shuffled_rouge_vs_original": out["conditions"]["SHUFFLED"]["mean_rougeL_f1_vs_original"],
        "constant_mean_rouge_vs_original":
            out["conditions"]["CONSTANT_MEAN"]["mean_rougeL_f1_vs_original"],
    }
    return out


def evaluate_plan(plans, plan_texts):
    """Did the checkpoint realize exactly the planned concepts?"""
    prec, rec, exact, valid, n_extra = [], [], 0, 0, 0
    per_cat = {c: {"p": [], "r": []} for c in CATEGORY_ORDER}
    for plan, text in zip(plans, plan_texts):
        if validate_structure(text)["valid"]:
            valid += 1
        planned_all, expressed_all = set(), set()
        for cat in CATEGORY_ORDER:
            title = CATEGORY_TITLES[cat]
            planned = set(plan[cat]["selected"])
            expressed = extract_concepts(section_body(text, title)) & set(TRAIT_PATTERNS)
            planned_all |= planned
            expressed_all |= expressed
            p, r = prf(expressed, planned)
            per_cat[cat]["p"].append(p)
            per_cat[cat]["r"].append(r)
        p, r = prf(expressed_all, planned_all)
        prec.append(p)
        rec.append(r)
        n_extra += len(expressed_all - planned_all)
        if expressed_all == planned_all:
            exact += 1
    n = len(plan_texts)
    return {
        "n": n,
        "structure_valid_rate": valid / n,
        "plan_precision_mean": mean(prec),
        "plan_recall_mean": mean(rec),
        "exact_plan_match_rate": exact / n,
        "mean_extra_concepts_per_row": n_extra / n,
        "per_category": {CATEGORY_TITLES[c]: {"precision_mean": mean(v["p"]),
                                              "recall_mean": mean(v["r"])}
                         for c, v in per_cat.items()},
        "usable": (mean(prec) >= THRESH["plan_precision_usable"]
                   and mean(rec) >= THRESH["plan_recall_usable"]),
    }


def print_summary(report):
    o = report["original"]
    print("\n=== ORIGINAL ===")
    print(f"  unique texts {o['n_unique_texts']}/{o['n']}  structure_valid {o['structure_valid_rate']:.2f}")
    print(f"  semantic precision {o['semantic_precision_mean']:.3f} recall {o['semantic_recall_mean']:.3f} "
          f"unsupported {o['unsupported_rate_mean']:.3f}")
    print("\n=== conditions (vs ORIGINAL) ===")
    for name, c in report["conditions"].items():
        print(f"  {name:28s} rougeL {c['mean_rougeL_f1_vs_original']:.4f}  "
              f"changed_any {c['changed_any_section_rate']:.3f}  uniq {c['n_unique_texts']}")
    print("\n=== trait responsiveness (own section) ===")
    for t, v in report["trait_responsiveness"].items():
        print(f"  {t:22s} {v['own_section']:16s} changed {v['own_section_changed_rate']:.3f} "
              f"resp={v['responsive']}")
    print("\n=== category responsiveness (own section) ===")
    for c, v in report["category_responsiveness"].items():
        print(f"  {c:16s} {v['own_section']:16s} changed {v['own_section_changed_rate']:.3f} "
              f"resp={v['responsive']}")
    if "content_plan_probe" in report:
        p = report["content_plan_probe"]
        print("\n=== content-plan probe ===")
        print(f"  valid {p['structure_valid_rate']:.2f}  plan_precision {p['plan_precision_mean']:.3f}  "
              f"plan_recall {p['plan_recall_mean']:.3f}  exact {p['exact_plan_match_rate']:.3f}  "
              f"extra/row {p['mean_extra_concepts_per_row']:.2f}  usable={p['usable']}")
    print(f"\n  summary {report['summary']}")


def main():
    ap = argparse.ArgumentParser(description="Frozen FLAN-T5 counterfactual conditioning diagnosis")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--n", type=int, default=12, help="rows from the frozen test split")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--plan-out", default=str(DEFAULT_PLAN_OUT))
    ap.add_argument("--skip-plan", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()