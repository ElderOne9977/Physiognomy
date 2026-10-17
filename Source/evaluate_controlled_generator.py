# -*- coding: utf-8 -*-
"""Frozen-vector comparison of text generators on the 702-row test split.

Compares, on IDENTICAL frozen 45-d semantic vectors (no re-extraction):
  A. flan_t5_continuous    (frozen checkpoint, greedy; reused precomputed output)
  B. template_calibrated   (deterministic)
  C. template_concise      (deterministic, top-k)
  D. controlled_deterministic (planner + phrase-bank realization)

Metrics: content faithfulness, unsupported concepts, sensitivity to semantic
changes, category-specific responsiveness, structural validity, repeated-output
rate, CPU latency. No ROUGE/BLEU-only optimization, no random-word injection.

FLAN texts are reused from Source/generated_text_test_normal.jsonl when present
(702 rows, deterministic greedy); the SHUFFLED file drives FLAN sensitivity.

Usage:
  python Source/evaluate_controlled_generator.py
  python Source/evaluate_controlled_generator.py --limit 120
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from text_gen_common import (  # noqa: E402
    CATEGORY_ORDER,
    CATEGORY_TITLES,
    load_schema,
    parse_sections,
    validate_structure,
)
from evaluate_text_model import norm_name  # noqa: E402
from generate_text_template import (  # noqa: E402
    DEFAULT_THRESHOLD,
    generate_text,
    generate_text_concise,
    load_thresholds,
)
from generate_text_controlled import (  # noqa: E402
    generate_text_controlled,
    lexicon_collisions,
)
from build_semantic_labels import TRAIT_PATTERNS  # noqa: E402

DEFAULT_TRAIN = ROOT / "text_generation_train.jsonl"
DEFAULT_SPLITS = SRC / "text_generation_splits.json"
DEFAULT_CALIB = SRC / "models_semantic" / "trait_thresholds.json"
FLAN_NORMAL = SRC / "generated_text_test_normal.jsonl"
FLAN_SHUFFLED = SRC / "generated_text_test_shuffled.jsonl"
DEFAULT_OUT = SRC / "controlled_generator_comparison.json"


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
    return {t for t, (_c, pats) in TRAIT_PATTERNS.items()
            if any(re.search(p, text, re.I) for p in pats)}


def prf(pred, true):
    tp, fp, fn = len(pred & true), len(pred - true), len(true - pred)
    p = tp / (tp + fp) if (tp + fp) else 1.0
    r = tp / (tp + fn) if (tp + fn) else 1.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def load_frozen(limit):
    trait_names, cat_of = load_schema()
    splits = json.loads(DEFAULT_SPLITS.read_text(encoding="utf-8"))
    test_set = set(splits["test_images"])
    rows = load_jsonl(DEFAULT_TRAIN)
    test_rows = [r for r in rows if norm_name(r["image_id"]) in test_set]
    test_rows = sorted(test_rows, key=lambda r: norm_name(r["image_id"]))
    assert len(test_rows) == 702, f"expected 702 got {len(test_rows)}"
    return (test_rows[:limit] if limit else test_rows), trait_names, cat_of


def row_support(row, trait_names, tmap):
    probs = {t: float(row["trait_probs"][t]) for t in trait_names}
    sup = {t for t in trait_names if probs[t] >= (tmap.get(t, DEFAULT_THRESHOLD) if tmap else DEFAULT_THRESHOLD)}
    return probs, sup


def evaluate_system(name, texts, rows, trait_names, cat_of, tmap, latency_s=None):
    """Content metrics for one generator over the same frozen rows."""
    n = len(texts)
    prec, rec, unsup, genuine = [], [], [], 0
    valid = 0
    cat_hit = {c: 0 for c in CATEGORY_ORDER}          # own-section concept present when supported
    cat_sup = {c: 0 for c in CATEGORY_ORDER}
    for text, row in zip(texts, rows):
        if validate_structure(text)["valid"]:
            valid += 1
        probs, sup = row_support(row, trait_names, tmap)
        secs, _ = parse_sections(text)
        expressed = extract_concepts(text)
        p, r, _f = prf(expressed, sup)
        prec.append(p)
        rec.append(r)
        unsup.append(len(expressed - sup) / len(expressed) if expressed else 0.0)
        # genuine unsupported = not explainable as a lexicon collision of any
        # selected phrase in the section where it appears
        collisions = set()
        for cat in CATEGORY_ORDER:
            body = secs.get(CATEGORY_TITLES[cat], "")
            sel = {t for t in sup if cat_of[t] == cat}
            coll, _g = lexicon_collisions(body, sel, cat)
            collisions |= coll
        genuine += len((expressed - sup) - collisions)
        # category responsiveness: is a supported concept of this category expressed?
        for cat in CATEGORY_ORDER:
            members = {t for t in trait_names if cat_of[t] == cat}
            if sup & members:
                cat_sup[cat] += 1
                if expressed & members:
                    cat_hit[cat] += 1
    uniq = len(set(texts))
    counts = Counter(texts)
    return {
        "system": name,
        "n": n,
        "structure_valid_rate": valid / n,
        "semantic_precision_mean": mean(prec),
        "semantic_recall_mean": mean(rec),
        "unsupported_rate_mean": mean(unsup),
        "genuine_unsupported_total": genuine,
        "n_unique_texts": uniq,
        "repeated_output_rate": 1.0 - uniq / n,
        "most_common_count": counts.most_common(1)[0][1] if counts else 0,
        "category_responsiveness": {
            c: (cat_hit[c] / cat_sup[c] if cat_sup[c] else None) for c in CATEGORY_ORDER
        },
        "category_supported_rows": {c: cat_sup[c] for c in CATEGORY_ORDER},
        "latency_s_total": latency_s,
        "latency_s_per_row": (latency_s / n) if latency_s is not None else None,
    }


def sensitivity(texts_a, texts_b):
    """Fraction of rows whose text changes when the semantic vector changes."""
    n = len(texts_a)
    changed = sum(1 for a, b in zip(texts_a, texts_b) if a != b)
    sec_changed = {CATEGORY_TITLES[c]: 0 for c in CATEGORY_ORDER}
    for a, b in zip(texts_a, texts_b):
        sa, _ = parse_sections(a)
        sb, _ = parse_sections(b)
        for c in CATEGORY_ORDER:
            t = CATEGORY_TITLES[c]
            if sa.get(t) != sb.get(t):
                sec_changed[t] += 1
    return {
        "changed_rate": changed / n,
        "section_changed_rate": {k: v / n for k, v in sec_changed.items()},
    }


def run(args):
    t0 = time.time()
    rows, trait_names, cat_of = load_frozen(args.limit)
    tmap, strategy = load_thresholds(str(args.calib))
    log(f"rows={len(rows)} thresholds={'yes' if tmap else 'no'} strategy={strategy}")

    probs_list = [{t: float(r["trait_probs"][t]) for t in trait_names} for r in rows]

    # ---- generate (deterministic systems; FLAN reused)
    gen = {}

    t = time.time()
    gen["template_calibrated"] = [generate_text(p, threshold=DEFAULT_THRESHOLD, thresholds=tmap)[0]
                                  for p in probs_list]
    lat_cal = time.time() - t

    t = time.time()
    gen["template_concise"] = [generate_text_concise(p, threshold=DEFAULT_THRESHOLD, thresholds=tmap)[0]
                               for p in probs_list]
    lat_con = time.time() - t

    t = time.time()
    gen["controlled_deterministic"] = [generate_text_controlled(
        p, thresholds=tmap, threshold=DEFAULT_THRESHOLD, realizer="deterministic")[0]
        for p in probs_list]
    lat_ctl = time.time() - t

    # FLAN reuse (precomputed greedy output on the same frozen rows)
    flan_rows = None
    if FLAN_NORMAL.exists():
        flan_rows = load_jsonl(FLAN_NORMAL)
        flan_by_id = {norm_name(r["image_id"]): r["generated_text"] for r in flan_rows}
        ids = [norm_name(r["image_id"]) for r in rows]
        if all(i in flan_by_id for i in ids):
            gen["flan_t5_continuous"] = [flan_by_id[i] for i in ids]
        else:
            log("FLAN reuse incomplete; skipping flan_t5_continuous")

    # ---- evaluate
    results = {}
    lat = {"template_calibrated": lat_cal, "template_concise": lat_con,
           "controlled_deterministic": lat_ctl, "flan_t5_continuous": None}
    for name, texts in gen.items():
        results[name] = evaluate_system(name, texts, rows, trait_names, cat_of, tmap,
                                        latency_s=lat.get(name))
        log(f"  {name}: valid {results[name]['structure_valid_rate']:.3f} "
            f"prec {results[name]['semantic_precision_mean']:.3f} "
            f"rec {results[name]['semantic_recall_mean']:.3f} "
            f"uniq {results[name]['n_unique_texts']}")

    # ---- sensitivity to semantic change (counterfactual: flip vector to train mean)
    # Deterministic systems are re-run on a counterfactual vector; FLAN uses the
    # precomputed SHUFFLED condition (different vectors, same rows).
    import numpy as np
    mean_vec = np.mean(np.array([[r["trait_probs"][t] for t in trait_names] for r in rows]), axis=0)
    cf_probs = [{t: float(mean_vec[j]) for j, t in enumerate(trait_names)} for _ in rows]
    sens = {}
    for name in ("template_calibrated", "template_concise", "controlled_deterministic"):
        if name == "template_calibrated":
            cf = [generate_text(p, threshold=DEFAULT_THRESHOLD, thresholds=tmap)[0] for p in cf_probs]
        elif name == "template_concise":
            cf = [generate_text_concise(p, threshold=DEFAULT_THRESHOLD, thresholds=tmap)[0] for p in cf_probs]
        else:
            cf = [generate_text_controlled(p, thresholds=tmap, threshold=DEFAULT_THRESHOLD)[0]
                  for p in cf_probs]
        sens[name] = sensitivity(gen[name], cf)
    if FLAN_SHUFFLED.exists() and "flan_t5_continuous" in gen:
        flan_sh = load_jsonl(FLAN_SHUFFLED)
        sh_by_id = {norm_name(r["image_id"]): r["generated_text"] for r in flan_sh}
        ids = [norm_name(r["image_id"]) for r in rows]
        if all(i in sh_by_id for i in ids):
            sens["flan_t5_continuous"] = sensitivity(
                gen["flan_t5_continuous"], [sh_by_id[i] for i in ids])

    report = {
        "n_rows": len(rows),
        "threshold_strategy": strategy,
        "vectors": "identical frozen 45-d test vectors (no re-extraction)",
        "flan_source": str(FLAN_NORMAL) if FLAN_NORMAL.exists() else None,
        "flan_sensitivity_source": str(FLAN_SHUFFLED) if FLAN_SHUFFLED.exists() else None,
        "systems": results,
        "sensitivity_to_semantic_change": sens,
        "meta": {"elapsed_s": round(time.time() - t0, 1),
                 "latency_definition": "wall-clock generation only (FLAN reused, not measured)"},
    }
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"wrote {args.out}")
    print_summary(report)
    return report


def print_summary(report):
    print("\n=== frozen-vector generator comparison ===")
    hdr = f"{'system':28s} {'valid':>6s} {'prec':>6s} {'rec':>6s} {'unsup':>6s} {'uniq':>6s} {'rep':>6s} {'ms/row':>8s}"
    print(hdr)
    for name, r in report["systems"].items():
        ms = (r["latency_s_per_row"] * 1000) if r["latency_s_per_row"] is not None else float("nan")
        print(f"{name:28s} {r['structure_valid_rate']:6.3f} {r['semantic_precision_mean']:6.3f} "
              f"{r['semantic_recall_mean']:6.3f} {r['unsupported_rate_mean']:6.3f} "
              f"{r['n_unique_texts']:6d} {r['repeated_output_rate']:6.3f} {ms:8.3f}")
    print("\n=== sensitivity to semantic change (changed_rate) ===")
    for name, s in report["sensitivity_to_semantic_change"].items():
        print(f"  {name:28s} changed {s['changed_rate']:.3f}")
    print("\n=== category responsiveness (supported concept expressed) ===")
    for name, r in report["systems"].items():
        print(f"  {name:28s} " + " ".join(
            f"{c[:4]}={(v if v is not None else float('nan')):.2f}"
            for c, v in r["category_responsiveness"].items()))


def main():
    ap = argparse.ArgumentParser(description="Frozen-vector generator comparison")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--calib", default=str(DEFAULT_CALIB))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()