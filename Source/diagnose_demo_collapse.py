# -*- coding: utf-8 -*-
"""Demo output-collapse diagnostics (Stages A-G). No training, no re-extraction.

Stages:
  A  image/feature/semantic duplication audit        (part of every report)
  B  semantic vector diversity                       -> Source/demo_semantic_diversity.json
  C  text output diversity                           -> Source/demo_text_diversity.json
  D  semantic distance vs text change                -> Source/demo_collapse_relationship.json
  E  template vs FLAN-T5 on identical inputs         -> Source/demo_generator_comparison.json
  F  semantic contradiction analysis                 -> Source/demo_semantic_contradictions.json
  G  provider corpus diversity + conditioning        -> Source/provider_text_diversity_report.json

Usage:
  python Source/diagnose_demo_collapse.py
  python Source/diagnose_demo_collapse.py --result-dir outputs --result-dir outputs/demo_tmp
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from text_gen_common import (  # noqa: E402
    CATEGORY_ORDER,
    CATEGORY_TITLES,
    load_schema,
    parse_sections,
)
from evaluate_text_model import norm_name, rouge_l_one  # noqa: E402
from generate_text_template import (  # noqa: E402
    generate_text as template_generate,
    load_thresholds as template_load_thresholds,
    generate_text_concise,
)
from build_semantic_labels import TRAIT_PATTERNS, match_trait  # noqa: E402

CALIB_PATH = SRC / "models_semantic" / "trait_thresholds.json"
TRAIN_JSONL = ROOT / "text_generation_train.jsonl"
INSTRUCT_JSONL = ROOT / "text_generation_instruct.jsonl"
ENRICHED_JSONL = ROOT / "labels_enriched.jsonl"
OOF_CSV = ROOT / "oof_semantic_probs.csv"
SPLITS = SRC / "text_generation_splits.json"

# Diagnostic constants (documented, not tuned on outcomes).
STRONG_UNSUPPORTED_PROB = 0.2   # CLI-configurable
VECT_MAD_COLLAPSE = 0.05        # distinct images + near-identical vectors => upstream collapse
TEXT_SIM_COLLAPSE = 0.90        # vectors differ but text similarity above this => generator collapse
TRAIT_BIG_DIFF = 0.3


def fin(x):
    """Serialize only finite floats (no NaN/Infinity in JSON)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def sha256_file(path: Path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def pixel_fingerprints(path: Path):
    """(full pixel hash, 32x32 gray thumbnail hash) or (None, None)."""
    try:
        from PIL import Image
    except ImportError:
        return None, None
    try:
        with Image.open(path) as im:
            rgb = im.convert("RGB")
            w, h = rgb.size
            full = hashlib.sha256(f"{w}x{h}|".encode() + rgb.tobytes()).hexdigest()
            thumb = rgb.convert("L").resize((32, 32))
            th = hashlib.sha256(thumb.tobytes()).hexdigest()
        return full, th
    except Exception:
        return None, None


def cosine(a, b):
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    return fin(num / (da * db)) if da and db else None


def euclid(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def mean(xs):
    xs = [x for x in xs if x is not None]
    return fin(sum(xs) / len(xs)) if xs else None


def norm_text(s: str) -> str:
    """Whitespace/case normalization only - keeps lexical differences."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def split_sentences(s: str):
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+", s or "") if x.strip()]


def pearson(xs, ys):
    xs, ys = list(xs), list(ys)
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return fin(num / (dx * dy)) if dx and dy else None


def _ranks(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs, ys):
    if len(xs) < 3:
        return None
    return pearson(_ranks(list(xs)), _ranks(list(ys)))


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)
    path.write_text(txt, encoding="utf-8")
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# Stage A: load demo result files + duplication audit
# ---------------------------------------------------------------------------

def is_mock_probs(probs: dict) -> bool:
    if not probs:
        return True
    if all(re.fullmatch(r"t\d+", k) for k in probs):
        return True
    return "warm" not in probs


def load_results(result_dirs) -> list:
    files = []
    for d in result_dirs:
        p = Path(d)
        if not p.exists():
            print(f"skip missing result dir: {p}")
            continue
        files.extend(sorted(p.glob("*result*.json")))
    seen, recs = set(), []
    schema_names, _ = load_schema()
    for f in files:
        if f in seen:
            continue
        seen.add(f)
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"unreadable {f}: {e}")
            continue
        if not isinstance(d, dict) or "text" not in d:
            continue
        probs = (d.get("semantic") or {}).get("probabilities") or {}
        mock = is_mock_probs(probs)
        img = d.get("image")
        img_path = Path(img) if img else None
        byte_sha = pixel_sha = thumb_sha = None
        img_exists = bool(img_path and img_path.exists())
        if img_exists:
            byte_sha = sha256_file(img_path)
            pixel_sha, thumb_sha = pixel_fingerprints(img_path)
        csv_path = (d.get("features") or {}).get("csv_path")
        vec = [fin(probs.get(t)) for t in schema_names] if not mock else None
        recs.append({
            "result_file": str(f),
            "result_id": f.stem.replace("_result", ""),
            "image_path": img,
            "image_exists": img_exists,
            "image_id": Path(img).name if img else None,
            "image_sha256": byte_sha,
            "image_pixel_sha256": pixel_sha,
            "image_thumb_sha256": thumb_sha,
            "feature_csv": csv_path,
            "feature_source": d.get("feature_source"),
            "face": d.get("face") or {},
            "qc_warnings": (d.get("face") or {}).get("warnings") or [],
            "trait_count": len(probs),
            "vector": vec,
            "text": d.get("text") or "",
            "predictions": d.get("predictions") or {},
            "is_mock": mock,
        })
    return recs


def feature_row_stats(csv_path):
    """Numeric feature fingerprint excluding id/timing columns."""
    if not csv_path or not Path(csv_path).exists():
        return None
    try:
        import pandas as pd
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception:
        return None
    if df.empty:
        return None
    row = df.iloc[0]
    drop = [c for c in df.columns
            if c.lower() in ("image", "image_id") or "time" in c.lower()]
    vals = pd.to_numeric(row.drop(labels=drop, errors="ignore"), errors="coerce")
    vals = vals.dropna()
    h = hashlib.sha256(
        "|".join(f"{c}={vals[c]:.9g}" for c in sorted(vals.index)).encode()
    ).hexdigest()
    return {"n_numeric": int(len(vals)), "values_sha256": h}


def stage_a(recs) -> dict:
    real = [r for r in recs if not r["is_mock"]]
    by_byte, by_pixel, by_thumb = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in real:
        if r["image_sha256"]:
            by_byte[r["image_sha256"]].append(r["result_id"])
        if r["image_pixel_sha256"]:
            by_pixel[r["image_pixel_sha256"]].append(r["result_id"])
        if r["image_thumb_sha256"]:
            by_thumb[r["image_thumb_sha256"]].append(r["result_id"])
    byte_groups = [v for v in by_byte.values() if len(v) > 1]
    pixel_groups = [v for v in by_pixel.values() if len(v) > 1]
    for r in real:
        r["feature_stats"] = feature_row_stats(r["feature_csv"])
    feat_by_hash = defaultdict(list)
    for r in real:
        st = r.get("feature_stats")
        if st:
            feat_by_hash[st["values_sha256"]].append(r["result_id"])
    feat_groups = [v for v in feat_by_hash.values() if len(v) > 1]
    vec_groups = defaultdict(list)
    for r in real:
        if r["vector"]:
            key = json.dumps([round(v, 12) if v is not None else None
                              for v in r["vector"]])
            vec_groups[key].append(r["result_id"])
    identical_vec_groups = [v for v in vec_groups.values() if len(v) > 1]
    return {
        "files_total": len(recs),
        "files_mock_excluded": sum(1 for r in recs if r["is_mock"]),
        "files_real": len(real),
        "unique_byte_hashes": len(by_byte),
        "unique_pixel_hashes": len(by_pixel),
        "unique_thumb_hashes": len(by_thumb),
        "byte_duplicate_groups": byte_groups,
        "pixel_duplicate_groups": pixel_groups,
        "missing_image_files": [r["result_id"] for r in real if not r["image_exists"]],
        "feature_csvs_compared": sum(1 for r in real if r.get("feature_stats")),
        "feature_duplicate_groups": feat_groups,
        "identical_vector_groups": identical_vec_groups,
        "same_image_diff_filename": [
            {"group": v,
             "filenames": sorted({Path(r["image_path"]).name for r in real
                                  if r["result_id"] in v and r["image_path"]})}
            for v in byte_groups
        ],
    }


# ---------------------------------------------------------------------------
# Stage B: semantic vector diversity
# ---------------------------------------------------------------------------

def stage_b(recs, schema_names, cat_of, tmap) -> dict:
    real = [r for r in recs if not r["is_mock"] and r["vector"]]
    order_ok = all(r["trait_count"] == len(schema_names) for r in real)
    pairs, cross_counts, big_examples, identical_pairs = [], [], [], []
    for i in range(len(real)):
        for j in range(i + 1, len(real)):
            a, b = real[i], real[j]
            va, vb = a["vector"], b["vector"]
            diffs = [abs((x or 0) - (y or 0)) for x, y in zip(va, vb)]
            if all(d == 0 for d in diffs):
                identical_pairs.append([a["result_id"], b["result_id"]])
                continue
            cross05 = sum(1 for x, y in zip(va, vb) if (x >= 0.5) != (y >= 0.5))
            cross_cal = sum(
                1 for t, x, y in zip(schema_names, va, vb)
                if tmap and (x >= tmap.get(t, 0.5)) != (y >= tmap.get(t, 0.5))
            )
            per_cat = {}
            for c in CATEGORY_ORDER:
                idx = [k for k, t in enumerate(schema_names) if cat_of.get(t) == c]
                per_cat[c] = fin(sum(diffs[k] for k in idx) / len(idx)) if idx else None
            big = sorted(
                (
                    {"trait": t, "a": fin(va[k]), "b": fin(vb[k]),
                     "abs_diff": fin(diffs[k])}
                    for k, t in enumerate(schema_names) if diffs[k] > TRAIT_BIG_DIFF
                ),
                key=lambda d: -(d["abs_diff"] or 0),
            )
            big_examples.append({
                "pair": [a["result_id"], b["result_id"]],
                "traits_differing_gt_0.3": len(big),
                "top_differences": big[:8],
            })
            cross_counts.append(cross05)
            pairs.append({
                "image_a": a["result_id"],
                "image_b": b["result_id"],
                "mad": fin(sum(diffs) / len(diffs)),
                "max_abs": fin(max(diffs)),
                "euclidean": fin(euclid(va, vb)),
                "cosine_similarity": cosine(va, vb),
                "traits_gt_0.1": sum(1 for d in diffs if d > 0.1),
                "traits_gt_0.3": sum(1 for d in diffs if d > TRAIT_BIG_DIFF),
                "traits_crossing_0.5": cross05,
                "traits_crossing_calibrated": cross_cal,
                "per_category_mad": per_cat,
            })
    vec_groups = defaultdict(list)
    for r in real:
        key = json.dumps([round(v, 12) if v is not None else None for v in r["vector"]])
        vec_groups[key].append(r["result_id"])
    identical_vec_groups = [v for v in vec_groups.values() if len(v) > 1]
    mads = [p["mad"] for p in pairs if p["mad"] is not None]
    maxs = [p["max_abs"] for p in pairs if p["max_abs"] is not None]
    distinct = len(vec_groups)
    return {
        "total_files": len(recs),
        "real_files_analyzed": len(real),
        "distinct_image_hashes": len({r["image_sha256"] for r in real if r["image_sha256"]}),
        "distinct_semantic_vectors": distinct,
        "duplicate_groups": identical_vec_groups,
        "identical_vector_pairs": identical_pairs,
        "trait_order": schema_names,
        "trait_order_consistent": order_ok,
        "duplicate_adjusted": {
            "pairs": len(pairs),
            "mean_mad": mean(mads),
            "max_mad": fin(max(mads)) if mads else None,
            "max_abs_overall": fin(max(maxs)) if maxs else None,
            "mean_traits_crossing_0.5": mean(cross_counts),
        },
        "pairwise": pairs,
        "per_category_examples": big_examples[:10],
        "evidence_of_semantic_collapse": bool(
            mads and mean(mads) is not None and mean(mads) < VECT_MAD_COLLAPSE
        ),
    }


# ---------------------------------------------------------------------------
# Stage C: text output diversity
# ---------------------------------------------------------------------------

def stage_c(recs, schema_names, cat_of) -> dict:
    real = [r for r in recs if not r["is_mock"]]
    texts = [r["text"] for r in real]
    normed = [norm_text(t) for t in texts]
    sec_maps = []
    sec_unique = {}
    for r in real:
        m, _ = parse_sections(r["text"])
        if not m:
            m = {CATEGORY_TITLES[c]: (r["predictions"] or {}).get(c, "")
                 for c in CATEGORY_ORDER}
        sec_maps.append(m)
    for c in CATEGORY_ORDER:
        title = CATEGORY_TITLES[c]
        sec_unique[title] = len({norm_text(m.get(title, "")) for m in sec_maps})
    pairs = []
    for i in range(len(real)):
        for j in range(i + 1, len(real)):
            a, b = real[i], real[j]
            p, rc, f = rouge_l_one(a["text"], b["text"])
            pairs.append({
                "image_a": a["result_id"], "image_b": b["result_id"],
                "rouge_l_precision": fin(p), "rouge_l_recall": fin(rc),
                "rouge_l_f1": fin(f),
                "exact_equal": a["text"] == b["text"],
                "normalized_equal": norm_text(a["text"]) == norm_text(b["text"]),
            })
    repeated = {}
    for c in CATEGORY_ORDER:
        title = CATEGORY_TITLES[c]
        sents = [s for m in sec_maps for s in split_sentences(m.get(title, ""))]
        cnt = Counter(norm_text(s) for s in sents if s.strip())
        repeated[title] = [{"sentence": s, "count": n}
                           for s, n in cnt.most_common(8) if n > 1]
    all_sents = Counter(norm_text(s) for t in texts for s in split_sentences(t))
    cross = [p for p in pairs if not p["exact_equal"] and p["normalized_equal"]]
    lens = [len(t.split()) for t in texts]
    n = len(real)
    uniq_exact = len(set(texts))
    uniq_norm = len(set(normed))
    f1s = [p["rouge_l_f1"] for p in pairs if p["rouge_l_f1"] is not None]
    return {
        "n_outputs": n,
        "unique_full_texts": uniq_exact,
        "unique_normalized_texts": uniq_norm,
        "exact_duplicate_rate": fin(1 - uniq_exact / n) if n else None,
        "normalized_duplicate_rate": fin(1 - uniq_norm / n) if n else None,
        "unique_outputs_per_section": sec_unique,
        "sections_effectively_constant": [t for t, v in sec_unique.items() if v == 1],
        "mean_pairwise_rouge_l_f1": mean(f1s),
        "max_pairwise_rouge_l_f1": fin(max(f1s)) if f1s else None,
        "normalized_only_duplicates": len(cross),
        "pairwise": pairs,
        "avg_output_words": mean(lens),
        "distinct_sentences_total": len(all_sents),
        "repeated_sentences_by_section": repeated,
        "top_repeated_sentences_all": [
            {"sentence": s, "count": n2}
            for s, n2 in all_sents.most_common(12) if n2 > 1
        ],
        "shared_text_proportion_across_outputs": mean(f1s),
    }


# ---------------------------------------------------------------------------
# Stage D: semantic distance vs text change
# ---------------------------------------------------------------------------

def stage_d(recs, schema_names, cat_of) -> dict:
    real = [r for r in recs if not r["is_mock"] and r["vector"]]
    rows = []
    for i in range(len(real)):
        for j in range(i + 1, len(real)):
            a, b = real[i], real[j]
            va, vb = a["vector"], b["vector"]
            diffs = [abs((x or 0) - (y or 0)) for x, y in zip(va, vb)]
            _, _, f = rouge_l_one(a["text"], b["text"])
            ma, _ = parse_sections(a["text"])
            mb, _ = parse_sections(b["text"])
            changed = [t for t in CATEGORY_TITLES.values()
                       if norm_text(ma.get(t, "")) != norm_text(mb.get(t, ""))]
            rows.append({
                "image_a": a["result_id"], "image_b": b["result_id"],
                "semantic_distance": fin(euclid(va, vb)),
                "semantic_mad": fin(sum(diffs) / len(diffs)),
                "semantic_cosine_similarity": cosine(va, vb),
                "semantic_traits_changed": sum(1 for d in diffs if d > TRAIT_BIG_DIFF),
                "text_similarity": fin(f),
                "text_changed": (f or 0) < 0.999,
                "sections_changed": changed,
            })
    sem = [r["semantic_distance"] for r in rows if r["semantic_distance"] is not None]
    txt = [1 - (r["text_similarity"] or 0) for r in rows]
    pc = pearson(sem, txt)
    return {
        "n_files": len(real),
        "n_pairs": len(rows),
        "note": "Diagnostic evidence only; n too small for significance claims.",
        "pairs": rows,
        "pearson_semantic_vs_text_distance": pc,
        "spearman_semantic_vs_text_distance": spearman(sem, txt),
        "larger_semantic_difference_larger_text_difference": bool(pc is not None and pc > 0),
    }


# ---------------------------------------------------------------------------
# Stage E: template vs FLAN-T5 on identical inputs
# ---------------------------------------------------------------------------

def _expressed_concepts(text):
    """Lexicon-matched concepts per category section (explicit mentions only)."""
    secs, _ = parse_sections(text)
    found = set()
    for c in CATEGORY_ORDER:
        body = secs.get(CATEGORY_TITLES[c], "")
        if not body:
            continue
        for t, (tc, pats) in TRAIT_PATTERNS.items():
            if tc == c and match_trait(body, pats):
                found.add(t)
    return found


def _threshold_of(t, tmap, policy):
    return tmap.get(t, 0.5) if (policy == "calibrated" and tmap) else 0.5


def _faithfulness(probs, expressed, tmap, policy):
    active = {t for t in expressed if probs.get(t, 0) >= _threshold_of(t, tmap, policy)}
    unsupported = expressed - active
    strong = {t for t in unsupported if probs.get(t, 1.0) < STRONG_UNSUPPORTED_PROB}
    recoverable = {t for t in probs if t in TRAIT_PATTERNS
                   and probs.get(t, 0) >= _threshold_of(t, tmap, policy)}
    p = len(active) / len(expressed) if expressed else None
    r = len(active) / len(recoverable) if recoverable else None
    f1 = (2 * p * r / (p + r)) if (p and r) else (fin(p) if p else None)
    return {"expressed": sorted(expressed), "supported": sorted(active),
            "unsupported": sorted(unsupported),
            "strongly_unsupported": sorted(strong),
            "precision": fin(p), "recall": fin(r), "f1": fin(f1)}


def _system_stats(texts, probs_list, tmap, policy):
    n = len(texts)
    sec_maps = []
    for t in texts:
        m, _ = parse_sections(t)
        sec_maps.append(m)
    sec_unique = {
        CATEGORY_TITLES[c]: len({norm_text(m.get(CATEGORY_TITLES[c], ""))
                                 for m in sec_maps})
        for c in CATEGORY_ORDER
    }
    faiths = [_faithfulness(probs, _expressed_concepts(t), tmap, policy)
              for t, probs in zip(texts, probs_list)]
    avg_words = mean([len(t.split()) for t in texts]) if texts else None
    def _m(key):
        return mean([f[key] for f in faiths if f[key] is not None])
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(rouge_l_one(texts[i], texts[j])[2])
    uniq = len(set(texts))
    # repeated-sentence per system (mirrors Stage C logic, per-system)
    all_sents = Counter(norm_text(s) for t in texts for s in split_sentences(t) if s.strip())
    total_sents = sum(all_sents.values())
    distinct_sents = len(all_sents)
    repeated_cnt = sum(1 for c in all_sents.values() if c > 1)
    repeated_rate = fin(1 - distinct_sents / total_sents) if total_sents else None
    # per-section repeated map (top duplicates)
    repeated_by_section = {}
    for c in CATEGORY_ORDER:
        title = CATEGORY_TITLES[c]
        sents = [s for m in sec_maps for s in split_sentences(m.get(title, ""))]
        cnt = Counter(norm_text(s) for s in sents if s.strip())
        repeated_by_section[title] = [{"sentence": s, "count": n2} for s, n2 in cnt.most_common(8) if n2 > 1]
    return {
        "n_outputs": n,
        "avg_output_words": fin(avg_words),
        "unique_full_texts": uniq,
        "unique_section_texts": sec_unique,
        "sections_effectively_constant": [k for k, v in sec_unique.items() if v == 1],
        "exact_duplicate_rate": fin(1 - uniq / n) if n else None,
        "mean_pairwise_rouge_l_f1": mean(sims),
        "distinct_sentences_total": distinct_sents,
        "total_sentences": total_sents,
        "repeated_sentence_rate": repeated_rate,
        "repeated_sentence_count": repeated_cnt,
        "repeated_sentences_by_section": repeated_by_section,
        "top_repeated_sentences": [{"sentence": s, "count": n2} for s, n2 in all_sents.most_common(12) if n2 > 1],
        "faithfulness": {
            "policy": policy,
            "precision_mean": _m("precision"),
            "recall_mean": _m("recall"),
            "f1_mean": _m("f1"),
            "unsupported_total": sum(len(f["unsupported"]) for f in faiths),
            "strongly_unsupported_total": sum(
                len(f["strongly_unsupported"]) for f in faiths),
            "avg_active_concepts": mean([len(f["supported"]) for f in faiths]),
        },
    }


def stage_e(recs, schema_names, cat_of, tmap) -> dict:
    real = [r for r in recs if not r["is_mock"] and r["vector"]]
    probs_list = [dict(zip(schema_names, r["vector"])) for r in real]
    flan_texts = [r["text"] for r in real]
    t05 = [template_generate(p, threshold=0.5, thresholds=None)[0] for p in probs_list]
    tcal = [template_generate(p, threshold=0.5, thresholds=tmap)[0] for p in probs_list]
    tcon = [generate_text_concise(p, threshold=0.5, thresholds=tmap)[0]
            for p in probs_list]
    systems = {
        "flan_t5_continuous": _system_stats(flan_texts, probs_list, tmap, "threshold_0.5"),
        "template_0.5": _system_stats(t05, probs_list, tmap, "threshold_0.5"),
        "template_calibrated": _system_stats(tcal, probs_list, tmap, "calibrated"),
        "template_concise": _system_stats(tcon, probs_list, tmap, "calibrated"),
    }
    template_varies = (systems["template_0.5"]["unique_full_texts"] >
                       systems["flan_t5_continuous"]["unique_full_texts"])
    flan_more_faithful = (
        (systems["flan_t5_continuous"]["faithfulness"]["f1_mean"] or 0) >=
        (systems["template_0.5"]["faithfulness"]["f1_mean"] or 0))
    return {
        "inputs": "identical 45-d semantic vectors from demo result JSONs; no re-extraction",
        "n_inputs": len(real),
        "systems": systems,
        "template_changes_when_vector_changes": template_varies,
        "flan_more_faithful_than_template": flan_more_faithful,
        "key_question_answer": (
            "template outputs vary with input while FLAN-T5 stays near-constant"
            if template_varies and systems["flan_t5_continuous"]["unique_full_texts"] <= 3
            else "measured; see systems block"),
    }


# ---------------------------------------------------------------------------
# Stage F: semantic contradiction analysis
# ---------------------------------------------------------------------------

def stage_f(recs, schema_names, cat_of, tmap) -> dict:
    real = [r for r in recs if not r["is_mock"] and r["vector"]]
    by_image, by_section = {}, defaultdict(list)
    tot_e = tot_u = tot_s = 0
    precisions, recalls, f1s = [], [], []
    for r in real:
        probs = dict(zip(schema_names, r["vector"]))
        expressed = _expressed_concepts(r["text"])
        res05 = _faithfulness(probs, expressed, tmap, "threshold_0.5")
        resc = _faithfulness(probs, expressed, tmap, "calibrated")
        tot_e += len(expressed)
        tot_u += len(resc["unsupported"])
        tot_s += len(resc["strongly_unsupported"])
        for key, bucket in (("precision", precisions), ("recall", recalls), ("f1", f1s)):
            if resc[key] is not None:
                bucket.append(resc[key])
        secs, _ = parse_sections(r["text"])
        for c in CATEGORY_ORDER:
            title = CATEGORY_TITLES[c]
            exp_c = {t for t in expressed if TRAIT_PATTERNS[t][0] == c}
            by_section[title].append({
                "result_id": r["result_id"],
                "expressed": sorted(exp_c),
                "unsupported": sorted(set(resc["unsupported"]) & exp_c),
                "strongly_unsupported": sorted(set(resc["strongly_unsupported"]) & exp_c),
                "section_present": bool(secs.get(title, "").strip()),
            })
        by_image[r["result_id"]] = {
            "probabilities": {k: fin(v) for k, v in probs.items()},
            "policy_calibrated": resc,
            "policy_threshold_0.5": res05,
        }
    sec_summary = {
        title: {
            "unsupported_total": sum(len(i["unsupported"]) for i in items),
            "strongly_unsupported_total": sum(
                len(i["strongly_unsupported"]) for i in items),
            "expressed_total": sum(len(i["expressed"]) for i in items),
            "empty_section_count": sum(1 for i in items if not i["section_present"]),
        }
        for title, items in by_section.items()
    }
    return {
        "n_outputs": len(real),
        "strongly_unsupported_prob_threshold": STRONG_UNSUPPORTED_PROB,
        "policy": "calibrated thresholds (trait_thresholds.json) unless noted",
        "contradictions_by_image": by_image,
        "contradictions_by_section": sec_summary,
        "total_expressed_concepts": tot_e,
        "total_unsupported_concepts": tot_u,
        "total_strongly_unsupported_concepts": tot_s,
        "input_faithfulness": {
            "precision_mean": mean(precisions),
            "recall_mean": mean(recalls),
            "f1_mean": mean(f1s),
        },
        "note": ("Low probabilities are model estimates of provider-text concepts, "
                 "not evidence about real human characteristics. Explicit lexicon "
                 "mentions only; loose associations excluded."),
    }


# ---------------------------------------------------------------------------
# Stage G: provider training-corpus diversity + conditional information
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _section_texts(rec):
    """Per-category target section text from a record."""
    pred = rec.get("predictions")
    if isinstance(pred, dict):
        out = {}
        for c in CATEGORY_ORDER:
            vals = pred.get(c) or []
            if isinstance(vals, str):
                vals = [vals]
            out[c] = " ".join(str(v) for v in vals)
        return out
    text = rec.get("completion") or rec.get("target_text") or ""
    m, _ = parse_sections(text)
    return {c: m.get(CATEGORY_TITLES[c], "") for c in CATEGORY_ORDER}


def _dup_stats(texts):
    n = len(texts)
    if not n:
        return {}
    norm = [norm_text(t) for t in texts]
    uniq = len(set(texts))
    uniq_norm = len(set(norm))
    cnt = Counter(norm)
    top1 = cnt.most_common(1)[0] if cnt else ("", 0)
    top5 = sum(c for _, c in cnt.most_common(5))
    lens = sorted(len(t.split()) for t in texts)

    def pct(p):
        return lens[min(len(lens) - 1, int(p * len(lens)))] if lens else 0

    return {
        "n": n,
        "unique_exact": uniq,
        "exact_duplicate_rate": fin(1 - uniq / n),
        "unique_normalized": uniq_norm,
        "normalized_duplicate_rate": fin(1 - uniq_norm / n),
        "top1_template_frequency": fin(top1[1] / n) if top1[0] else None,
        "top1_template_text_preview": top1[0][:200],
        "top5_template_coverage": fin(top5 / n),
        "mean_length_words": mean(lens),
        "length_p50": pct(0.5),
        "length_p90": pct(0.9),
        "unique_sentences": len({norm_text(s) for t in texts
                                 for s in split_sentences(t)}),
        "most_common_texts": [{"text_preview": t[:160], "count": c}
                              for t, c in cnt.most_common(5) if c > 1],
    }


def stage_g(sample_n=300, neighbor_n=150) -> dict:
    report = {"sources": {}}
    for label, path in (("text_generation_train", TRAIN_JSONL),
                        ("text_generation_instruct", INSTRUCT_JSONL),
                        ("labels_enriched", ENRICHED_JSONL)):
        if not path.exists():
            report["sources"][label] = {"error": "missing", "path": str(path)}
            continue
        rows = _read_jsonl(path)
        if label == "text_generation_train":
            full = [r.get("target_text") or "" for r in rows]
        elif label == "text_generation_instruct":
            full = [r.get("completion") or "" for r in rows]
        else:
            full = [" \n".join(_section_texts(r)[c] for c in CATEGORY_ORDER)
                    for r in rows]
        stats = _dup_stats(full)
        per_cat = {}
        for c in CATEGORY_ORDER:
            per_cat[c] = _dup_stats([_section_texts(r).get(c, "") for r in rows])
        report["sources"][label] = {
            "path": str(path),
            "total_records": len(rows),
            "unique_full_targets": stats.get("unique_exact"),
            "duplicate_target_rate": stats.get("exact_duplicate_rate"),
            "normalized_duplicate_rate": stats.get("normalized_duplicate_rate"),
            "top1_template_frequency": stats.get("top1_template_frequency"),
            "top5_template_coverage": stats.get("top5_template_coverage"),
            "mean_target_length_words": stats.get("mean_length_words"),
            "per_category": per_cat,
            "most_common_full_targets": stats.get("most_common_texts"),
            "length_distribution": {"p50": stats.get("length_p50"),
                                    "p90": stats.get("length_p90")},
            "unique_sentences": stats.get("unique_sentences"),
        }

    # ---- conditional information using OOF probabilities ----
    cond = {"oof_source": str(OOF_CSV),
            "note": "OOF train-only candidate pool; no test leak."}
    if TRAIN_JSONL.exists() and OOF_CSV.exists():
        import pandas as pd
        oof = pd.read_csv(OOF_CSV, encoding="utf-8-sig")
        id_col = oof.columns[0]
        trait_cols = [c for c in oof.columns if c != id_col]
        oof_map = {norm_name(str(row[id_col])): [fin(row[c]) for c in trait_cols]
                   for _, row in oof.iterrows()}
        train_rows = _read_jsonl(TRAIN_JSONL)
        splits = json.loads(SPLITS.read_text(encoding="utf-8")) if SPLITS.exists() else {}
        test_set = set(splits.get("test_images") or [])
        paired = [(r, oof_map[norm_name(r.get("image_id", ""))])
                  for r in train_rows if norm_name(r.get("image_id", "")) in oof_map]
        cond["paired_records"] = len(paired)
        rng = random.Random(42)
        sample = paired if len(paired) <= sample_n else rng.sample(paired, sample_n)
        xs, ys = [], []
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                (ra, va), (rb, vb) = sample[i], sample[j]
                xs.append(euclid([v or 0 for v in va], [v or 0 for v in vb]))
                ys.append(rouge_l_one(ra.get("target_text", ""),
                                      rb.get("target_text", ""))[2])
        cond["sample_pairs"] = len(xs)
        cond["pearson_semantic_distance_vs_target_similarity"] = pearson(xs, ys)
        cond["spearman_semantic_distance_vs_target_similarity"] = spearman(xs, ys)

        def _pair_desc(pick, dist):
            i, j = pick
            ra, rb = sample[i][0], sample[j][0]
            return {"semantic_distance": fin(dist),
                    "image_a": ra.get("image_id"), "image_b": rb.get("image_id"),
                    "target_a_preview": (ra.get("target_text") or "")[:220],
                    "target_b_preview": (rb.get("target_text") or "")[:220],
                    "targets_equal": norm_text(ra.get("target_text", "")) ==
                    norm_text(rb.get("target_text", ""))}

        best_far, best_near = (None, -1e9), (None, 1e9)
        m = min(len(sample), 80)
        for i in range(m):
            for j in range(i + 1, m):
                (_, va), (_, vb) = sample[i], sample[j]
                d = euclid([v or 0 for v in va], [v or 0 for v in vb])
                if d > best_far[1]:
                    best_far = ((i, j), d)
                if d < best_near[1]:
                    best_near = ((i, j), d)
        cond["illustrative_pair_most_distant"] = _pair_desc(best_far[0], best_far[1])
        cond["illustrative_pair_most_similar"] = _pair_desc(best_near[0], best_near[1])

        # held-out baselines (candidate pool = train only)
        test_rows = [(r, oof_map[norm_name(r.get("image_id", ""))])
                     for r in train_rows
                     if norm_name(r.get("image_id", "")) in test_set
                     and norm_name(r.get("image_id", "")) in oof_map]
        train_only = [(r, v) for r, v in paired
                      if norm_name(r.get("image_id", "")) not in test_set]
        if test_rows and train_only:
            take = test_rows if len(test_rows) <= neighbor_n else rng.sample(test_rows, neighbor_n)
            mc = Counter(norm_text(r.get("target_text", ""))
                         for r, _ in train_only).most_common(1)
            mc_text = next((r.get("target_text", "") for r, _ in train_only
                            if norm_text(r.get("target_text", "")) == mc[0][0]), "") if mc else ""
            mc_scores, nn_scores, nn_diff = [], [], 0
            train_vecs = [[x or 0 for x in v] for _, v in train_only]
            train_texts = [r.get("target_text", "") for r, _ in train_only]
            for r, v in take:
                own = r.get("target_text", "")
                mc_scores.append(rouge_l_one(mc_text, own)[2] if mc_text else 0)
                vv = [x or 0 for x in v]
                k = min(range(len(train_vecs)), key=lambda i: euclid(vv, train_vecs[i]))
                nn_scores.append(rouge_l_one(train_texts[k], own)[2])
                if norm_text(train_texts[k]) != norm_text(own):
                    nn_diff += 1
            cond["heldout_baselines"] = {
                "n_test_sampled": len(take),
                "candidate_pool_train_only": len(train_only),
                "most_common_target_rougeL": mean(mc_scores),
                "nearest_neighbor_target_rougeL": mean(nn_scores),
                "nearest_neighbor_differs_rate": fin(nn_diff / len(take)),
            }
    report["conditional_information"] = cond

    # ---- label-noise diagnostics (read-only) ----
    noise = {"traits_correlated": [], "notes": []}
    if TRAIN_JSONL.exists() and OOF_CSV.exists():
        import pandas as pd
        oof = pd.read_csv(OOF_CSV, encoding="utf-8-sig")
        id_col = oof.columns[0]
        cols = [c for c in oof.columns if c != id_col]
        corr = oof[cols].corr().abs()
        pairs = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                pairs.append((cols[i], cols[j], fin(corr.iloc[i, j])))
        pairs = [p for p in pairs if p[2] is not None]
        pairs.sort(key=lambda p: -p[2])
        noise["traits_correlated"] = [
            {"trait_a": a, "trait_b": b, "abs_corr": c}
            for a, b, c in pairs[:15] if c >= 0.5]
        noise["n_pairs_abs_corr_ge_0.5"] = sum(1 for _, _, c in pairs if c >= 0.5)
        noise["notes"].append(
            "Traits extracted from provider text with a fixed lexicon; high "
            "co-occurrence can reflect template reuse rather than human traits.")
    report["label_noise_findings"] = noise
    return report


# ---------------------------------------------------------------------------
# Root-cause classification
# ---------------------------------------------------------------------------

def classify(a, b, c, e, g) -> dict:
    causes = []
    dup_imgs = len(a["byte_duplicate_groups"])
    mad = (b.get("duplicate_adjusted") or {}).get("mean_mad")
    text_sim = c.get("mean_pairwise_rouge_l_f1")
    provider = (g.get("sources") or {}).get("text_generation_train") or {}
    provider_dup = provider.get("normalized_duplicate_rate")

    if dup_imgs and mad is None:
        causes.append("A. image reuse explains identical vectors (expected, not a bug)")
    elif dup_imgs and mad is not None and mad < VECT_MAD_COLLAPSE:
        causes.append("A. image input reuse contributes (duplicate uploads)")
    if mad is not None and mad < VECT_MAD_COLLAPSE:
        causes.append("B/C. feature or semantic collapse (vectors nearly identical across distinct images)")
    elif mad is not None:
        causes.append("no B/C: distinct images -> substantially different semantic vectors")
    if (mad is not None and mad >= VECT_MAD_COLLAPSE
            and text_sim is not None and text_sim >= TEXT_SIM_COLLAPSE):
        causes.append("D. generator conditioning collapse (vectors differ, text nearly identical)")
    if provider_dup is not None and provider_dup >= 0.3:
        causes.append("E. provider target collapse (highly repetitive training targets)")
    elif provider_dup is not None:
        causes.append("no E: provider targets retain diversity")
    if any(x.startswith("D.") for x in causes) and any(x.startswith("E.") for x in causes):
        causes.append("F. combined cause: generator conditioning + dataset repetition")
    primary = "D. generator conditioning collapse"
    if any(x.startswith("B/C") for x in causes):
        primary = "B/C. upstream feature/semantic collapse"
    elif any(x.startswith("F.") for x in causes):
        primary = "F. combined (D + E)"
    return {"classified_causes": causes, "primary": primary,
            "retraining_justified": "not yet - audit first; see recommendations"}


def main():
    global STRONG_UNSUPPORTED_PROB
    ap = argparse.ArgumentParser(description="Demo output-collapse diagnostics")
    ap.add_argument("--result-dir", action="append", default=None,
                    help="directory with *_result.json (repeatable); "
                         "default: outputs and outputs/demo_tmp")
    ap.add_argument("--strong-unsupported-prob", type=float,
                    default=STRONG_UNSUPPORTED_PROB)
    ap.add_argument("--out-dir", default=str(SRC))
    args = ap.parse_args()
    STRONG_UNSUPPORTED_PROB = args.strong_unsupported_prob
    dirs = args.result_dir or ["outputs", str(ROOT / "outputs" / "demo_tmp")]
    out = Path(args.out_dir)

    schema_names, cat_of = load_schema()
    tmap, strategy = (template_load_thresholds(str(CALIB_PATH))
                      if CALIB_PATH.exists() else (None, None))

    recs = load_results(dirs)
    print(f"Loaded {len(recs)} result files "
          f"({sum(1 for r in recs if not r['is_mock'])} real)")
    a = stage_a(recs)
    b = stage_b(recs, schema_names, cat_of, tmap)
    c = stage_c(recs, schema_names, cat_of)
    d = stage_d(recs, schema_names, cat_of)
    e = stage_e(recs, schema_names, cat_of, tmap)
    f = stage_f(recs, schema_names, cat_of, tmap)
    g = stage_g()
    root = classify(a, b, c, e, g)

    write_json(out / "demo_semantic_diversity.json",
               {"stage": "B", "stage_a_image_audit": a, **b})
    write_json(out / "demo_text_diversity.json", {"stage": "C", **c})
    write_json(out / "demo_collapse_relationship.json",
               {"stage": "D", **d, "root_cause_assessment": root})
    write_json(out / "demo_generator_comparison.json",
               {"stage": "E", "threshold_policy": strategy, **e})
    write_json(out / "demo_semantic_contradictions.json", {"stage": "F", **f})
    write_json(out / "provider_text_diversity_report.json", {"stage": "G", **g})
    print("\nRoot cause:", json.dumps(root, indent=2))


if __name__ == "__main__":
    main()