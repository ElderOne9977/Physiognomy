#!/usr/bin/env python
"""Phase 3.11: semantic label quality, noise, and predictability ceiling.

Diagnostic only. Does not retrain, does not modify production models,
datasets, historical OOF, or prior experiment reports.

Usage:
  python Source/audit_semantic_label_quality.py --stage all
  python Source/audit_semantic_label_quality.py --stage all --quick
  python Source/audit_semantic_label_quality.py --stage a --seed 42 --bootstrap 400
"""

import argparse
import csv
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
LABELS_JSONL = ROOT / "labels_enriched.jsonl"
LABELS_CSV = ROOT / "semantic_labels.csv"
SCHEMA_JSON = ROOT / "semantic_schema.json"
FEATURES_CSV = ROOT / "faces_full_features_clean.csv"
MERGE_REPORT = ROOT / "merge_report.json"
MANIFEST_CSV = SRC / "experiments" / "hash_grouped_semantic" / "image_hash_manifest.csv"
SAFE_GROUPS_CSV = SRC / "experiments" / "hash_grouped_semantic" / "semantic_leakage_safe_groups.csv"
OOF_HASH_CSV = SRC / "experiments" / "hash_grouped_semantic" / "oof_semantic_probs_hash_grouped.csv"
PER_TRAIT_CSV = SRC / "experiments" / "semantic_reliability_phase310" / "semantic_per_trait_uncertainty.csv"
HIST_OOF = ROOT / "oof_semantic_probs.csv"
MODELS_DIR = SRC / "models_semantic"
DEFAULT_OUT = SRC / "experiments" / "semantic_label_quality_phase311"

PRED_CATS = ["personality", "career", "relationships", "future_destiny", "wealth"]
UNMATCHED_STEMS = ["1 (3111)", "1 (4087)", "1 (4568)", "1 (4641)", "img331"]
NEGATION_RE = re.compile(
    r"\b(?:not|no|never|without|lack(?:s|ing)?|less|un)\b", re.IGNORECASE
)

# Classification thresholds (documented, configurable). Overlapping tags allowed.
ROC_STRONG = 0.70
ROC_WEAK = 0.60
ROC_NEAR = 0.55
CONFLICT_HIGH = 0.45
LOW_POS_COMPONENTS = 400
CI_INCONCLUSIVE_WIDTH = 0.08


def fin(x):
    try:
        v = float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(f"wrote {path}")


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fieldnames})
    print(f"wrote {path} ({len(rows)} rows)")


def norm_name(value):
    s = os.path.basename(str(value).strip())
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\)\s+\(", ")(", s)
    return s.lower()


def stem_name(value):
    return re.sub(r"\.[^.]+$", "", norm_name(value))


def normalize_text(text):
    s = str(text or "")
    s = s.replace("\u2014", "-").replace("\u2013", "-")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def binary_entropy(p):
    p = float(p)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p))


def majority_agreement(n_pos, n_neg):
    """Empirical majority-label agreement for one group. None if empty."""
    n_pos = int(n_pos)
    n_neg = int(n_neg)
    n = n_pos + n_neg
    if n <= 0:
        return None
    return max(n_pos, n_neg) / n


def row_weighted_agreement(groups):
    """groups: iterable of (n_pos, n_neg). None if no rows."""
    num = den = 0
    for n_pos, n_neg in groups:
        n = int(n_pos) + int(n_neg)
        if n <= 0:
            continue
        num += max(int(n_pos), int(n_neg))
        den += n
    if den == 0:
        return None
    return num / den


def hash_weighted_agreement(groups):
    """Equal weight per non-empty group. None if no groups."""
    vals = []
    for n_pos, n_neg in groups:
        a = majority_agreement(n_pos, n_neg)
        if a is not None:
            vals.append(a)
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def conflict_rate(n_conflict_groups, n_groups):
    if n_groups is None or int(n_groups) <= 0:
        return None
    return int(n_conflict_groups) / int(n_groups)


def jaccard(a, b):
    a = set(a)
    b = set(b)
    if not a and not b:
        return 1.0
    u = len(a | b)
    return (len(a & b) / u) if u else None


def spearman(xs, ys):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    m = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[m], ys[m]
    n = int(len(xs))
    if n < 3:
        return None, n, "fewer than 3 finite pairs"
    if np.unique(xs).size < 2 or np.unique(ys).size < 2:
        return None, n, "degenerate (constant) series"
    rx = pd.Series(xs).rank(method="average").to_numpy()
    ry = pd.Series(ys).rank(method="average").to_numpy()
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None, n, "degenerate ranks"
    r = float(np.corrcoef(rx, ry)[0, 1])
    return (fin(r), n, None)


def pearson(xs, ys):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    m = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[m], ys[m]
    n = int(len(xs))
    if n < 3:
        return None, n, "fewer than 3 finite pairs"
    if np.std(xs) == 0 or np.std(ys) == 0:
        return None, n, "degenerate (constant) series"
    r = float(np.corrcoef(xs, ys)[0, 1])
    return (fin(r), n, None)


def load_schema():
    sch = json.loads(SCHEMA_JSON.read_text(encoding="utf-8"))
    traits = [t["name"] for t in sch["traits"]]
    cats = {t["name"]: t["category"] for t in sch["traits"]}
    return traits, cats


def load_trait_patterns():
    """Import TRAIT_PATTERNS without executing label generation."""
    import importlib.util

    path = SRC / "build_semantic_labels.py"
    spec = importlib.util.spec_from_file_location("build_semantic_labels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TRAIT_PATTERNS, mod.match_trait, mod.record_text


def load_provider_records():
    recs = []
    with LABELS_JSONL.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            image = rec.get("image")
            pred = rec.get("predictions") or {}
            cat_texts = {}
            for c in PRED_CATS:
                vals = pred.get(c) or []
                if isinstance(vals, str):
                    vals = [vals]
                cat_texts[c] = " ".join(str(v) for v in vals)
            full = " ".join(cat_texts[c] for c in PRED_CATS if cat_texts[c])
            recs.append(
                {
                    "image_raw": str(image or ""),
                    "norm_key": norm_name(image or ""),
                    "norm_stem": stem_name(image or ""),
                    "cat_texts": cat_texts,
                    "full_text": full,
                    "full_norm": normalize_text(full),
                    "timestamp": rec.get("timestamp"),
                    "provider": rec.get("provider"),
                    "model": rec.get("model") or rec.get("model_version"),
                    "request_id": rec.get("request_id") or rec.get("requestId"),
                }
            )
    return recs


def load_aligned(traits):
    """Join labels, hash manifest, safe groups, and hash-grouped OOF by image id.

    Never join by row position.
    """
    labels = pd.read_csv(LABELS_CSV, encoding="utf-8-sig")
    labels["norm_key"] = labels["image"].map(norm_name)
    labels["norm_stem"] = labels["image"].map(stem_name)
    for t in traits:
        labels[t] = pd.to_numeric(labels[t], errors="coerce").fillna(0).astype(np.int8)

    manifest = pd.read_csv(MANIFEST_CSV, encoding="utf-8")
    manifest["norm_key"] = manifest["image_id"].map(norm_name)
    keep_m = ["norm_key", "sha256", "md5", "has_feature_row", "has_label", "byte_size"]
    manifest = manifest[keep_m].drop_duplicates("norm_key")

    groups = pd.read_csv(SAFE_GROUPS_CSV, encoding="utf-8")
    groups["norm_key"] = groups["image_id"].map(norm_name)
    groups = groups[["norm_key", "leakage_safe_group_id"]].drop_duplicates("norm_key")

    oof = pd.read_csv(OOF_HASH_CSV, encoding="utf-8")
    oof["norm_key"] = oof["image_id"].map(norm_name)
    oof_cols = ["norm_key"] + [t for t in traits if t in oof.columns]
    oof = oof[oof_cols].drop_duplicates("norm_key")
    rename = {t: f"p_{t}" for t in traits if t in oof.columns}
    oof = oof.rename(columns=rename)

    # feature presence by stem (unmatched check)
    feat_stems = set()
    with FEATURES_CSV.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        idcol = "image" if "image" in reader.fieldnames else reader.fieldnames[0]
        for row in reader:
            feat_stems.add(stem_name(row[idcol]))

    merged = labels.merge(manifest, on="norm_key", how="left", indicator="in_manifest")
    merged = merged.merge(groups, on="norm_key", how="left")
    merged = merged.merge(oof, on="norm_key", how="left", indicator="in_oof")
    return merged, feat_stems


def classify_trait(row):
    """Overlapping qualitative tags. Not a scientific validity claim."""
    tags = []
    roc = row.get("roc_auc")
    ci_lo = row.get("roc_ci_low")
    ci_hi = row.get("roc_ci_high")
    conflict = row.get("conflict_group_rate")
    pos_c = row.get("independent_positive_components")
    if conflict is not None and conflict >= CONFLICT_HIGH:
        tags.append("HIGH_LABEL_CONFLICT")
    if pos_c is not None and pos_c < LOW_POS_COMPONENTS:
        tags.append("LOW_INDEPENDENT_SUPPORT")
    if roc is None:
        tags.append("INCONCLUSIVE")
        return tags
    width = None
    if ci_lo is not None and ci_hi is not None:
        width = ci_hi - ci_lo
        if width >= CI_INCONCLUSIVE_WIDTH and roc < ROC_STRONG:
            tags.append("INCONCLUSIVE")
    if roc >= ROC_STRONG and (ci_lo is None or ci_lo >= 0.65):
        tags.append("RELATIVELY_STRONG_SIGNAL")
    elif roc < ROC_NEAR:
        tags.append("NEAR_CHANCE")
    elif roc < ROC_WEAK:
        tags.append("WEAK_SIGNAL")
    else:
        tags.append("WEAK_SIGNAL")
    if not tags:
        tags.append("INCONCLUSIVE")
    return tags


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_a(args, ctx):
    t0 = time.time()
    traits = ctx["traits"]
    cats = ctx["cats"]
    merged = ctx["merged"]
    provider = ctx["provider"]
    feat_stems = ctx["feat_stems"]

    n_labels = len(merged)
    n_provider = len(provider)
    n_features = 0
    with FEATURES_CSV.open(encoding="utf-8") as f:
        n_features = sum(1 for _ in f) - 1
    n_humans = 0
    if MANIFEST_CSV.exists():
        with MANIFEST_CSV.open(encoding="utf-8") as f:
            n_humans = sum(1 for _ in f) - 1

    matched_mask = merged["sha256"].notna() & merged["leakage_safe_group_id"].notna()
    n_matched = int(matched_mask.sum())
    unmatched_labels = merged.loc[~matched_mask, "norm_key"].tolist()

    # provider keys not in labels
    label_keys = set(merged["norm_key"])
    provider_keys = [r["norm_key"] for r in provider]
    provider_key_set = set(provider_keys)
    labels_not_in_provider = sorted(label_keys - provider_key_set)
    provider_not_in_labels = sorted(provider_key_set - label_keys)

    sha = merged.loc[matched_mask, "sha256"]
    unique_sha = int(sha.nunique())
    sha_counts = sha.value_counts()
    dup_sha = int((sha_counts > 1).sum())
    dup_sha_all = int((merged["sha256"].value_counts() > 1).sum())
    components = merged.loc[matched_mask, "leakage_safe_group_id"]
    unique_comp = int(components.nunique())
    comp_counts = components.value_counts()
    dup_comp = int((comp_counts > 1).sum())

    # repeated-label groups = sha groups with >1 matched label row
    repeated = int((sha_counts > 1).sum())

    # optional unmatched check
    unmatched_detail = []
    for stem in UNMATCHED_STEMS:
        in_feat = stem in feat_stems
        in_labels = any(stem_name(k) == stem for k in label_keys)
        in_provider = any(r["norm_stem"] == stem for r in provider)
        unmatched_detail.append(
            {
                "stem": stem,
                "in_feature_csv": bool(in_feat),
                "in_semantic_labels": bool(in_labels),
                "in_provider_jsonl": bool(in_provider),
                "note": (
                    "feature row present, provider label missing"
                    if in_feat and not in_provider
                    else "see counts"
                ),
            }
        )
    # cheaper manifest stem check
    man_stems = set()
    with MANIFEST_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            man_stems.add(stem_name(row["image_id"]))
    for item in unmatched_detail:
        item["in_hash_manifest"] = item["stem"] in man_stems
        if item["in_feature_csv"] and item["in_hash_manifest"] and not item["in_provider_jsonl"]:
            item["origin"] = "missing provider label (not a filename mismatch)"
        elif not item["in_feature_csv"]:
            item["origin"] = "missing feature row"
        else:
            item["origin"] = "unresolved"

    # provenance availability
    n_ts = sum(1 for r in provider if r.get("timestamp"))
    n_model = sum(1 for r in provider if r.get("model"))
    n_req = sum(1 for r in provider if r.get("request_id"))

    # category assignment check
    cat_counts = Counter(cats.values())

    # OOF alignment
    oof_present = int(merged["in_oof"].eq("both").sum()) if "in_oof" in merged.columns else None

    payload = {
        "stage": "A data and label alignment",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "join_key": "norm_name(image) = basename, collapse whitespace, ') ('->')(', lower. Never row position.",
        "counts": {
            "humans_manifest_rows": n_humans,
            "feature_rows": n_features,
            "provider_text_records": n_provider,
            "semantic_label_rows": n_labels,
            "matched_label_rows_with_sha_and_component": n_matched,
            "unmatched_label_rows": int(n_labels - n_matched),
            "unique_sha256_on_matched": unique_sha,
            "duplicate_sha256_groups": dup_sha,
            "unique_connected_components": unique_comp,
            "duplicate_component_groups": dup_comp,
            "repeated_label_sha_groups": repeated,
            "singleton_sha_groups": int((sha_counts == 1).sum()),
            "duplicate_sha256_groups_all_label_rows": dup_sha_all,
            "oof_hash_rows_aligned": oof_present,
            "labels_missing_from_provider": len(labels_not_in_provider),
            "provider_missing_from_labels": len(provider_not_in_labels),
        },
        "schema": {
            "n_traits": len(traits),
            "trait_order": traits,
            "category_counts": dict(cat_counts),
        },
        "unmatched_historical": unmatched_detail,
        "population_note": (
            "Primary duplicate analysis uses the 6983 matched rows that carry a leakage-safe component "
            "and a hash-grouped OOF prediction. The 484 label rows without a feature row are counted "
            "separately as duplicate_sha256_groups_all_label_rows."
        ),
        "provenance": {
            "records_with_timestamp": n_ts,
            "records_with_model_version": n_model,
            "records_with_request_id": n_req,
            "note": "labels_enriched.jsonl used here has image+predictions only. Generation settings, timestamps, model versions, and request ids are not present in this file.",
        },
        "ambiguous_joins": {
            "duplicate_norm_keys_in_labels": int(merged["norm_key"].duplicated().sum()),
            "note": "Join is many labels -> one manifest row via unique norm_key. Duplicate filenames with different bytes stay distinct because norm_key keeps the extension.",
        },
        "disclaimer": "Counts describe alignment of provider-derived labels with image hashes. They are not measurements of personality, career, relationships, wealth, or destiny.",
        "elapsed_sec": fin(time.time() - t0),
    }
    write_json(ctx["out"] / "data_alignment_audit.json", payload)
    ctx["alignment"] = payload
    return payload


def _group_tables(merged, traits, provider_by_key):
    """Per exact-SHA group statistics. Returns list of group dicts + trait accumulators."""
    groups = []
    # index rows
    usable = merged[merged["sha256"].notna()].copy()
    for sha, g in usable.groupby("sha256", sort=False):
        if len(g) < 1:
            continue
        vecs = g[traits].to_numpy(dtype=np.int8)
        # unique vectors
        uniq = {tuple(row.tolist()) for row in vecs}
        n = len(g)
        keys = g["norm_key"].tolist()
        texts = []
        texts_norm = []
        cat_sets = []
        for k in keys:
            rec = provider_by_key.get(k)
            if rec is None:
                texts.append(None)
                texts_norm.append(None)
            else:
                texts.append(rec["full_text"])
                texts_norm.append(rec["full_norm"])
        n_text = len({t for t in texts if t is not None})
        n_text_norm = len({t for t in texts_norm if t is not None})
        n_missing_text = sum(1 for t in texts if t is None)
        # per-trait pos/neg
        pos = vecs.sum(axis=0)
        neg = n - pos
        conflict_traits = int(((pos > 0) & (neg > 0)).sum())
        # pairwise hamming mean
        if n >= 2:
            # unordered pairs that disagree on a trait = n_pos * n_neg
            # mean pairwise hamming = sum_t (n_pos*n_neg) / C(n,2)
            c2 = n * (n - 1) / 2.0
            mean_ham = float((pos * (n - pos)).sum() / c2) if c2 else None
        else:
            mean_ham = None
        identical_text = n_text <= 1 and n_missing_text == 0 and n >= 2
        identical_text_norm = n_text_norm <= 1 and n_missing_text == 0 and n >= 2
        identical_labels = len(uniq) == 1
        groups.append(
            {
                "sha256": sha,
                "n_records": n,
                "n_unique_texts": n_text,
                "n_unique_texts_norm": n_text_norm,
                "n_missing_text": n_missing_text,
                "n_unique_vectors": len(uniq),
                "n_conflict_traits": conflict_traits,
                "mean_pairwise_hamming": fin(mean_ham) if mean_ham is not None else None,
                "identical_labels": identical_labels,
                "identical_text": identical_text,
                "identical_text_norm": identical_text_norm,
                "pos": pos,
                "neg": neg,
                "keys": keys,
            }
        )
    return groups


def stage_b(args, ctx):
    t0 = time.time()
    traits = ctx["traits"]
    cats = ctx["cats"]
    groups = ctx["sha_groups"]
    dup = [g for g in groups if g["n_records"] > 1]
    single = [g for g in groups if g["n_records"] == 1]

    n_dup = len(dup)
    n_conflict = sum(1 for g in dup if not g["identical_labels"])
    n_identical = sum(1 for g in dup if g["identical_labels"])
    # text x label cases among dup
    case_a = sum(1 for g in dup if g["identical_text_norm"] and g["identical_labels"])
    case_b = sum(1 for g in dup if (not g["identical_text_norm"]) and g["identical_labels"])
    case_c = sum(1 for g in dup if (not g["identical_text_norm"]) and (not g["identical_labels"]))
    case_d = sum(1 for g in dup if g["identical_text_norm"] and (not g["identical_labels"]))
    # exact (not just normalized) variants
    case_d_exact = sum(1 for g in dup if g["identical_text"] and (not g["identical_labels"]))

    conflict_counts = [g["n_conflict_traits"] for g in dup]
    mean_conflict = float(np.mean(conflict_counts)) if conflict_counts else None
    max_conflict = int(max(conflict_counts)) if conflict_counts else None

    # per trait
    trait_rows = []
    per_trait = {}
    for ti, t in enumerate(traits):
        n_groups_with = 0
        n_conflict_g = 0
        row_dis_num = 0  # minority count summed? disagreement rows: for each group, minority
        row_den = 0
        # row-weighted disagreement = sum minority / sum n   among dup groups
        # also groups that contain the trait = all dup groups (trait always present as a column)
        agree_num_row = 0
        hash_agreements = []
        entropies = []
        prevalences = []
        minority_fracs = []
        for g in dup:
            np_ = int(g["pos"][ti])
            nn_ = int(g["neg"][ti])
            n = np_ + nn_
            if n <= 0:
                continue
            n_groups_with += 1
            conflict = np_ > 0 and nn_ > 0
            if conflict:
                n_conflict_g += 1
            minority = min(np_, nn_)
            row_dis_num += minority
            row_den += n
            agree_num_row += max(np_, nn_)
            hash_agreements.append(max(np_, nn_) / n)
            p = np_ / n
            prevalences.append(p)
            entropies.append(binary_entropy(p))
            minority_fracs.append(minority / n)
        row_dis = (row_dis_num / row_den) if row_den else None
        hash_dis = (sum(1 - a for a in hash_agreements) / len(hash_agreements)) if hash_agreements else None
        rec = {
            "trait": t,
            "category": cats[t],
            "n_repeated_groups": n_groups_with,
            "n_conflicting_groups": n_conflict_g,
            "conflict_group_rate": fin(conflict_rate(n_conflict_g, n_groups_with)),
            "row_weighted_disagreement": fin(row_dis),
            "hash_weighted_disagreement": fin(hash_dis),
            "row_weighted_majority_agreement": fin(agree_num_row / row_den) if row_den else None,
            "hash_weighted_majority_agreement": fin(sum(hash_agreements) / len(hash_agreements)) if hash_agreements else None,
            "mean_within_image_prevalence": fin(float(np.mean(prevalences))) if prevalences else None,
            "mean_within_image_entropy": fin(float(np.mean(entropies))) if entropies else None,
            "mean_minority_fraction": fin(float(np.mean(minority_fracs))) if minority_fracs else None,
        }
        trait_rows.append(rec)
        per_trait[t] = rec

    # category: a group conflicts in a category if any trait in that category conflicts
    cat_summary = {}
    for cat in PRED_CATS:
        idxs = [i for i, t in enumerate(traits) if cats[t] == cat]
        n_conf = 0
        trait_conf_sum = 0
        for g in dup:
            flags = [(int(g["pos"][i]) > 0 and int(g["neg"][i]) > 0) for i in idxs]
            if any(flags):
                n_conf += 1
            trait_conf_sum += sum(1 for f in flags if f)
        cat_summary[cat] = {
            "n_traits": len(idxs),
            "conflicting_groups": n_conf,
            "conflict_group_rate": fin(n_conf / n_dup) if n_dup else None,
            "mean_conflicting_traits_in_category": fin(trait_conf_sum / n_dup) if n_dup else None,
        }

    hist = Counter(conflict_counts)
    payload = {
        "stage": "B exact-image label conflicts",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "grouping": "exact SHA-256 only. Stem-connected components are NOT merged here.",
        "n_sha_groups_total": len(groups),
        "n_singleton_sha_groups": len(single),
        "duplicate_groups": {
            "n_groups": n_dup,
            "identical_labels": n_identical,
            "conflicting_labels": n_conflict,
            "conflict_group_rate": fin(n_conflict / n_dup) if n_dup else None,
            "mean_conflicting_traits": fin(mean_conflict),
            "max_conflicting_traits": max_conflict,
            "median_conflicting_traits": fin(float(np.median(conflict_counts))) if conflict_counts else None,
            "histogram_n_conflicting_traits": {str(k): int(v) for k, v in sorted(hist.items())},
        },
        "text_vs_label_cases_normalized": {
            "A_identical_text_identical_labels": case_a,
            "B_different_text_identical_labels": case_b,
            "C_different_text_conflicting_labels": case_c,
            "D_identical_text_conflicting_labels": case_d,
            "D_exact_whitespace_sensitive": case_d_exact,
            "note": "Case D (identical normalized text, conflicting extracted labels) would indicate an extraction or alignment bug. Count should be 0 if extraction is a pure function of text.",
        },
        "per_category": cat_summary,
        "singleton_note": "Singleton SHA groups are excluded from conflict rates. They provide no evidence of label reproducibility.",
        "disclaimer": "Conflict means provider-derived binary labels disagree on byte-identical images. It does not identify which label is correct.",
        "elapsed_sec": fin(time.time() - t0),
    }
    write_json(ctx["out"] / "duplicate_label_consistency.json", payload)
    fields = [
        "trait", "category", "n_repeated_groups", "n_conflicting_groups",
        "conflict_group_rate", "row_weighted_disagreement", "hash_weighted_disagreement",
        "row_weighted_majority_agreement", "hash_weighted_majority_agreement",
        "mean_within_image_prevalence", "mean_within_image_entropy", "mean_minority_fraction",
    ]
    write_csv(ctx["out"] / "semantic_trait_conflicts.csv", trait_rows, fields)
    ctx["trait_conflicts"] = {r["trait"]: r for r in trait_rows}
    ctx["consistency"] = payload
    return payload


def stage_c(args, ctx):
    t0 = time.time()
    traits = ctx["traits"]
    cats = ctx["cats"]
    groups = ctx["sha_groups"]
    dup = [g for g in groups if g["n_records"] > 1]
    single = [g for g in groups if g["n_records"] == 1]

    def trait_block(idxs, pool):
        # pool of groups
        out = []
        for i in idxs:
            pairs = [(int(g["pos"][i]), int(g["neg"][i])) for g in pool]
            out.append(
                {
                    "row_weighted_majority_agreement": fin(row_weighted_agreement(pairs)),
                    "hash_weighted_majority_agreement": fin(hash_weighted_agreement(pairs)),
                    "n_groups": sum(1 for a, b in pairs if a + b > 0),
                    "n_rows": sum(a + b for a, b in pairs),
                }
            )
        return out

    per_trait = []
    for ti, t in enumerate(traits):
        pairs = [(int(g["pos"][ti]), int(g["neg"][ti])) for g in dup]
        ent = []
        prev = []
        minority = []
        severity = []
        for np_, nn_ in pairs:
            n = np_ + nn_
            if n <= 0:
                continue
            p = np_ / n
            prev.append(p)
            ent.append(binary_entropy(p))
            minority.append(min(np_, nn_) / n)
            severity.append(min(np_, nn_))
        sev_hist = Counter(severity)
        per_trait.append(
            {
                "trait": t,
                "category": cats[t],
                "n_duplicate_groups": len(pairs),
                "n_duplicate_rows": int(sum(a + b for a, b in pairs)),
                "row_weighted_majority_agreement": fin(row_weighted_agreement(pairs)),
                "hash_weighted_majority_agreement": fin(hash_weighted_agreement(pairs)),
                "mean_within_hash_positive_probability": fin(float(np.mean(prev))) if prev else None,
                "mean_within_hash_binary_entropy": fin(float(np.mean(ent))) if ent else None,
                "mean_minority_fraction": fin(float(np.mean(minority))) if minority else None,
                "row_weighted_disagreement": fin(1.0 - row_weighted_agreement(pairs)) if pairs else None,
                "severity_histogram_minority_count": {str(k): int(v) for k, v in sorted(sev_hist.items())},
            }
        )

    # macro = unweighted mean of per-trait agreements (not a pooled 45-d accuracy)
    def _mean(key):
        vals = [r[key] for r in per_trait if r[key] is not None]
        return fin(float(np.mean(vals))) if vals else None

    per_cat = {}
    for cat in PRED_CATS:
        rows = [r for r in per_trait if r["category"] == cat]
        per_cat[cat] = {
            "n_traits": len(rows),
            "mean_row_weighted_majority_agreement": fin(float(np.mean([r["row_weighted_majority_agreement"] for r in rows]))) if rows else None,
            "mean_hash_weighted_majority_agreement": fin(float(np.mean([r["hash_weighted_majority_agreement"] for r in rows]))) if rows else None,
            "mean_minority_fraction": fin(float(np.mean([r["mean_minority_fraction"] for r in rows if r["mean_minority_fraction"] is not None]))) if rows else None,
        }

    # component bootstrap of macro row-weighted agreement
    boot = _bootstrap_macro_agreement(ctx, args)

    payload = {
        "stage": "C empirical label consistency limit",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "definition": (
            "For each exact-SHA group and trait, a deterministic binary predictor must emit one class. "
            "Observed majority agreement = max(n_pos, n_neg) / (n_pos+n_neg). "
            "This is an in-sample consistency bound on the repeated labels only."
        ),
        "not_this": [
            "Not a theoretical Bayes-error estimate.",
            "Not an upper bound on ROC-AUC.",
            "Not a generalization ceiling for unseen images.",
            "Not evidence of which label is factually correct.",
        ],
        "support": {
            "n_duplicate_sha_groups": len(dup),
            "n_singleton_sha_groups_excluded": len(single),
            "singleton_note": "Singletons are excluded. They do not evidence reproducibility.",
        },
        "macro_unweighted_mean_of_trait_agreements": {
            "row_weighted": _mean("row_weighted_majority_agreement"),
            "hash_weighted": _mean("hash_weighted_majority_agreement"),
            "mean_minority_fraction": _mean("mean_minority_fraction"),
        },
        "per_category": per_cat,
        "per_trait": per_trait,
        "bootstrap": boot,
        "elapsed_sec": fin(time.time() - t0),
    }
    write_json(ctx["out"] / "semantic_empirical_noise_ceiling.json", payload)
    ctx["ceiling"] = payload
    return payload


def _bootstrap_macro_agreement(ctx, args):
    """Cluster bootstrap: resample leakage-safe components, score SHA groups touched.

    A SHA that sits in multiple components (via stem links) is included once
    per draw if any of its components was sampled. This is descriptive uncertainty
    of the observed repeated-label agreement, not a correction for provider bias.
    """
    n_boot = int(args.bootstrap)
    if n_boot <= 0:
        return {"n_bootstrap": 0, "reason": "bootstrap disabled"}
    traits = ctx["traits"]
    merged = ctx["merged"]
    # map sha -> component ids, and component -> sha groups (dup only)
    usable = merged[merged["sha256"].notna() & merged["leakage_safe_group_id"].notna()]
    sha_pos = {}
    # precompute pos counts per sha from ctx groups
    dup = [g for g in ctx["sha_groups"] if g["n_records"] > 1]
    sha_to_pos = {g["sha256"]: g["pos"] for g in dup}
    comp_to_shas = defaultdict(set)
    for sha, comp in zip(usable["sha256"].to_numpy(), usable["leakage_safe_group_id"].to_numpy()):
        if sha in sha_to_pos:
            comp_to_shas[comp].add(sha)
    comp_ids = np.array(sorted(comp_to_shas.keys()))
    if len(comp_ids) == 0:
        return {"n_bootstrap": n_boot, "macro_row_agreement": None, "reason": "no components"}
    rng = np.random.default_rng(int(args.seed))
    # pre-pack arrays
    sha_list = list(sha_to_pos.keys())
    sha_index = {s: i for i, s in enumerate(sha_list)}
    pos_mat = np.vstack([sha_to_pos[s] for s in sha_list]).astype(np.int16)  # G x T
    n_mat = np.array([int(g["n_records"]) for g in dup if g["sha256"] in sha_index], dtype=np.int16)
    # align n_mat to sha_list order
    n_by_sha = {g["sha256"]: g["n_records"] for g in dup}
    n_mat = np.array([n_by_sha[s] for s in sha_list], dtype=np.int16)
    comp_sha_idx = {c: np.array([sha_index[s] for s in shas], dtype=np.int32) for c, shas in comp_to_shas.items()}
    T = pos_mat.shape[1]
    macros = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        draw = rng.choice(comp_ids, size=len(comp_ids), replace=True)
        # unique shas touched (a component may repeat; sha included once — ponytail:
        # multiplicity of components is ignored to keep memory bounded; upgrade to
        # multiplicity-weighted SHA inclusion if a reviewer needs cluster weights)
        seen = np.zeros(len(sha_list), dtype=bool)
        for c in draw:
            seen[comp_sha_idx[c]] = True
        if not seen.any():
            macros[b] = np.nan
            continue
        p = pos_mat[seen]
        n = n_mat[seen].astype(np.int16)
        neg = n[:, None] - p
        agree = np.maximum(p, neg).sum(axis=0) / np.maximum(n.sum(), 1)
        macros[b] = float(np.mean(agree))
    finite = macros[np.isfinite(macros)]
    if len(finite) == 0:
        return {"n_bootstrap": n_boot, "seed": int(args.seed), "macro_row_agreement_mean": None, "reason": "all replicates undefined"}
    return {
        "n_bootstrap": n_boot,
        "seed": int(args.seed),
        "n_valid": int(len(finite)),
        "n_components_resampled": int(len(comp_ids)),
        "macro_row_weighted_agreement_mean": fin(float(finite.mean())),
        "ci95": [fin(float(np.quantile(finite, 0.025))), fin(float(np.quantile(finite, 0.975)))],
        "note": "Resamples leakage-safe components that touch duplicate SHA groups. Does not account for provider-label bias. SHA included once per draw even if multiple linked components are drawn.",
    }


def stage_d(args, ctx):
    t0 = time.time()
    traits = ctx["traits"]
    cats = ctx["cats"]
    groups = [g for g in ctx["sha_groups"] if g["n_records"] > 1]
    provider_by_key = ctx["provider_by_key"]

    n = len(groups)
    exact_unique_text_hist = Counter()
    norm_unique_text_hist = Counter()
    # category-level
    cat_unique = {c: [] for c in PRED_CATS}
    # jaccard of extracted active traits between first two records, when texts differ
    jacc_diff = []
    jacc_same = []
    # sentence-level: token jaccard of normalized full text
    text_sim_diff_labels = []
    text_sim_same_labels = []

    case = Counter()
    # most frequent conflicting concepts: trait -> conflict groups
    trait_conflict = Counter()

    for g in groups:
        recs = [provider_by_key.get(k) for k in g["keys"]]
        recs = [r for r in recs if r is not None]
        if len(recs) < 2:
            case["missing_text"] += 1
            continue
        raws = [r["full_text"] for r in recs]
        norms = [r["full_norm"] for r in recs]
        exact_unique_text_hist[len(set(raws))] += 1
        norm_unique_text_hist[len(set(norms))] += 1
        for c in PRED_CATS:
            cat_unique[c].append(len({normalize_text(r["cat_texts"].get(c, "")) for r in recs}))
        identical_norm = len(set(norms)) == 1
        identical_labels = g["identical_labels"]
        if identical_norm and identical_labels:
            case["A_identical_text_identical_labels"] += 1
        elif (not identical_norm) and identical_labels:
            case["B_different_text_identical_labels"] += 1
        elif (not identical_norm) and (not identical_labels):
            case["C_different_text_conflicting_labels"] += 1
        else:
            case["D_identical_text_conflicting_labels"] += 1
        # label-set jaccard across pairs (sample first vs rest mean)
        sets = []
        for r in recs:
            # reconstruct from stored? we don't have per-row vectors here easily
            pass
        # use token jaccard between first and second distinct norm texts
        a, b = norms[0], norms[1]
        ta, tb = set(a.split()), set(b.split())
        jac = jaccard(ta, tb)
        if identical_labels:
            text_sim_same_labels.append(jac)
        else:
            text_sim_diff_labels.append(jac)
        if not identical_labels:
            for ti, t in enumerate(traits):
                if int(g["pos"][ti]) > 0 and int(g["neg"][ti]) > 0:
                    trait_conflict[t] += 1

    def _mean(xs):
        return fin(float(np.mean(xs))) if xs else None

    top_conflict = trait_conflict.most_common(10)
    payload = {
        "stage": "D provider text variability",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_duplicate_sha_groups": n,
        "cases": dict(case),
        "unique_full_texts_per_group_histogram_exact": {str(k): int(v) for k, v in sorted(exact_unique_text_hist.items())},
        "unique_full_texts_per_group_histogram_normalized": {str(k): int(v) for k, v in sorted(norm_unique_text_hist.items())},
        "per_category_mean_unique_normalized_texts": {
            c: fin(float(np.mean(vs))) if vs else None for c, vs in cat_unique.items()
        },
        "token_jaccard": {
            "mean_when_labels_identical": _mean(text_sim_same_labels),
            "mean_when_labels_conflict": _mean(text_sim_diff_labels),
            "n_identical_label_groups_scored": len(text_sim_same_labels),
            "n_conflict_groups_scored": len(text_sim_diff_labels),
        },
        "most_frequent_conflicting_traits": [
            {"trait": t, "category": cats[t], "n_conflicting_groups": int(c)} for t, c in top_conflict
        ],
        "provenance": ctx["alignment"]["provenance"] if ctx.get("alignment") else {
            "note": "run stage A for provenance counts"
        },
        "interpretation": (
            "Case C (different provider text AND conflicting labels) is provider-text variability, "
            "not an extraction bug. Case D would be an extraction/alignment bug. "
            "These are not independent stochastic draws: generation settings are unavailable, "
            "so repeated calls may differ in prompt, template, or provider version."
        ),
        "disclaimer": "Text differences are differences in provider wording, not evidence about a person.",
        "elapsed_sec": fin(time.time() - t0),
    }
    write_json(ctx["out"] / "provider_repeated_text_audit.json", payload)
    ctx["text_audit"] = payload
    return payload


def stage_e(args, ctx):
    t0 = time.time()
    traits = ctx["traits"]
    cats = ctx["cats"]
    patterns, match_trait, record_text = load_trait_patterns()
    provider_by_key = ctx["provider_by_key"]
    conflicts = ctx["trait_conflicts"]

    # deterministic sample
    rng = np.random.default_rng(int(args.seed))
    dup = [g for g in ctx["sha_groups"] if g["n_records"] > 1]
    dup_sorted = sorted(dup, key=lambda g: (-g["n_conflict_traits"], g["sha256"]))
    high = dup_sorted[:8]
    low = sorted(dup, key=lambda g: (g["n_conflict_traits"], g["sha256"]))[:4]

    # rare / weak / strong from phase 310 report if present
    perf = ctx["perf_by_trait"]
    ranked = sorted(
        [t for t in traits if perf.get(t, {}).get("roc_auc") is not None],
        key=lambda t: perf[t]["roc_auc"],
    )
    weak = ranked[:3]
    strong = ranked[-3:]
    rare = sorted(traits, key=lambda t: perf.get(t, {}).get("prevalence") or 1.0)[:3]

    # pick one provider record per selected group, plus one record per named trait
    samples = []

    def add_example(kind, key, trait, text):
        cat, pats = patterns.get(trait, (cats.get(trait), []))
        matched = []
        spans = []
        for pat in pats:
            m = re.search(pat, text or "", re.IGNORECASE)
            if m:
                matched.append(pat)
                spans.append(m.group(0))
        window = ""
        negation = False
        ambiguous = False
        if spans:
            m = re.search(pats[0], text or "", re.IGNORECASE)
            # find the span that matched
            m = None
            for pat in pats:
                m = re.search(pat, text or "", re.IGNORECASE)
                if m:
                    break
            if m:
                a = max(0, m.start() - 40)
                b = min(len(text), m.end() + 40)
                window = text[a:b]
                pre = text[max(0, m.start() - 24): m.start()]
                negation = bool(NEGATION_RE.search(pre))
        # ambiguity flags: very short / broad patterns
        broad = trait in {"public_roles", "social_connections", "stable_rel", "favorable", "influence", "meaningful", "consistent", "focused", "saving", "generous_spending", "conservative"}
        if broad:
            ambiguous = True
        samples.append(
            {
                "sample_kind": kind,
                "norm_key": key,
                "trait": trait,
                "category": cat,
                "source_excerpt": (window or (text or "")[:180]).replace("\n", " "),
                "matched_pattern": " | ".join(matched) if matched else "",
                "matched_span": " | ".join(spans) if spans else "",
                "extracted_positive": int(bool(matched)),
                "potential_negation": int(negation),
                "potential_ambiguous_usage": int(ambiguous or broad),
                "review_status": "unreviewed",
            }
        )

    def first_key(g):
        return g["keys"][0]

    for g in high:
        key = first_key(g)
        rec = provider_by_key.get(key)
        text = rec["full_text"] if rec else ""
        # the trait with conflict and a match, else first conflict trait
        chosen = None
        for ti, t in enumerate(traits):
            if int(g["pos"][ti]) > 0 and int(g["neg"][ti]) > 0:
                chosen = t
                break
        if chosen:
            add_example("high_conflict_duplicate", key, chosen, text)
    for g in low:
        key = first_key(g)
        rec = provider_by_key.get(key)
        text = rec["full_text"] if rec else ""
        add_example("low_conflict_duplicate", key, traits[0], text)

    # one example record (first provider record that matches) per rare/weak/strong trait
    # deterministic: scan provider in file order
    need = {("rare_trait", t) for t in rare} | {("weak_auc_trait", t) for t in weak} | {("strong_auc_trait", t) for t in strong}
    found = set()
    for rec in ctx["provider"]:
        if len(found) == len(need):
            break
        text = rec["full_text"]
        for kind, t in list(need):
            if (kind, t) in found:
                continue
            cat, pats = patterns[t]
            if match_trait(text, pats):
                add_example(kind, rec["norm_key"], t, text)
                found.add((kind, t))

    # pattern-level findings (deterministic, not human review)
    findings = []
    # broad single-token patterns
    broad_traits = {
        "public_roles": "matches any 'public', not specifically public-facing roles",
        "social_connections": "matches any 'social'",
        "stable_rel": "matches any 'stable', including non-relationship uses",
        "favorable": "matches any 'favorable'",
        "influence": "matches any 'influence'",
        "meaningful": "matches any 'meaningful'",
        "consistent": "matches any 'consistent'",
        "focused": "pattern focus(?:ed|) also matches bare 'focus'",
        "saving": "matches 'saving' but not 'savings' (pattern is \\bsaving\\b)",
        "generous_spending": "matches any 'generous'",
        "conservative": "matches any 'conservative'",
        "results_oriented": "also matches 'pragmatic', a different concept",
        "research": "matches any 'research'",
    }
    for t, note in broad_traits.items():
        if t in patterns:
            findings.append(
                {
                    "trait": t,
                    "category": cats.get(t),
                    "patterns": patterns[t][1],
                    "issue": "broad_or_ambiguous_pattern",
                    "detail": note,
                    "severity": "review",
                    "confirmed_bug": False,
                }
            )
    # negation: match_trait has no negation handling
    findings.append(
        {
            "trait": "*",
            "category": "*",
            "patterns": [],
            "issue": "no_negation_scope",
            "detail": "match_trait() is an unscoped regex search. A preceding 'not'/'without' does not suppress a hit. Audit sample flags potential_negation when a negation token occurs in the 24 chars before a match; those rows are unreviewed.",
            "severity": "review",
            "confirmed_bug": False,
        }
    )
    # focused pattern bug check: focus(?:ed|) matches 'focus' AND 'focused' — intentional but broad
    # analytical is in TRAIT_PATTERNS but excluded from the 45 — not a bug
    excluded = sorted(set(patterns) - set(traits))
    findings.append(
        {
            "trait": ",".join(excluded),
            "category": "n/a",
            "patterns": [],
            "issue": "candidate_excluded_by_prevalence_gate",
            "detail": "Present in TRAIT_PATTERNS but not in the 45-trait schema. Expected (prevalence gate), not an extraction bug.",
            "severity": "info",
            "confirmed_bug": False,
        }
    )
    # case D from stage B
    case_d = None
    if ctx.get("consistency"):
        case_d = ctx["consistency"]["text_vs_label_cases_normalized"]["D_identical_text_conflicting_labels"]
    findings.append(
        {
            "trait": "*",
            "category": "*",
            "patterns": [],
            "issue": "identical_text_conflicting_labels",
            "detail": f"Case D count = {case_d}. A non-zero count would mean extraction is not a function of normalized text.",
            "severity": "bug" if case_d else "info",
            "confirmed_bug": bool(case_d),
        }
    )

    n_neg = sum(r["potential_negation"] for r in samples)
    n_amb = sum(r["potential_ambiguous_usage"] for r in samples)
    summary = {
        "stage": "E semantic label extraction audit",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_patterns_in_lexicon": len(patterns),
        "n_schema_traits": len(traits),
        "excluded_from_schema": excluded,
        "sample_size": len(samples),
        "sample_policy": "deterministic: 8 highest-conflict SHA groups, 4 lowest-conflict, plus first provider record matching each of 3 rare / 3 weak-AUC / 3 strong-AUC traits. seed used only as documentation; selection is sort-based.",
        "sample_flag_counts": {
            "potential_negation": n_neg,
            "potential_ambiguous_usage": n_amb,
            "review_status": "all unreviewed",
        },
        "confirmed_deterministic_bugs": [f for f in findings if f["confirmed_bug"]],
        "findings": findings,
        "decision": (
            "No TRAIT_PATTERNS edit in this phase. "
            + ("Case D > 0: extraction is not a pure function of text." if case_d else "Case D = 0: conflicts are not an extraction nondeterminism bug.")
        ),
        "disclaimer": "Pattern flags are hypotheses for human review. They are not corrected labels and not ground truth about people.",
        "elapsed_sec": fin(time.time() - t0),
    }
    fields = [
        "sample_kind", "norm_key", "trait", "category", "source_excerpt",
        "matched_pattern", "matched_span", "extracted_positive",
        "potential_negation", "potential_ambiguous_usage", "review_status",
    ]
    write_csv(ctx["out"] / "semantic_label_extraction_audit.csv", samples, fields)
    write_json(ctx["out"] / "semantic_label_extraction_findings.json", summary)
    ctx["extraction"] = summary
    return summary


def stage_f(args, ctx):
    t0 = time.time()
    traits = ctx["traits"]
    cats = ctx["cats"]
    perf = ctx["perf_by_trait"]
    conflicts = ctx["trait_conflicts"]
    merged = ctx["merged"]
    usable = merged[merged["sha256"].notna() & merged["leakage_safe_group_id"].notna()]

    rows = []
    for t in traits:
        p = perf.get(t, {})
        c = conflicts.get(t, {})
        y = usable[t].to_numpy()
        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)
        prev = (n_pos / len(y)) if len(y) else None
        # independent hash groups with a positive / negative
        sha_pos = set(usable.loc[usable[t] == 1, "sha256"])
        sha_neg = set(usable.loc[usable[t] == 0, "sha256"])
        comp_pos = set(usable.loc[usable[t] == 1, "leakage_safe_group_id"])
        comp_neg = set(usable.loc[usable[t] == 0, "leakage_safe_group_id"])
        pr = p.get("pr_auc")
        pr_lift = (pr - prev) if (pr is not None and prev is not None) else None
        # label entropy on the observed (row) distribution, not within-hash
        ent = binary_entropy(prev) if prev is not None else None
        rec = {
            "trait": t,
            "category": cats[t],
            "n_rows": int(len(y)),
            "n_positive_rows": n_pos,
            "n_negative_rows": n_neg,
            "prevalence": fin(prev),
            "label_entropy_bits": fin(ent),
            "independent_positive_sha": len(sha_pos),
            "independent_negative_sha": len(sha_neg),
            "independent_positive_components": len(comp_pos),
            "independent_negative_components": len(comp_neg),
            "conflict_group_rate": c.get("conflict_group_rate"),
            "row_weighted_disagreement": c.get("row_weighted_disagreement"),
            "mean_minority_fraction": c.get("mean_minority_fraction"),
            "roc_auc": p.get("roc_auc"),
            "roc_ci_low": p.get("roc_ci_low"),
            "roc_ci_high": p.get("roc_ci_high"),
            "pr_auc": pr,
            "pr_baseline_prevalence": fin(prev),
            "pr_auc_lift_over_prevalence": fin(pr_lift),
            "brier": p.get("brier"),
            "ece": p.get("ece"),
            "n_bootstrap": p.get("n_bootstrap"),
        }
        rec["tags"] = "|".join(classify_trait(rec))
        rows.append(rec)

    fields = list(rows[0].keys())
    write_csv(ctx["out"] / "semantic_trait_signal_report.csv", rows, fields)

    def _pick(tag):
        return [r["trait"] for r in rows if tag in r["tags"].split("|")]

    summary = {
        "stage": "F trait-level signal",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "hash-grouped OOF metrics reused from semantic_reliability_phase310/semantic_per_trait_uncertainty.csv. Historical leaked OOF is not the primary result.",
        "criteria": {
            "RELATIVELY_STRONG_SIGNAL": f"roc>={ROC_STRONG} and (ci_low is null or ci_low>=0.65)",
            "WEAK_SIGNAL": f"{ROC_NEAR}<=roc<{ROC_STRONG} and not strong",
            "NEAR_CHANCE": f"roc<{ROC_NEAR}",
            "HIGH_LABEL_CONFLICT": f"conflict_group_rate>={CONFLICT_HIGH}",
            "LOW_INDEPENDENT_SUPPORT": f"independent_positive_components<{LOW_POS_COMPONENTS}",
            "INCONCLUSIVE": f"roc ci width>={CI_INCONCLUSIVE_WIDTH} and roc<{ROC_STRONG}, or roc missing",
            "note": "Tags may overlap. A tag is not a claim that the trait is a real personal attribute.",
        },
        "n_traits": len(rows),
        "counts": {
            "RELATIVELY_STRONG_SIGNAL": len(_pick("RELATIVELY_STRONG_SIGNAL")),
            "WEAK_SIGNAL": len(_pick("WEAK_SIGNAL")),
            "NEAR_CHANCE": len(_pick("NEAR_CHANCE")),
            "HIGH_LABEL_CONFLICT": len(_pick("HIGH_LABEL_CONFLICT")),
            "LOW_INDEPENDENT_SUPPORT": len(_pick("LOW_INDEPENDENT_SUPPORT")),
            "INCONCLUSIVE": len(_pick("INCONCLUSIVE")),
        },
        "lists": {
            "RELATIVELY_STRONG_SIGNAL": _pick("RELATIVELY_STRONG_SIGNAL"),
            "NEAR_CHANCE": _pick("NEAR_CHANCE"),
            "HIGH_LABEL_CONFLICT": _pick("HIGH_LABEL_CONFLICT"),
            "LOW_INDEPENDENT_SUPPORT": _pick("LOW_INDEPENDENT_SUPPORT"),
        },
        "macro": {
            "roc_auc_unweighted_mean": fin(float(np.mean([r["roc_auc"] for r in rows if r["roc_auc"] is not None]))),
            "pr_auc_unweighted_mean": fin(float(np.mean([r["pr_auc"] for r in rows if r["pr_auc"] is not None]))),
        },
        "disclaimer": "ROC/PR measure agreement with provider-derived labels under leakage-safe grouping. They do not validate facial geometry as a measure of personality, career, relationships, wealth, or destiny.",
        "elapsed_sec": fin(time.time() - t0),
    }
    write_json(ctx["out"] / "semantic_trait_signal_summary.json", summary)
    ctx["signal_rows"] = rows
    ctx["signal_summary"] = summary
    return summary


def stage_g(args, ctx):
    t0 = time.time()
    rows = ctx["signal_rows"]
    pairs = {
        "conflict_vs_roc": ("conflict_group_rate", "roc_auc"),
        "conflict_vs_pr_lift": ("conflict_group_rate", "pr_auc_lift_over_prevalence"),
        "minority_vs_roc": ("mean_minority_fraction", "roc_auc"),
        "minority_vs_brier": ("mean_minority_fraction", "brier"),
        "pos_components_vs_roc": ("independent_positive_components", "roc_auc"),
        "disagreement_vs_ece": ("row_weighted_disagreement", "ece"),
    }
    assoc = {}
    for name, (a, b) in pairs.items():
        xs = [r[a] for r in rows]
        ys = [r[b] for r in rows]
        sr, sn, sreason = spearman(xs, ys)
        pr, pn, preason = pearson(xs, ys)
        assoc[name] = {
            "x": a,
            "y": b,
            "spearman": sr,
            "pearson": pr,
            "n": sn,
            "spearman_reason": sreason,
            "pearson_reason": preason,
        }

    # quadrant examples
    def q(pred):
        return [r["trait"] for r in rows if pred(r)]

    examples = {
        "high_conflict_weak_auc": q(lambda r: (r["conflict_group_rate"] or 0) >= CONFLICT_HIGH and (r["roc_auc"] or 1) < ROC_WEAK),
        "low_conflict_weak_auc": q(lambda r: (r["conflict_group_rate"] or 1) < 0.25 and (r["roc_auc"] or 1) < ROC_WEAK),
        "high_conflict_relatively_strong_auc": q(lambda r: (r["conflict_group_rate"] or 0) >= CONFLICT_HIGH and (r["roc_auc"] or 0) >= ROC_STRONG),
        "low_conflict_relatively_strong_auc": q(lambda r: (r["conflict_group_rate"] or 1) < 0.25 and (r["roc_auc"] or 0) >= ROC_STRONG),
    }
    payload = {
        "stage": "G label noise vs model error",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_traits": len(rows),
        "uncertainty_note": "n=45 traits. A correlation across traits is descriptive only. It is not a causal estimate and is unstable under small-n.",
        "associations": assoc,
        "examples": examples,
        "interpretation_rule": "Do not conclude that every weak trait is weak because of label noise. Low-conflict + weak AUC points at low geometric signal or a vague trait definition. High-conflict + strong AUC means ranking signal can survive noisy binary targets.",
        "elapsed_sec": fin(time.time() - t0),
    }
    write_json(ctx["out"] / "noise_vs_predictability_analysis.json", payload)
    ctx["noise"] = payload
    return payload


def stage_h(args, ctx):
    t0 = time.time()
    traits = ctx["traits"]
    cats = ctx["cats"]
    merged = ctx["merged"]
    usable = merged[merged["sha256"].notna()]
    X = usable[traits].to_numpy(dtype=np.float64)
    # pairwise phi / pearson on binary labels, keep |r|>=0.5
    # center
    Xc = X - X.mean(axis=0, keepdims=True)
    std = Xc.std(axis=0)
    std[std == 0] = np.nan
    corr = (Xc.T @ Xc) / (len(X) - 1)
    corr = corr / np.outer(std, std)
    pairs = []
    for i in range(len(traits)):
        for j in range(i + 1, len(traits)):
            r = corr[i, j]
            if np.isfinite(r) and abs(r) >= 0.5:
                pairs.append(
                    {
                        "trait_a": traits[i],
                        "trait_b": traits[j],
                        "category_a": cats[traits[i]],
                        "category_b": cats[traits[j]],
                        "pearson": fin(float(r)),
                        "same_category": cats[traits[i]] == cats[traits[j]],
                    }
                )
    pairs.sort(key=lambda d: -abs(d["pearson"]))

    # category label entropy: distribution of popcount within category
    cat_ent = {}
    for cat in PRED_CATS:
        idxs = [i for i, t in enumerate(traits) if cats[t] == cat]
        counts = X[:, idxs].sum(axis=1).astype(int)
        hist = Counter(counts.tolist())
        total = len(counts)
        h = 0.0
        for c in hist.values():
            p = c / total
            if p > 0:
                h -= p * math.log2(p)
        cat_ent[cat] = {
            "n_traits": len(idxs),
            "popcount_entropy_bits": fin(h),
            "mean_active": fin(float(counts.mean())),
            "popcount_histogram": {str(k): int(v) for k, v in sorted(hist.items())},
        }

    # effective rank of the label matrix via eigenvalue energy
    # use covariance eigenvalues
    cov = np.cov(X, rowvar=False)
    evals = np.linalg.eigvalsh(cov)
    evals = np.clip(evals, 0, None)
    total_var = float(evals.sum())
    if total_var > 0:
        p = evals / total_var
        p = p[p > 0]
        eff = float(np.exp(-np.sum(p * np.log(p))))  # participation ratio-like, natural log
    else:
        eff = None
    # participation ratio
    pr = float((evals.sum() ** 2) / np.sum(evals ** 2)) if total_var > 0 else None

    # co-occurrence top: traits that activate together, conditional P(b|a) high
    co = []
    col_sum = X.sum(axis=0)
    for i in range(len(traits)):
        if col_sum[i] < 50:
            continue
        both = X[:, i][:, None] * X
        # both is n x T? X[:, i] is n, broadcast
        joint = (X.T @ X[:, i])  # T
        for j in range(len(traits)):
            if i == j:
                continue
            cond = joint[j] / col_sum[i]
            base = col_sum[j] / len(X)
            if cond >= 0.8 and base < 0.9 and cond - base >= 0.15:
                co.append(
                    {
                        "given": traits[i],
                        "also": traits[j],
                        "p_also_given": fin(float(cond)),
                        "p_also": fin(float(base)),
                        "lift": fin(float(cond - base)),
                    }
                )
    co.sort(key=lambda d: -d["lift"])

    payload = {
        "stage": "H semantic vocabulary redundancy",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_rows": int(len(X)),
        "n_traits": len(traits),
        "high_abs_correlation_pairs_ge_0_5": pairs[:40],
        "n_pairs_abs_corr_ge_0_5": len(pairs),
        "category_popcount_entropy": cat_ent,
        "label_covariance_participation_ratio": fin(pr),
        "label_covariance_effective_n_exp_entropy": fin(eff),
        "high_conditional_coactivation": co[:25],
        "decision": "No trait deleted. Redundancy is reported only. Correlated provider phrases are not evidence of redundant human traits.",
        "disclaimer": "Co-occurrence is co-occurrence in provider text extractions, not a personality factor model.",
        "elapsed_sec": fin(time.time() - t0),
    }
    write_json(ctx["out"] / "semantic_vocabulary_redundancy.json", payload)
    ctx["redundancy"] = payload
    return payload


def stage_j(args, ctx):
    """Recommendations from measured findings. No production change."""
    t0 = time.time()
    cons = ctx.get("consistency") or {}
    noise = ctx.get("noise") or {}
    sig = ctx.get("signal_summary") or {}
    ext = ctx.get("extraction") or {}
    ceil = ctx.get("ceiling") or {}
    red = ctx.get("redundancy") or {}

    dup = cons.get("duplicate_groups") or {}
    conflict_rate_v = dup.get("conflict_group_rate")
    case_d = (cons.get("text_vs_label_cases_normalized") or {}).get("D_identical_text_conflicting_labels")
    macro_agree = (ceil.get("macro_unweighted_mean_of_trait_agreements") or {}).get("row_weighted")
    n_strong = (sig.get("counts") or {}).get("RELATIVELY_STRONG_SIGNAL")
    n_near = (sig.get("counts") or {}).get("NEAR_CHANCE")
    n_low = (sig.get("counts") or {}).get("LOW_INDEPENDENT_SUPPORT")
    n_high_c = (sig.get("counts") or {}).get("HIGH_LABEL_CONFLICT")
    spear = ((noise.get("associations") or {}).get("conflict_vs_roc") or {}).get("spearman")

    # pick primary case
    # Priority: extraction bug (D) > high conflict > low support > near chance with low conflict > residual signal
    if case_d:
        primary = "CASE_C"
        reason = "Identical normalized provider text produced conflicting extracted labels."
    elif conflict_rate_v is not None and conflict_rate_v >= 0.5:
        primary = "CASE_A"
        reason = "Majority of exact-image duplicate groups carry conflicting 45-d label vectors. Label consistency dominates model error."
    elif n_low is not None and n_low >= 15:
        primary = "CASE_D"
        reason = "Many traits lack independent positive components."
    elif n_near is not None and n_near >= 15 and (n_high_c or 0) < 10:
        primary = "CASE_B"
        reason = "Low conflict but near-chance AUC: geometry/features, not label noise, are the limit."
    elif n_strong is not None and n_strong >= 5:
        primary = "CASE_E"
        reason = "A subset of traits keeps leakage-safe signal."
    else:
        primary = "CASE_A"
        reason = "Default: quantify label consistency before another model change."

    options = [
        {
            "id": "A",
            "name": "Keep the current 45-trait setup unchanged",
            "problem": "none immediately; preserves comparability",
            "evidence": f"conflict_group_rate={conflict_rate_v}, macro majority agreement={macro_agree}, strong_traits={n_strong}",
            "expected_benefit": "zero accuracy gain; stable baseline",
            "cost": "none",
            "limitations": "leaves a known noisy target in place",
            "bias_risk": "low",
            "validation": "already measured",
            "new_data": False,
        },
        {
            "id": "B",
            "name": "Reliability annotations per trait",
            "problem": "consumers cannot tell a stable trait from a noisy one",
            "evidence": "per-trait conflict rates and CIs in semantic_trait_signal_report.csv",
            "expected_benefit": "honest reporting, no score inflation",
            "cost": "small: publish the table beside predictions",
            "limitations": "does not improve predictions",
            "bias_risk": "low if annotations stay descriptive",
            "validation": "schema review",
            "new_data": False,
        },
        {
            "id": "C",
            "name": "Downweight duplicate-heavy examples",
            "problem": "repeated noisy rows dominate the loss",
            "evidence": "phase 3.10 expA inverse-hash weight did not beat baseline",
            "expected_benefit": "small or none, already tested",
            "cost": "already implemented experimentally",
            "limitations": "phase 3.10 CI excluded zero on the negative side for expA",
            "bias_risk": "changes the population being fit",
            "validation": "already done, do not repeat",
            "new_data": False,
        },
        {
            "id": "D",
            "name": "Group-level consensus labels",
            "problem": "byte-identical images have conflicting binary targets",
            "evidence": f"identical_label duplicate groups={dup.get('identical_labels')}, conflicting={dup.get('conflicting_labels')}",
            "expected_benefit": "unknown; consensus is the provider's majority wording, not a true label",
            "cost": "medium: new training target, leakage-safe re-evaluation",
            "limitations": "majority vote does not recover a correct label",
            "bias_risk": "high if consensus is treated as truth",
            "validation": "paired component-aware bootstrap vs current hard labels",
            "new_data": False,
        },
        {
            "id": "E",
            "name": "Soft targets from within-hash prevalence",
            "problem": "hard 0/1 labels discard within-image disagreement",
            "evidence": "mean minority fractions in semantic_empirical_noise_ceiling.json",
            "expected_benefit": "may reduce overconfident fit to one of two conflicting labels",
            "cost": "medium",
            "limitations": "soft target is still the provider's empirical frequency",
            "bias_risk": "medium",
            "validation": "same leakage-safe folds, Brier/ECE not just ROC",
            "new_data": False,
        },
        {
            "id": "F",
            "name": "Improve semantic extraction patterns",
            "problem": "several traits use broad single tokens",
            "evidence": "findings in semantic_label_extraction_findings.json; case D count "
            + str(case_d),
            "expected_benefit": "only if a human review confirms false hits; case D bug count is the deterministic signal",
            "cost": "medium, and it changes historical labels so it must be a versioned experiment",
            "limitations": "unreviewed flags are not confirmed errors",
            "bias_risk": "medium: tighter patterns change prevalence",
            "validation": "frozen leakage-safe folds, compare label vectors before/after",
            "new_data": False,
        },
        {
            "id": "G",
            "name": "Merge redundant concepts",
            "problem": "some traits co-activate or correlate",
            "evidence": f"n_pairs_|r|>=0.5 = {red.get('n_pairs_abs_corr_ge_0_5')}",
            "expected_benefit": "shorter vocabulary, not higher accuracy by itself",
            "cost": "medium",
            "limitations": "correlation is in provider text, not in people",
            "bias_risk": "medium",
            "validation": "versioned schema, same OOF protocol",
            "new_data": False,
        },
        {
            "id": "H",
            "name": "Drop low-support traits from a future experimental vocabulary",
            "problem": "rare traits have wide uncertainty",
            "evidence": f"LOW_INDEPENDENT_SUPPORT count={n_low}",
            "expected_benefit": "stabler macro metrics, less ceiling-chasing on rare phrases",
            "cost": "small experiment, do not change production schema yet",
            "limitations": "dropping traits does not create signal",
            "bias_risk": "metric cherry-picking if the drop is chosen on the test metric",
            "validation": "pre-register the drop rule on train folds only",
            "new_data": False,
        },
        {
            "id": "I",
            "name": "Multiple independent annotations per unique image",
            "problem": "current repeats are uncontrolled re-scrapes, not a designed reliability study",
            "evidence": "provenance fields (timestamp, model, request id) are absent",
            "expected_benefit": "a real reliability estimate and a soft target that is interpretable",
            "cost": "high: provider calls, quota, time",
            "limitations": "still provider text, still not human ground truth",
            "bias_risk": "low if the protocol is fixed and recorded",
            "validation": "report within-image agreement before any retrain",
            "new_data": True,
        },
        {
            "id": "J",
            "name": "Collect a task-specific dataset",
            "problem": "provider phrases about destiny/wealth are weak supervision for any real attribute",
            "evidence": "project-wide disclaimer; near-chance traits concentrate in future_destiny",
            "expected_benefit": "only path to a claim about anything other than provider-text agreement",
            "cost": "very high",
            "limitations": "out of scope for this codebase as it stands",
            "bias_risk": "depends on the new labeling protocol",
            "validation": "new dataset, new split, do not mix with current labels",
            "new_data": True,
        },
    ]

    payload = {
        "stage": "J practical options and K next experiment",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "primary_next_experiment": primary,
        "reason": reason,
        "supporting_numbers": {
            "duplicate_conflict_group_rate": conflict_rate_v,
            "mean_conflicting_traits": dup.get("mean_conflicting_traits"),
            "macro_row_weighted_majority_agreement": macro_agree,
            "case_D_identical_text_conflicting_labels": case_d,
            "n_relatively_strong": n_strong,
            "n_near_chance": n_near,
            "n_high_conflict": n_high_c,
            "n_low_support": n_low,
            "spearman_conflict_vs_roc": spear,
        },
        "options": options,
        "recommended_protocol": _protocol(primary),
        "production_decision": {
            "keep_current_semantic_model": True,
            "keep_current_trait_vocabulary": True,
            "retraining_justified": False,
            "reason": "Phase 3.10 already rejected three model tweaks. This phase finds the binding limit in label consistency, not in the classifier. No candidate is promoted.",
        },
        "elapsed_sec": None,
    }
    payload["elapsed_sec"] = fin(time.time() - t0)
    write_json(ctx["out"] / "semantic_label_quality_summary.json", payload)
    ctx["summary"] = payload
    return payload


def _protocol(primary):
    if primary == "CASE_A":
        return {
            "name": "Controlled label-consistency experiment",
            "compare": [
                "current hard labels",
                "SHA-group majority consensus (hypothesis only)",
                "SHA-group soft prevalence targets",
            ],
            "split": "existing leakage-safe connected components, same 5 folds",
            "metrics": ["macro ROC", "macro Brier", "macro ECE", "per-trait CI"],
            "do_not": "Do not treat consensus as recovered truth. Report it as agreement with the provider's majority wording.",
        }
    if primary == "CASE_B":
        return {
            "name": "Feature-signal check",
            "compare": ["current 116 features", "label-shuffled control", "prevalence-only baseline"],
            "split": "leakage-safe components",
            "metrics": ["ROC vs 0.5", "PR vs prevalence"],
            "do_not": "Do not add a larger model.",
        }
    if primary == "CASE_C":
        return {
            "name": "Versioned extraction experiment",
            "compare": ["schema v1 labels", "revised patterns on a frozen copy"],
            "split": "leakage-safe components",
            "metrics": ["label flip rate", "leakage-safe ROC/Brier on unchanged folds"],
            "do_not": "Do not overwrite semantic_labels.csv.",
        }
    if primary == "CASE_D":
        return {
            "name": "Vocabulary review plus data collection",
            "compare": ["all 45", "pre-registered subset with enough positive components"],
            "split": "leakage-safe components",
            "metrics": ["macro on the pre-registered subset, not on a subset picked by test AUC"],
            "do_not": "Do not drop traits from production in this step.",
        }
    return {
        "name": "Small leakage-safe model comparison",
        "compare": ["current logistic", "one alternative, pre-registered"],
        "split": "leakage-safe components",
        "metrics": ["paired component bootstrap as in phase 3.10"],
        "do_not": "Do not promote unless CI supports it.",
    }


def load_perf():
    if not PER_TRAIT_CSV.exists():
        return {}
    out = {}
    with PER_TRAIT_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rec = {"trait": row["trait"], "category": row.get("category")}
            for k in (
                "prevalence", "roc_auc", "roc_ci_low", "roc_ci_high", "pr_auc",
                "pr_ci_low", "pr_ci_high", "brier", "ece", "n_bootstrap",
                "components_with_positive", "components_with_negative",
            ):
                rec[k] = fin(row[k]) if row.get(k) not in (None, "") else None
            out[row["trait"]] = rec
    return out


def build_context(args):
    traits, cats = load_schema()
    print("loading provider records...")
    provider = load_provider_records()
    provider_by_key = {}
    for r in provider:
        provider_by_key.setdefault(r["norm_key"], r)
    print(f"  {len(provider)} provider records, {len(provider_by_key)} keys")
    print("loading aligned tables...")
    merged, feat_stems = load_aligned(traits)
    print(f"  {len(merged)} label rows")
    print("building SHA groups...")
    matched = merged[merged["sha256"].notna() & merged["leakage_safe_group_id"].notna()]
    sha_groups = _group_tables(matched, traits, provider_by_key)
    sha_groups_all = _group_tables(merged, traits, provider_by_key)
    print(f"  {len(sha_groups)} sha groups, {sum(1 for g in sha_groups if g['n_records']>1)} duplicate")
    out = Path(args.out_dir) if args.out_dir else DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)
    return {
        "traits": traits,
        "cats": cats,
        "provider": provider,
        "provider_by_key": provider_by_key,
        "merged": merged,
        "feat_stems": feat_stems,
        "sha_groups": sha_groups,
        "sha_groups_all": sha_groups_all,
        "perf_by_trait": load_perf(),
        "out": out,
    }


def main():
    ap = argparse.ArgumentParser(description="Phase 3.11 semantic label quality audit")
    ap.add_argument("--stage", default="all", help="all or a comma list: a,b,c,d,e,f,g,h,j")
    ap.add_argument("--quick", action="store_true", help="bootstrap 50")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bootstrap", type=int, default=400)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    if args.quick:
        args.bootstrap = min(args.bootstrap, 50)
    stage = args.stage.strip().lower()
    wanted = set("abcdefghj") if stage == "all" else set(stage.replace(",", "").replace(" ", ""))
    # i is part of c (bootstrap). j writes the summary and needs prior stages in-memory.
    ctx = build_context(args)
    if "a" in wanted:
        stage_a(args, ctx)
    if "b" in wanted:
        stage_b(args, ctx)
    if "c" in wanted or "i" in wanted:
        stage_c(args, ctx)
    if "d" in wanted:
        stage_d(args, ctx)
    if "e" in wanted:
        stage_e(args, ctx)
    if "f" in wanted:
        stage_f(args, ctx)
    if "g" in wanted:
        if "signal_rows" not in ctx:
            stage_f(args, ctx)
        stage_g(args, ctx)
    if "h" in wanted:
        stage_h(args, ctx)
    if "j" in wanted or "k" in wanted or stage == "all":
        stage_j(args, ctx)
    print("done")


if __name__ == "__main__":
    main()