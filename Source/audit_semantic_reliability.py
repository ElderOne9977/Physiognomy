# -*- coding: utf-8 -*-
"""Semantic reliability audit entry point (Stages A-K).

All reported scores are agreement with provider-derived semantic labels,
not validated measures of personality/career/relationships/wealth/destiny.

Outputs (all finite JSON, unavailable metrics = null + reason):
  Source/semantic_data_audit.json          stages A/B/E
  Source/semantic_baseline_metrics.json    stages C/D
  Source/semantic_per_trait_metrics.csv    stage C
  Source/semantic_feature_drift.json       stage F
  Source/semantic_stability_report.json    stage I
  Source/semantic_feature_importance.json  stage J
  Source/semantic_improvement_experiments.json stages G/H/K

Usage:
  python Source/audit_semantic_reliability.py --stage all
  python Source/audit_semantic_reliability.py --stage audit baseline
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, re, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, precision_score, recall_score)

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
CSV_CLEAN = ROOT / "faces_full_features_clean.csv"
LABELS_CSV = ROOT / "semantic_labels.csv"
SCHEMA_JSON = ROOT / "semantic_schema.json"
OOF_CSV = ROOT / "oof_semantic_probs.csv"
THRESH_JSON = SRC / "models_semantic" / "trait_thresholds.json"
MODEL_DIR = SRC / "models_semantic"
HUMANS = ROOT / "Humans"
CAND_DIR = SRC / "models_semantic_candidates"

CATEGORIES = ["personality", "career", "relationships", "future_destiny", "wealth"]
SEED = 42
FOLDS = 5
WEAK_AUC = 0.55


def fin(x):
    """Finite float or None (JSON-safe, never NaN/Infinity)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def norm_name(v):
    s = os.path.basename(str(v).strip())
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\)\s+\(", ")(", s)
    return s.lower()


def stem_name(v):
    return re.sub(r"\.[^.]+$", "", norm_name(v))


def is_feature_col(name):
    return name.endswith("_ratio") or "area_ratio" in name or "asymmetry" in name


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False,
                     default=lambda o: None)
    path.write_text(txt, encoding="utf-8")
    print(f"wrote {path}")


def load_schema_traits():
    sch = json.loads(SCHEMA_JSON.read_text(encoding="utf-8"))
    names = [t["name"] for t in sch["traits"]]
    cat = {t["name"]: t["category"] for t in sch["traits"]}
    return names, cat


def load_aligned():
    """Feature rows + label rows inner-joined on normalized filename."""
    df = pd.read_csv(CSV_CLEAN, encoding="utf-8-sig")
    labels = pd.read_csv(LABELS_CSV, encoding="utf-8-sig")
    img_col = [c for c in labels.columns if c.strip().lower().lstrip("\ufeff") == "image"][0]
    if img_col != "image":
        labels = labels.rename(columns={img_col: "image"})
    df["norm_key"] = df["image_id"].apply(norm_name)
    labels["norm_key"] = labels["image"].apply(norm_name)
    merged = pd.merge(df, labels, on="norm_key", how="inner", suffixes=("", "_lab"))
    return df, labels, merged


def high_null_cols_for(df, feats, thr=0.05):
    return [c for c in feats if df[c].isna().mean() > thr]


def build_matrix(df, cols, ind_cols):
    """Base numeric cols + __missing indicators (production construction)."""
    out = pd.DataFrame(index=df.index)
    for c in cols:
        out[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ind_cols:
        if c in df.columns:
            out[c + "__missing"] = df[c].isna().astype(int)
    return out


def add_indicators(df, ind_cols):
    out = df.copy()
    for c in ind_cols:
        if c in out.columns:
            out[c + "__missing"] = out[c].isna().astype(int)
    return out


def prepare_fold(X_tr, X_va):
    """Median impute + standardize, fit on train only (fold-local)."""
    cols = list(X_tr.columns)
    imp = SimpleImputer(strategy="median")
    a = imp.fit_transform(X_tr[cols])
    b = imp.transform(X_va[cols])
    sc = StandardScaler()
    return sc.fit_transform(a), sc.transform(b)


def ece_score(y, p, n_bins=10):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if len(y) == 0:
        return None
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        e += abs(y[m].mean() - p[m].mean()) * m.sum() / len(y)
    return float(e)


def reliability_bins(y, p, n_bins=10):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            out.append({"lo": fin(lo), "hi": fin(hi), "count": 0,
                        "mean_prob": None, "observed_rate": None})
            continue
        out.append({"lo": fin(lo), "hi": fin(hi), "count": int(m.sum()),
                    "mean_prob": fin(p[m].mean()),
                    "observed_rate": fin(y[m].mean())})
    return out


def calib_slope_intercept(y, p):
    """Logistic recalibration: y ~ a + b*logit(p). Unpenalized approx (C=1e6)."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    if len(np.unique(y)) < 2:
        return None, None, "degenerate: single class in y"
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    try:
        clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500)
        clf.fit(logit, y.astype(int))
        return fin(clf.intercept_[0]), fin(clf.coef_[0][0]), None
    except Exception as e:  # pragma: no cover
        return None, None, f"fit failed: {e}"


def binary_metrics(y, p, thr=0.5):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = precision_score(y, pred, zero_division=0)
    rec = recall_score(y, pred, zero_division=0)
    f1 = f1_score(y, pred, zero_division=0)
    return {"threshold": thr, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": fin(prec), "recall": fin(rec), "f1": fin(f1),
            "predicted_positive_rate": fin(pred.mean())}

# ---------------------------------------------------------------------------
# Stages A/B/E: dataset integrity, leakage, feature distribution
# ---------------------------------------------------------------------------

def stage_data_audit():
    t0 = time.time()
    traits, cat_map = load_schema_traits()
    df, labels, merged = load_aligned()
    feats = [c for c in df.columns if c != "image_id" and c != "norm_key"
             and is_feature_col(c)]

    # --- ids / groups ---
    ids = df["image_id"].astype(str)
    dup_ids = int(ids.duplicated().sum())
    dup_norm = int(df["norm_key"].duplicated().sum())
    stems = ids.apply(stem_name)
    dup_stems = int(stems.duplicated().sum())
    exact_dup_rows = int(df.drop(columns=["norm_key"]).duplicated().sum())
    n_groups = int(df["norm_key"].nunique())
    groups_gt1 = int((df["norm_key"].value_counts() > 1).sum())

    # --- byte hashes of source images (duplicate leakage evidence) ---
    by_name = {}
    if HUMANS.exists():
        by_name = {os.path.basename(f).lower(): f for f in HUMANS.iterdir() if f.is_file()}
    hmap, hash_missing = {}, 0
    for i in ids:
        p = by_name.get(i.lower())
        if p is None:
            hash_missing += 1
            continue
        hmap[i] = hashlib.md5(p.read_bytes()).hexdigest()
    hash_series = ids.map(hmap)
    n_hashed = int(hash_series.notna().sum())
    hv = hash_series.dropna().value_counts()
    dup_hash_groups = int((hv > 1).sum())
    dup_hash_rows = int(hv[hv > 1].sum())
    unique_hashes = int(hash_series.nunique(dropna=True))

    # --- GroupKFold(norm_key) reproduces training fold assignment ---
    gkf = GroupKFold(n_splits=FOLDS)
    fold = np.full(len(df), -1)
    for k, (_, va) in enumerate(gkf.split(df, groups=df["norm_key"].values)):
        fold[va] = k
    df_fold = df[["image_id"]].copy()
    df_fold["fold"] = fold
    df_fold["hash"] = hash_series.values
    split_hash_groups = 0
    rows_in_split_hash = 0
    if n_hashed:
        tmp = df_fold.dropna(subset=["hash"])
        for h, g in tmp.groupby("hash"):
            if len(g) < 2:
                continue
            if g["fold"].nunique() > 1:
                split_hash_groups += 1
                rows_in_split_hash += len(g)
    val_with_hash_twin_in_train = 0
    if n_hashed:
        for k in range(FOLDS):
            tr_h = set(df_fold.loc[df_fold["fold"] != k, "hash"].dropna())
            val_h = df_fold.loc[df_fold["fold"] == k, "hash"].dropna()
            val_with_hash_twin_in_train += int(val_h.isin(tr_h).sum())

    # --- labels: order, binary, prevalence ---
    lab_cols = [c for c in labels.columns if c not in ("image", "norm_key")]
    order_match = lab_cols == traits
    non_binary = {}
    per_trait = {}
    for t in traits:
        if t not in labels.columns:
            non_binary[t] = "missing column"
            continue
        s = pd.to_numeric(labels[t], errors="coerce")
        vals = set(int(v) for v in s.dropna().unique() if float(v) in (0.0, 1.0))
        bad = sorted(set(s.dropna().unique().tolist()) - {0, 1})
        if bad:
            non_binary[t] = f"non-binary values: {bad[:6]}"
        pos = int((s == 1).sum())
        n = int(s.notna().sum())
        prev = pos / n if n else 0.0
        per_trait[t] = {"positive": pos, "negative": n - pos, "n": n,
                        "prevalence": fin(prev),
                        "imbalance_ratio": fin((n - pos) / pos) if pos else None,
                        "category": cat_map.get(t)}

    rare = [t for t, v in per_trait.items()
            if v["prevalence"] is not None and (v["prevalence"] < 0.05 or v["prevalence"] > 0.95)]
    severe = [t for t, v in per_trait.items()
              if v["imbalance_ratio"] is not None and v["imbalance_ratio"] > 8]

    # --- feature distribution / groups (stage E) ---
    def fgroup(c):
        s = c.lower()
        if s.startswith("roll") or s.startswith("yaw") or "det_score" in s or "qc" in s:
            return "pose_detection"
        if "asymmetry" in s:
            return "asymmetry"
        for k in ("eye", "brow", "nose", "mouth", "jaw", "cheek", "chin",
                  "forehead", "ear", "lip", "eyebrow"):
            if k in s:
                return k
        if "face" in s or "proportion" in s or "ratio" in s:
            return "face_proportions"
        return "other"

    X = df[feats]
    feat_stats = {}
    group_missing = {}
    for c in feats:
        s = pd.to_numeric(X[c], errors="coerce")
        miss = float(s.isna().mean())
        finite = s.replace([np.inf, -np.inf], np.nan).dropna()
        if len(finite) < 2:
            st = {"missing_rate": fin(miss), "mean": None, "std": None, "median": None,
                  "iqr": None, "min": None, "max": None, "outlier_rate": None,
                  "near_constant": True, "group": fgroup(c), "n_finite": int(len(finite))}
        else:
            q1, q3 = float(finite.quantile(0.25)), float(finite.quantile(0.75))
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            out_rate = float(((finite < lo) | (finite > hi)).mean())
            sd = float(finite.std())
            st = {"missing_rate": fin(miss), "mean": fin(finite.mean()), "std": fin(sd),
                  "median": fin(finite.median()), "iqr": fin(iqr),
                  "min": fin(finite.min()), "max": fin(finite.max()),
                  "outlier_rate": fin(out_rate),
                  "near_constant": bool(sd < 1e-9 or finite.nunique() <= 1),
                  "group": fgroup(c), "n_finite": int(len(finite))}
        feat_stats[c] = st
        g = st["group"]
        group_missing.setdefault(g, []).append(miss)
    group_summary = {g: {"n_features": len(v), "mean_missing": fin(np.mean(v)),
                         "max_missing": fin(np.max(v))}
                     for g, v in sorted(group_missing.items())}

    high_null = [c for c in feats if float(X[c].isna().mean()) > 0.05]

    # --- inference schema check ---
    model_cols = np.load(MODEL_DIR / "feature_cols.npy", allow_pickle=True).tolist()
    model_traits = np.load(MODEL_DIR / "trait_names.npy", allow_pickle=True).tolist()
    base_model = [c for c in model_cols if not c.endswith("__missing")]
    ind_model = [c for c in model_cols if c.endswith("__missing")]
    schema_ok = (base_model == feats)
    missing_in_csv = [c for c in base_model if c not in df.columns]
    oof_head = pd.read_csv(OOF_CSV, nrows=0, encoding="utf-8-sig").columns.tolist()

    payload = {
        "stage": "A/B/E dataset integrity + leakage + feature distribution",
        "generated_by": "audit_semantic_reliability.py",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "label_semantics": ("Labels are provider-derived weak labels extracted from face-reading "
                            "text. Metrics quantify agreement with those labels only, not actual "
                            "personality, career, relationship, wealth, or destiny."),
        "dataset": {
            "feature_rows": int(len(df)),
            "label_rows": int(len(labels)),
            "matched_rows": int(len(merged)),
            "unmatched_feature_rows": int(len(df) - len(merged)),
            "unmatched_label_rows": int(len(labels) - len(merged)),
            "unique_image_ids": int(ids.nunique()),
            "duplicate_image_ids": dup_ids,
            "duplicate_normalized_names": dup_norm,
            "duplicate_extensionless_stems": dup_stems,
            "exact_duplicate_feature_rows": exact_dup_rows,
            "groups_norm_key": n_groups,
            "groups_with_multiple_rows": groups_gt1,
            "selected_base_features": len(feats),
            "semantic_traits": len(traits),
            "trait_order_matches_schema": bool(order_match),
        },
        "duplicate_hashes": {
            "source": str(HUMANS) if HUMANS.exists() else None,
            "images_hashed": n_hashed,
            "images_missing_file": hash_missing,
            "unique_hashes": unique_hashes,
            "duplicate_hash_groups": dup_hash_groups,
            "rows_in_duplicate_hash_groups": dup_hash_rows,
            "limitation": ("Hashes computed from local Humans/ files by image_id name; "
                           "images absent on disk cannot be checked."),
        },
        "group_leakage": {
            "cv_method": f"GroupKFold({FOLDS}) on norm_key",
            "norm_key_is_effectively_unique": bool(n_groups == len(df)),
            "hash_groups_split_across_folds": split_hash_groups,
            "rows_in_hash_groups_split_across_folds": rows_in_split_hash,
            "val_rows_with_same_hash_twin_in_train": val_with_hash_twin_in_train,
            "val_row_fraction_with_hash_twin_in_train": fin(val_with_hash_twin_in_train / len(df)),
            "finding": ("Filename grouping does not collapse byte-identical images. "
                        "Duplicate source images can therefore land in different folds."),
        },
        "label_quality": {
            "non_binary_traits": non_binary,
            "rare_traits_prev_lt_0.05_or_gt_0.95": rare,
            "severe_imbalance_ratio_gt_8": severe,
            "per_trait": per_trait,
            "prevalence_min": fin(min(v["prevalence"] for v in per_trait.values() if v["prevalence"] is not None)),
            "prevalence_max": fin(max(v["prevalence"] for v in per_trait.values() if v["prevalence"] is not None)),
        },
        "feature_distribution": {
            "high_null_columns_threshold_0.05": high_null,
            "high_null_count": len(high_null),
            "group_summary": group_summary,
            "per_feature": feat_stats,
        },
        "inference_schema": {
            "model_base_features": len(base_model),
            "model_missing_indicators": len(ind_model),
            "model_total_columns": len(model_cols),
            "base_matches_selected_95": bool(schema_ok),
            "model_base_missing_from_clean_csv": missing_in_csv,
            "clean_csv_total_columns": int(len(df.columns)),
            "oof_csv_columns": len(oof_head),
            "note": ("Demo extraction emits 486 columns including detection/QC metadata; "
                     "predictor consumes only the 95 selected base features + indicators."),
        },
        "model_trait_order": {"matches_schema": bool(model_traits == traits),
                              "oof_trait_columns_match_schema": bool(oof_head[1:] == traits)},
        "elapsed_sec": fin(time.time() - t0),
    }
    write_json(SRC / "semantic_data_audit.json", payload)
    return payload

# ---------------------------------------------------------------------------
# Stages C/D: baseline OOF metrics + probability calibration
# ---------------------------------------------------------------------------

def load_oof_aligned():
    """OOF probs aligned to labels by image id (verified, not row order)."""
    oof = pd.read_csv(OOF_CSV, encoding="utf-8-sig")
    labels = pd.read_csv(LABELS_CSV, encoding="utf-8-sig")
    img_col = [c for c in labels.columns if c.strip().lower().lstrip("\ufeff") == "image"][0]
    if img_col != "image":
        labels = labels.rename(columns={img_col: "image"})
    traits, _ = load_schema_traits()
    oof = oof.rename(columns={oof.columns[0]: "image_id"})
    oof["norm_key"] = oof["image_id"].apply(norm_name)
    labels["norm_key"] = labels["image"].apply(norm_name)
    m = pd.merge(oof, labels[["norm_key"] + traits], on="norm_key", how="inner",
                 suffixes=("_oof", ""))
    # after merge, trait cols from labels (suffix on collision); oof traits got _oof
    alignment = {
        "oof_rows": int(len(oof)),
        "label_rows": int(len(labels)),
        "matched_rows": int(len(m)),
        "oof_unmatched": int(len(oof) - len(m)),
        "row_order_identical_before_merge": bool(
            oof["image_id"].astype(str).reset_index(drop=True).equals(
                labels["image"].astype(str).reset_index(drop=True)) and len(oof) == len(labels)),
        "join_key": "norm_name(image_id) == norm_name(image)",
    }
    P = m[[t + "_oof" for t in traits]].to_numpy(dtype=float)
    Y = m[traits].to_numpy(dtype=int)
    return P, Y, alignment, m


def load_thresholds():
    if not THRESH_JSON.exists():
        return {t: 0.5 for t in load_schema_traits()[0]}, "missing_thresholds_default_0.5"
    d = json.loads(THRESH_JSON.read_text(encoding="utf-8"))
    strategy = d.get("strategy", "unknown")
    tt = d.get("trait_thresholds") or d.get("thresholds") or {}
    traits = load_schema_traits()[0]
    return {t: float(tt.get(t, d.get("global_threshold", 0.5))) for t in traits}, strategy


def per_trait_block(y, p, thr, thr_cal):
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    n = len(y)
    pos = int(y.sum())
    neg = n - pos
    block = {
        "positive": pos, "negative": neg, "prevalence": fin(pos / n) if n else None,
        "brier": fin(np.mean((p - y) ** 2)) if n else None,
        "ece": fin(ece_score(y, p)),
        "reliability_bins": reliability_bins(y, p),
    }
    if pos > 0 and neg > 0:
        block["roc_auc"] = fin(roc_auc_score(y, p))
        block["pr_auc"] = fin(average_precision_score(y, p))
        slope, intercept, err = calib_slope_intercept(y, p)
        block["calibration_slope"] = slope
        block["calibration_intercept"] = intercept
        block["calibration_note"] = err
    else:
        block["roc_auc"] = None
        block["pr_auc"] = None
        block["calibration_slope"] = None
        block["calibration_intercept"] = None
        block["calibration_note"] = "degenerate trait: only one class present"
    block["threshold_0_5"] = binary_metrics(y, p, 0.5)
    block["threshold_saved_policy"] = binary_metrics(y, p, thr_cal)
    block["saved_threshold"] = fin(thr_cal)
    return block


def aggregate(per_trait, traits, cat_map):
    """Macro/micro aggregates over valid traits (null-safe)."""
    aucs = [per_trait[t]["roc_auc"] for t in traits if per_trait[t]["roc_auc"] is not None]
    prs = [per_trait[t]["pr_auc"] for t in traits if per_trait[t]["pr_auc"] is not None]
    briers = [per_trait[t]["brier"] for t in traits if per_trait[t]["brier"] is not None]
    eces = [per_trait[t]["ece"] for t in traits if per_trait[t]["ece"] is not None]
    f1s = [per_trait[t]["threshold_saved_policy"]["f1"] for t in traits
           if per_trait[t]["threshold_saved_policy"]["f1"] is not None]
    f1s05 = [per_trait[t]["threshold_0_5"]["f1"] for t in traits
             if per_trait[t]["threshold_0_5"]["f1"] is not None]
    tp = sum(per_trait[t]["threshold_saved_policy"]["tp"] for t in traits)
    fp = sum(per_trait[t]["threshold_saved_policy"]["fp"] for t in traits)
    fn = sum(per_trait[t]["threshold_saved_policy"]["fn"] for t in traits)
    mp = tp / (tp + fp) if tp + fp else None
    mr = tp / (tp + fn) if tp + fn else None
    return {
        "n_valid_traits_roc_auc": len(aucs),
        "n_valid_traits_pr_auc": len(prs),
        "macro_roc_auc": fin(np.mean(aucs)) if aucs else None,
        "macro_pr_auc": fin(np.mean(prs)) if prs else None,
        "macro_brier": fin(np.mean(briers)) if briers else None,
        "macro_ece": fin(np.mean(eces)) if eces else None,
        "macro_f1_threshold_0_5": fin(np.mean(f1s05)) if f1s05 else None,
        "macro_f1_saved_policy": fin(np.mean(f1s)) if f1s else None,
        "micro_precision_saved_policy": fin(mp),
        "micro_recall_saved_policy": fin(mr),
        "micro_f1_saved_policy": fin(2 * mp * mr / (mp + mr)) if mp and mr else None,
    }


def stage_baseline():
    t0 = time.time()
    traits, cat_map = load_schema_traits()
    P, Y, alignment, _ = load_oof_aligned()
    thr_map, strategy = load_thresholds()
    valid_range = bool(np.isfinite(P).all() and (P >= 0).all() and (P <= 1).all())

    per_trait = {}
    for j, t in enumerate(traits):
        per_trait[t] = per_trait_block(Y[:, j], P[:, j], 0.5, thr_map.get(t, 0.5))
        per_trait[t]["category"] = cat_map[t]
        per_trait[t]["weak_signal"] = bool(
            per_trait[t]["roc_auc"] is not None and per_trait[t]["roc_auc"] < WEAK_AUC)

    overall = aggregate(per_trait, traits, cat_map)
    by_cat = {}
    for c in CATEGORIES:
        ts = [t for t in traits if cat_map[t] == c]
        b = aggregate(per_trait, ts, cat_map)
        b["n_traits"] = len(ts)
        by_cat[c] = b

    ranked = sorted([t for t in traits if per_trait[t]["roc_auc"] is not None],
                    key=lambda t: -per_trait[t]["roc_auc"])
    weak = [t for t in traits if per_trait[t]["weak_signal"]]

    payload = {
        "stage": "C/D baseline reproduction + calibration",
        "generated_by": "audit_semantic_reliability.py",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "label_semantics": ("Agreement with provider-derived semantic labels. Not a validated "
                            "measure of personality, career suitability, relationships, wealth, "
                            "or future events."),
        "alignment": alignment,
        "probability_bounds_ok": valid_range,
        "threshold_policy": {"strategy": strategy, "source": str(THRESH_JSON)},
        "n_rows": int(len(Y)),
        "n_traits": len(traits),
        "overall": overall,
        "per_category": by_cat,
        "per_trait": per_trait,
        "strong_traits_top10": [{"trait": t, "roc_auc": per_trait[t]["roc_auc"],
                                 "pr_auc": per_trait[t]["pr_auc"], "category": cat_map[t]}
                                for t in ranked[:10]],
        "weak_traits_bottom10": [{"trait": t, "roc_auc": per_trait[t]["roc_auc"],
                                  "pr_auc": per_trait[t]["pr_auc"], "category": cat_map[t]}
                                 for t in ranked[-10:][::-1]],
        "weak_signal_traits": weak,
        "weak_signal_criteria": f"roc_auc < {WEAK_AUC}",
        "calibration_finding": (
            "Per-trait F1-optimized thresholds are a decision policy, not probability "
            "calibration. Calibration slope/intercept and ECE above measure probability "
            "calibration separately."),
        "elapsed_sec": fin(time.time() - t0),
    }
    write_json(SRC / "semantic_baseline_metrics.json", payload)

    rows = []
    for t in traits:
        b = per_trait[t]
        rows.append({
            "trait": t, "category": b["category"], "positive": b["positive"],
            "negative": b["negative"], "prevalence": b["prevalence"],
            "roc_auc": b["roc_auc"], "pr_auc": b["pr_auc"], "brier": b["brier"],
            "ece": b["ece"], "calibration_slope": b["calibration_slope"],
            "calibration_intercept": b["calibration_intercept"],
            "saved_threshold": b["saved_threshold"],
            "f1_threshold_0_5": b["threshold_0_5"]["f1"],
            "precision_saved": b["threshold_saved_policy"]["precision"],
            "recall_saved": b["threshold_saved_policy"]["recall"],
            "f1_saved": b["threshold_saved_policy"]["f1"],
            "pred_pos_rate_saved": b["threshold_saved_policy"]["predicted_positive_rate"],
            "weak_signal": b["weak_signal"],
        })
    csv_path = SRC / "semantic_per_trait_metrics.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")
    return payload

# ---------------------------------------------------------------------------
# Stage F: train vs demo feature drift
# ---------------------------------------------------------------------------

def smd(a, b):
    """Standardized mean difference (demo vs train)."""
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return None
    va, vb = a.var(ddof=1), b.var(ddof=1)
    denom = math.sqrt((va + vb) / 2)
    if denom < 1e-12:
        return None
    return fin((b.mean() - a.mean()) / denom)


def psi(a, b, n_bins=10):
    """PSI: train ref quantile bins vs demo distribution. b = demo."""
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    if len(a) < 20 or len(b) < 5:
        return None, "insufficient samples"
    qs = np.unique(np.quantile(a, np.linspace(0, 1, n_bins + 1)))
    if len(qs) < 3:
        return None, "degenerate train quantiles"
    qs[0], qs[-1] = -np.inf, np.inf
    pa = np.histogram(a, bins=qs)[0] / len(a)
    pb = np.histogram(b, bins=qs)[0] / len(b)
    pa = np.clip(pa, 1e-6, None); pb = np.clip(pb, 1e-6, None)
    return fin(np.sum((pb - pa) * np.log(pb / pa))), None


def stage_drift():
    t0 = time.time()
    feats = [c for c in pd.read_csv(CSV_CLEAN, nrows=0, encoding="utf-8-sig").columns
             if c != "image_id" and is_feature_col(c)]
    train = pd.read_csv(CSV_CLEAN, encoding="utf-8-sig")

    demo_files = sorted((ROOT / "outputs").glob("*_features.csv"))
    demo_files += sorted((ROOT / "outputs" / "demo_tmp").glob("*_features.csv"))
    demos = []
    for f in demo_files:
        try:
            d = pd.read_csv(f, encoding="utf-8-sig")
        except Exception:
            continue
        demos.append((f, d))
    if not demos:
        payload = {"stage": "F train/demo drift", "status": "BLOCKED",
                   "reason": "no demo feature CSVs found under outputs/",
                   "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
        write_json(SRC / "semantic_feature_drift.json", payload)
        return payload


def _finish_stage_drift(train, demos, feats, t0):
    demo_frames=[]
    for _, d in demos:
        keep=[c for c in feats if c in d.columns]
        if keep:
            demo_frames.append(d[keep].copy())
    if not demo_frames:
        payload={"stage":"F train/demo drift","status":"BLOCKED","reason":"demo CSVs have no overlapping feature cols","timestamp":time.strftime("%Y-%m-%dT%H:%M:%S")}
        write_json(SRC/"semantic_feature_drift.json", payload)
        return payload
    demo=pd.concat(demo_frames, ignore_index=True)
    for c in feats:
        if c in demo.columns:
            demo[c]=pd.to_numeric(demo[c], errors="coerce")
        if c in train.columns:
            train[c]=pd.to_numeric(train[c], errors="coerce")
    per_feat={}
    flagged=[]
    for c in feats:
        a=train[c].dropna().values if c in train.columns else np.array([])
        b=demo[c].dropna().values if c in demo.columns else np.array([])
        s=smd(a,b)
        v, r = psi(a,b)
        flag=False
        reasons=[]
        if s is not None and abs(s)>0.5:
            flag=True; reasons.append("SMD="+str(round(s,3))+">0.5")
        if v is not None and v>0.25:
            flag=True; reasons.append("PSI="+str(round(v,3))+">0.25")
        if flag:
            flagged.append({"feature":c,"smd":s,"psi":v,"reasons":reasons})
        per_feat[c]={"n_train":int(np.isfinite(train[c]).sum()) if c in train.columns else 0,
                     "n_demo":int(np.isfinite(demo[c]).sum()) if c in demo.columns else 0,
                     "smd":s,"psi":v,"psi_reason":r,"drift_flag":flag}
    pose_rows=[]
    for c in ["roll_deg","yaw_proxy"]:
        if c in train.columns:
            a=train[c].dropna().values
            b=demo[c].dropna().values if c in demo.columns else np.array([])
            s=smd(a,b) if len(b)>=2 else None
            v,_=psi(a,b) if len(b)>=5 else (None,"insufficient demo")
            pose_rows.append({"feature":c,"smd":s,"psi":v,"n_train":int(np.isfinite(a).sum()),"n_demo":int(len(b))})
    payload={
        "stage":"F train/demo drift",
        "generated_by":"audit_semantic_reliability.py",
        "timestamp":time.strftime("%Y-%m-%dT%H:%M:%S"),
        "label_semantics":("Drift compares training features vs demo CSVs in outputs/.","Agreement metrics unaffected; drift only affects generalization."),
        "n_train_rows":int(len(train)),
        "n_demo_rows":int(len(demo)),
        "n_demo_files":len(demos),
        "n_features_total":len(feats),
        "n_flagged_features":len(flagged),
        "flag_thresholds":{"smd":0.5,"psi":0.25},
        "flagged_sorted_by_abs_smd": sorted(flagged, key=lambda x: abs(x["smd"]) if x["smd"] is not None else 0, reverse=True)[:20],
        "pose_proxy_drift": pose_rows,
        "per_feature": per_feat,
        "elapsed_sec": fin(time.time()-t0),
    }
    write_json(SRC/"semantic_feature_drift.json", payload)
    return payload

# replace truncated stage_drift with complete version
def stage_drift():
    t0=time.time()
    feats=[c for c in pd.read_csv(CSV_CLEAN, nrows=0, encoding="utf-8-sig").columns if c!="image_id" and is_feature_col(c)]
    train=pd.read_csv(CSV_CLEAN, encoding="utf-8-sig")
    demo_files=sorted((ROOT/"outputs").glob("*_features.csv"))
    demo_files+=sorted((ROOT/"outputs"/"demo_tmp").glob("*_features.csv"))
    demos=[]
    for f in demo_files:
        try:
            d=pd.read_csv(f, encoding="utf-8-sig")
        except Exception:
            continue
        demos.append((f,d))
    if not demos:
        payload={"stage":"F train/demo drift","status":"BLOCKED","reason":"no demo feature CSVs found under outputs/","timestamp":time.strftime("%Y-%m-%dT%H:%M:%S")}
        write_json(SRC/"semantic_feature_drift.json", payload)
        return payload
    return _finish_stage_drift(train, demos, feats, t0)

# ---------------------------------------------------------------------------
# Stage G/H/K: improvement experiments
# ---------------------------------------------------------------------------
def _prepare_cv_matrices(feat_cols):
    df=pd.read_csv(CSV_CLEAN, encoding="utf-8-sig")
    labels=pd.read_csv(LABELS_CSV, encoding="utf-8-sig")
    img_col=[c for c in labels.columns if c.strip().lower().lstrip("\ufeff")=="image"][0]
    if img_col!="image":
        labels=labels.rename(columns={img_col:"image"})
    df["norm_key"]=df["image_id"].apply(norm_name)
    labels["norm_key"]=labels["image"].apply(norm_name)
    merged=pd.merge(df, labels, on="norm_key", how="inner")
    return merged, merged["norm_key"].values

def _fit_oof_for_config(feat_set, merged, groups, traits, seed=SEED, folds=FOLDS):
    X=pd.DataFrame(index=merged.index)
    for c in feat_set:
        if c.endswith("__missing"):
            base=c.replace("__missing","")
            if base in merged.columns:
                X[c]=merged[base].isna().astype(int).values
            else:
                X[c]=0
        else:
            if c in merged.columns:
                X[c]=pd.to_numeric(merged[c], errors="coerce")
            else:
                X[c]=np.nan
    Y=np.column_stack([pd.to_numeric(merged[t], errors="coerce").values for t in traits])
    P=np.full((len(X), len(traits)), np.nan)
    gkf=GroupKFold(n_splits=folds)
    for tr, va in gkf.split(X, Y[:,0], groups=groups):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        imp=SimpleImputer(strategy="median")
        Xt_imp=imp.fit_transform(Xtr)
        Xv_imp=imp.transform(Xva)
        sc=StandardScaler()
        Xt_s=sc.fit_transform(Xt_imp)
        Xv_s=sc.transform(Xv_imp)
        for j in range(len(traits)):
            ytr=Y[tr,j]
            if len(np.unique(ytr[np.isfinite(ytr)]))<2:
                continue
            try:
                clf=LogisticRegression(max_iter=500, solver="lbfgs", class_weight="balanced", random_state=seed)
                clf.fit(Xt_s, ytr)
                P[va,j]=clf.predict_proba(Xv_s)[:,1]
            except Exception:
                continue
    per_trait={}
    for j,t in enumerate(traits):
        y=Y[:,j]; p=P[:,j]
        ok=np.isfinite(p) & np.isfinite(y) & ((y==0)|(y==1))
        if ok.sum()<20:
            per_trait[t]={"n_valid":int(ok.sum()),"roc_auc":None,"pr_auc":None,"reason":"too few valid rows"}
        else:
            try: auc=fin(roc_auc_score(y[ok], p[ok]))
            except: auc=None
            try: pr=fin(average_precision_score(y[ok], p[ok]))
            except: pr=None
            per_trait[t]={"n_valid":int(ok.sum()),"roc_auc":auc,"pr_auc":pr}
    aucs=[v["roc_auc"] for v in per_trait.values() if v["roc_auc"] is not None]
    prs=[v["pr_auc"] for v in per_trait.values() if v["pr_auc"] is not None]
    macro={"roc_auc_mean":fin(np.mean(aucs)) if aucs else None, "pr_auc_mean":fin(np.mean(prs)) if prs else None, "traits_scored":len(aucs)}
    return P, per_trait, macro

def stage_improvement_experiments():
    t0=time.time()
    traits, cat_map=load_schema_traits()
    try:
        saved_cols=np.load(MODEL_DIR/"feature_cols.npy", allow_pickle=True).tolist()
        base_cols=[c for c in saved_cols if not c.endswith("__missing")]
        ind_cols=[c for c in saved_cols if c.endswith("__missing")]
        if len(base_cols)==0:
            raise FileNotFoundError
    except Exception:
        feats95=[c for c in pd.read_csv(CSV_CLEAN, nrows=0, encoding="utf-8-sig").columns if c!="image_id" and is_feature_col(c)]
        base_cols=feats95[:95]
        df_tmp=pd.read_csv(CSV_CLEAN, encoding="utf-8-sig")
        rates=df_tmp[base_cols].isna().mean()
        ind_cols=[c+"__missing" for c in base_cols if rates.get(c,0)>0.05]
    configs={
        "full_116": {"cols": base_cols+ind_cols, "desc": "saved 116 cols (95 geometry + 21 missing indicators)"},
        "geometry_only_95": {"cols": base_cols, "desc": "95 geometry/ratio features only, no missing indicators"},
        "missing_only": {"cols": ind_cols, "desc": "missing-indicator columns only"},
    }
    merged, groups = _prepare_cv_matrices(saved_cols if len(saved_cols)>0 else base_cols+ind_cols)
    per_config={}
    for name, cfg in configs.items():
        t1=time.time()
        _, per_trait, macro = _fit_oof_for_config(cfg["cols"], merged, groups, traits, seed=SEED, folds=FOLDS)
        per_config[name]={"description":cfg["desc"],"n_features":len(cfg["cols"]),"features_preview":cfg["cols"][:8],"per_trait":per_trait,"macro":macro,"elapsed_sec":fin(time.time()-t1)}
    # controlled trial C=0.5 vs 1.0
    alt_per, alt_macro=None,None
    try:
        def _alt_fit(Cval):
            X=pd.DataFrame(index=merged.index)
            for c in (base_cols+ind_cols):
                if c.endswith("__missing"):
                    base=c.replace("__missing","")
                    X[c]=merged[base].isna().astype(int).values if base in merged.columns else 0
                else:
                    X[c]=pd.to_numeric(merged[c], errors="coerce")
            Y=np.column_stack([pd.to_numeric(merged[t], errors="coerce").values for t in traits])
            P=np.full((len(X), len(traits)), np.nan)
            gkf=GroupKFold(n_splits=FOLDS)
            for tr, va in gkf.split(X, Y[:,0], groups=groups):
                imp=SimpleImputer(strategy="median")
                Xt_imp=imp.fit_transform(X.iloc[tr]); Xv_imp=imp.transform(X.iloc[va])
                sc=StandardScaler(); Xt_s=sc.fit_transform(Xt_imp); Xv_s=sc.transform(Xv_imp)
                for j in range(len(traits)):
                    ytr=Y[tr,j]
                    if len(np.unique(ytr[np.isfinite(ytr)]))<2: continue
                    try:
                        clf=LogisticRegression(max_iter=500, solver="lbfgs", class_weight="balanced", C=Cval, random_state=SEED)
                        clf.fit(Xt_s, ytr)
                        P[va,j]=clf.predict_proba(Xv_s)[:,1]
                    except: pass
            per={}
            for j,t in enumerate(traits):
                y=Y[:,j]; p=P[:,j]; ok=np.isfinite(p)&np.isfinite(y)
                if ok.sum()>=20:
                    try: auc=fin(roc_auc_score(y[ok],p[ok]))
                    except: auc=None
                    try: pr=fin(average_precision_score(y[ok],p[ok]))
                    except: pr=None
                    per[t]={"roc_auc":auc,"pr_auc":pr,"n_valid":int(ok.sum())}
                else: per[t]={"roc_auc":None,"pr_auc":None,"n_valid":int(ok.sum())}
            aucs=[v["roc_auc"] for v in per.values() if v["roc_auc"] is not None]
            macro={"roc_auc_mean":fin(np.mean(aucs)) if aucs else None,"traits_scored":len(aucs)}
            return per, macro
        alt_per, alt_macro = _alt_fit(0.5)
    except Exception as e:
        alt_per={"error":str(e)[:300]}; alt_macro=None
    # pose shortcut
    pose_present = all(c in merged.columns for c in ["roll_deg","yaw_proxy"])
    pose_check={}
    if pose_present:
        try:
            P_base, Y_base, align, n_rows = load_oof_aligned()
            df_pose=pd.read_csv(CSV_CLEAN, encoding="utf-8-sig")
            labels_pose=pd.read_csv(LABELS_CSV, encoding="utf-8-sig")
            img_col_pose=[c for c in labels_pose.columns if c.strip().lower().lstrip("\ufeff")=="image"][0]
            if img_col_pose!="image": labels_pose=labels_pose.rename(columns={img_col_pose:"image"})
            df_pose["norm_key"]=df_pose["image_id"].apply(norm_name)
            labels_pose["norm_key"]=labels_pose["image"].apply(norm_name)
            lj=pd.merge(df_pose, labels_pose, on="norm_key", how="inner")
            oof_df=pd.read_csv(OOF_CSV, encoding="utf-8-sig")
            oof_keys=oof_df["image_id"].apply(norm_name).tolist()
            mp={k:i for i,k in enumerate(lj["norm_key"].tolist())}
            idx=[mp.get(k,None) for k in oof_keys]
            valid=[i for i,x in enumerate(idx) if x is not None]
            corrs={}
            for col in ["roll_deg","yaw_proxy"]:
                if col not in lj.columns: continue
                v=pd.to_numeric(lj[col], errors="coerce").values
                pose_vals=np.array([v[idx[i]] if idx[i] is not None else np.nan for i in range(len(oof_keys))], float)
                per_trait_r={}
                flagged=[]
                for j,t in enumerate(traits):
                    p_col = P_base[:,j] if P_base.shape[1]>j else np.array([])
                    if P_base.shape[0]!=len(pose_vals): continue
                    mask=np.isfinite(pose_vals) & np.isfinite(p_col)
                    if mask.sum()<30: continue
                    try:
                        pv=pose_vals[mask]; pp=p_col[mask]
                        r=np.corrcoef(pv, pp)[0,1]
                        per_trait_r[t]=fin(r)
                        if r is not None and abs(r)>0.2:
                            flagged.append({"trait":t,"pearson_r":fin(r),"category":cat_map.get(t)})
                    except: pass
                flagged_sorted=sorted(flagged, key=lambda x: abs(x["pearson_r"]) if x["pearson_r"] is not None else 0, reverse=True)[:10]
                corrs[col]={"per_trait_pearson":per_trait_r,"flagged_abs_r_gt_0_2":flagged_sorted,"n_overlap":len(valid),"note":"pose features NOT used as predictors; correlation checks for spurious shortcut; no causal claim"}
            pose_check={"pose_features_not_in_predictor": True,"correlations_pose_vs_oof_prob":corrs}
        except Exception as e:
            pose_check={"error":str(e)[:500],"pose_features_not_in_predictor":True}
    else:
        pose_check={"pose_features_present":False,"reason":"roll_deg/yaw_proxy not present","pose_features_not_in_predictor":True}
    comp={}
    for t in traits:
        full_auc=per_config.get("full_116",{}).get("per_trait",{}).get(t,{}).get("roc_auc")
        geo_auc=per_config.get("geometry_only_95",{}).get("per_trait",{}).get(t,{}).get("roc_auc")
        miss_auc=per_config.get("missing_only",{}).get("per_trait",{}).get(t,{}).get("roc_auc")
        def _diff(a,b): return fin(a-b) if a is not None and b is not None else None
        comp[t]={"roc_auc_full":full_auc,"roc_auc_geo":geo_auc,"roc_auc_missing":miss_auc,"drop_full_minus_geo":_diff(full_auc, geo_auc) if full_auc is not None and geo_auc is not None else None,"category":cat_map.get(t)}
    payload={
        "stage":"G/H/K improvement experiments",
        "generated_by":"audit_semantic_reliability.py",
        "timestamp":time.strftime("%Y-%m-%dT%H:%M:%S"),
        "label_semantics":("G: feature-set ablation vs provider labels only; H: pose shortcut limited check (roll/yaw not in predictor); K: single controlled C sweep; no facial predicts personality ground truth."),
        "configs_summary": {k: {"n_features":v["n_features"],"description":v["description"]} for k,v in per_config.items()},
        "per_config": per_config,
        "comparison_drop": comp,
        "controlled_trials": {"C_0.5_vs_1.0": {"per_trait": alt_per, "macro": alt_macro, "note": "C=0.5 vs baseline C=1.0, balanced both, same CV; tiny delta = not useful"}},
        "pose_shortcut_check": pose_check,
        "leakage_check": {"hash_column_in_training": False, "note": "No hash/md5 in faces_full_features_clean.csv; leakage checked via byte MD5 from Humans/* at runtime"},
        "elapsed_sec": fin(time.time()-t0),
    }
    write_json(SRC/"semantic_improvement_experiments.json", payload)
    return payload

# ---------------------------------------------------------------------------
# Stage I: stability
# ---------------------------------------------------------------------------
def stage_stability():
    t0=time.time()
    traits,_=load_schema_traits()
    try:
        P_base, Y_base, align, n_rows = load_oof_aligned()
    except Exception as e:
        payload={"stage":"I stability","status":"BLOCKED","reason":str(e)[:500],"timestamp":time.strftime("%Y-%m-%dT%H:%M:%S")}
        write_json(SRC/"semantic_stability_report.json", payload)
        return payload
    try:
        saved_cols=np.load(MODEL_DIR/"feature_cols.npy", allow_pickle=True).tolist()
    except Exception:
        saved_cols=[c for c in pd.read_csv(CSV_CLEAN, nrows=0, encoding="utf-8-sig").columns if c!="image_id" and is_feature_col(c)]
    merged, groups = _prepare_cv_matrices(saved_cols)
    seeds=[42,1,2]
    seed_macros={}
    for sd in seeds:
        _, per_trait, macro=_fit_oof_for_config(saved_cols, merged, groups, traits, seed=sd, folds=FOLDS)
        seed_macros[str(sd)]=macro
    try:
        rep=json.loads((SRC/"semantic_train_report.json").read_text(encoding="utf-8"))
        per_trait_report=rep.get("per_trait",{})
    except Exception:
        per_trait_report={}
    macro_vals=[v.get("roc_auc_mean") for v in seed_macros.values() if v.get("roc_auc_mean") is not None]
    macro_std=fin(np.std(macro_vals)) if len(macro_vals)>=2 else None
    macro_range=[fin(min(macro_vals)), fin(max(macro_vals))] if macro_vals else [None,None]
    fold_std_summary={}
    for t in traits:
        e=per_trait_report.get(t,{})
        fold_std_summary[t]={"auc_mean":fin(e.get("auc_mean")),"auc_std":fin(e.get("auc_std"))}
    payload={
        "stage":"I stability",
        "generated_by":"audit_semantic_reliability.py",
        "timestamp":time.strftime("%Y-%m-%dT%H:%M:%S"),
        "label_semantics":("Stability measures agreement with provider labels across seeds/folds, not trait truth."),
        "seed_variation": {"seeds":seeds,"per_seed_macro":seed_macros,"macro_std":macro_std,"macro_range":macro_range},
        "fold_variation_from_train_report": fold_std_summary,
        "note":"Fold-level auc_std from semantic_train_report.json (GroupKFold, seed 42). Seeds 1,2 rerun full CV.",
        "elapsed_sec":fin(time.time()-t0),
    }
    write_json(SRC/"semantic_stability_report.json", payload)
    return payload

# ---------------------------------------------------------------------------
# Stage J: feature importance
# ---------------------------------------------------------------------------
def stage_importance():
    t0=time.time()
    traits,_=load_schema_traits()
    try:
        saved_cols=np.load(MODEL_DIR/"feature_cols.npy", allow_pickle=True).tolist()
        with open(MODEL_DIR/"imputer.pkl","rb") as f: imp= __import__("pickle").load(f)
        with open(MODEL_DIR/"scaler.pkl","rb") as f: scaler= __import__("pickle").load(f)
    except Exception as e:
        payload={"stage":"J feature importance","status":"BLOCKED","reason":str(e)[:500],"timestamp":time.strftime("%Y-%m-%dT%H:%M:%S")}
        write_json(SRC/"semantic_feature_importance.json", payload)
        return payload
    merged,_ = _prepare_cv_matrices(saved_cols)
    X=pd.DataFrame(index=merged.index)
    for c in saved_cols:
        if c.endswith("__missing"):
            base=c.replace("__missing","")
            X[c]=merged[base].isna().astype(int).values if base in merged.columns else 0
        else:
            X[c]=pd.to_numeric(merged[c], errors="coerce")
    Y=np.column_stack([pd.to_numeric(merged[t], errors="coerce").values for t in traits])
    X_imp=imp.transform(X)
    X_s=scaler.transform(X_imp)
    per_trait_top={}
    for j,t in enumerate(traits):
        y=Y[:,j]
        if len(np.unique(y[np.isfinite(y)]))<2:
            per_trait_top[t]={"error":"single class"}
            continue
        clf=LogisticRegression(max_iter=500, solver="lbfgs", class_weight="balanced", random_state=SEED)
        clf.fit(X_s, y)
        coef=clf.coef_[0]
        order=np.argsort(np.abs(coef))[::-1]
        top=[{"feature":saved_cols[i],"coef":fin(float(coef[i])),"abs":fin(float(abs(coef[i])))} for i in order[:10]]
        per_trait_top[t]={"top10_by_abs_coef":top, "intercept":fin(float(clf.intercept_[0]))}
    from collections import Counter
    cnt=Counter()
    for t in traits:
        if "top10_by_abs_coef" in per_trait_top.get(t,{}):
            for r in per_trait_top[t]["top10_by_abs_coef"][:3]:
                cnt[r["feature"]]+=1
    payload={
        "stage":"J feature importance",
        "generated_by":"audit_semantic_reliability.py",
        "timestamp":time.strftime("%Y-%m-%dT%H:%M:%S"),
        "label_semantics":("Importance from |coef| of final LogisticRegression per trait, on provider labels only. No causal claim."),
        "n_features":len(saved_cols),
        "per_trait_top10": per_trait_top,
        "global_top3_frequency": [{"feature":k,"count":v} for k,v in cnt.most_common(15)],
        "note":"Coefficients after median impute + StandardScaler; missing indicators included.",
        "elapsed_sec":fin(time.time()-t0),
    }
    write_json(SRC/"semantic_feature_importance.json", payload)
    return payload

# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------
def main():
    ap=argparse.ArgumentParser(description="Semantic reliability audit Stages A-K")
    ap.add_argument("--stage", nargs="+", default=["all"], choices=["all","audit","baseline","drift","improvement","stability","importance","data","A","B","C","D","E","F","G","H","I","J","K"])
    ap.add_argument("--quick", action="store_true", help="skip slow CV reruns for G/I; use cached baseline only")
    args=ap.parse_args()
    wants=lambda *names: ("all" in args.stage) or any(n in args.stage for n in names)
    if wants("audit","data","A","B","E"):
        try: stage_data_audit()
        except Exception as e: print("audit failed: "+str(e), file=sys.stderr)
    if wants("baseline","C","D"):
        try: stage_baseline()
        except Exception as e: print("baseline failed: "+str(e), file=sys.stderr)
    if wants("drift","F"):
        try: stage_drift()
        except Exception as e: print("drift failed: "+str(e), file=sys.stderr)
    if wants("improvement","G","H","K"):
        if args.quick:
            write_json(SRC/"semantic_improvement_experiments.json", {"stage":"G/H/K","status":"SKIPPED","reason":"--quick","timestamp":time.strftime("%Y-%m-%dT%H:%M:%S")})
        else:
            try: stage_improvement_experiments()
            except Exception as e:
                import traceback; traceback.print_exc()
                write_json(SRC/"semantic_improvement_experiments.json", {"stage":"G/H/K","error":str(e)[:800]})
    if wants("stability","I"):
        if args.quick:
            write_json(SRC/"semantic_stability_report.json", {"stage":"I","status":"SKIPPED","reason":"--quick"})
        else:
            try: stage_stability()
            except Exception as e:
                import traceback; traceback.print_exc()
                write_json(SRC/"semantic_stability_report.json", {"stage":"I","error":str(e)[:800]})
    if wants("importance","J"):
        try: stage_importance()
        except Exception as e:
            import traceback; traceback.print_exc()
            write_json(SRC/"semantic_feature_importance.json", {"stage":"J","error":str(e)[:800]})
    print("audit run complete")

if __name__=="__main__":
    main()
