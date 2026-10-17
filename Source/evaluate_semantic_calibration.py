# -*- coding: utf-8 -*-
"""Phase 3.13: semantic probability calibration (exploratory, no promotion).

Loads Strategy A hash-safe OOF predictions (phase 3.12). Does not retrain
classifiers, does not touch production models, labels, historical OOF, or
threshold files.

Protocol (EXPLORATORY, not nested end-to-end):
  Base OOF probs come from models trained on overlapping folds. Calibrators
  are cross-fit over those saved OOF rows, splitting by connected component
  so a component never sits in both calibration-train and calibration-eval.
  This is NOT a fully nested leakage-free evaluation. Results must not be
  used alone to approve production promotion.

Candidates (existing sklearn only):
  A uncalibrated
  B sigmoid (Platt) logistic on logit(p), class_weight=None
  C isotonic, only when independent pos/neg component support is enough
  D analytic prior correction for class_weight="balanced":
      logit(p') = logit(p) - log((1-pi)/pi)
    pi = prevalence estimated on calibration-train rows only.

Usage:
  python Source/evaluate_semantic_calibration.py
  python Source/evaluate_semantic_calibration.py --bootstrap 200
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
SCHEMA_JSON = ROOT / "semantic_schema.json"
LABELS_CSV = ROOT / "semantic_labels.csv"
OOF_A = SRC / "experiments" / "semantic_noise_phase312" / "oof_strategy_A.csv"
FOLD_CSV = SRC / "experiments" / "hash_grouped_semantic" / "cv_fold_assignments.csv"
PHASE312_METRICS = (
    SRC / "experiments" / "semantic_noise_phase312" / "semantic_noise_strategy_metrics.json"
)
THRESH_JSON = SRC / "models_semantic" / "trait_thresholds.json"
OUT_DIR = SRC / "experiments" / "semantic_calibration_phase313"

SEED = 42
N_BINS = 10
CLIP = 1e-6
# Isotonic needs both classes and enough independent components.
# Below this, isotonic is refused and sigmoid is used instead.
MIN_ISOTONIC_POS_GROUPS = 30
MIN_ISOTONIC_NEG_GROUPS = 30
META_COLS = ("image_id", "fold", "component_group_id", "sha256")
# Production artifacts that must not change. Sizes recorded before/after.
PROTECTED = [
    ROOT / "faces_full_features_clean.csv",
    ROOT / "labels_enriched.jsonl",
    ROOT / "oof_semantic_probs.csv",
    SRC / "models_semantic" / "trait_thresholds.json",
]


def fin(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def sanitize(o):
    if isinstance(o, dict):
        return {k: sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitize(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return fin(float(o))
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float):
        return fin(o)
    return o


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sanitize(obj), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def clip_prob(p):
    return np.clip(np.asarray(p, float), CLIP, 1.0 - CLIP)


def logit(p):
    p = clip_prob(p)
    return np.log(p / (1.0 - p))


def ece_score(y, p, n_bins=N_BINS):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    if len(y) == 0:
        return None
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    e = 0.0
    n = len(y)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p <= hi) if i == n_bins - 1 else (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        e += abs(y[m].mean() - p[m].mean()) * (m.sum() / n)
    return float(e)


def safe_auc(y, p):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return None
    return fin(roc_auc_score(y, p))


def safe_ap(y, p):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return None
    return fin(average_precision_score(y, p))


def safe_brier(y, p):
    if len(y) == 0:
        return None
    return fin(brier_score_loss(y, np.clip(p, 0.0, 1.0)))


def safe_logloss(y, p):
    if len(y) == 0 or len(np.unique(y)) < 1:
        return None
    return fin(log_loss(y, clip_prob(p), labels=[0, 1]))


def calibration_slope_intercept(y, p):
    """Logistic of y on logit(p). Slope~1 and intercept~0 means calibrated.

    Returns (slope, intercept) or (None, None) when a class is missing or
    the fit fails to converge to finite coefficients.
    """
    y = np.asarray(y, float)
    if len(y) < 10 or len(np.unique(y)) < 2:
        return None, None
    x = logit(p).reshape(-1, 1)
    try:
        clf = LogisticRegression(solver="lbfgs", max_iter=200, class_weight=None)
        clf.fit(x, y)
    except Exception:
        return None, None
    return fin(float(clf.coef_[0, 0])), fin(float(clf.intercept_[0]))


def metrics_block(y, p):
    return {
        "roc_auc": safe_auc(y, p),
        "pr_auc": safe_ap(y, p),
        "brier": safe_brier(y, p),
        "ece": ece_score(y, p),
        "log_loss": safe_logloss(y, p),
        "n": int(len(y)),
    }


def load_schema():
    sch = json.loads(SCHEMA_JSON.read_text(encoding="utf-8"))
    traits = [t["name"] for t in sch["traits"]]
    cat = {t["name"]: t["category"] for t in sch["traits"]}
    return traits, cat


def load_labels(traits):
    labels = pd.read_csv(LABELS_CSV, encoding="utf-8-sig")
    img_col = [c for c in labels.columns if c.strip().lower().lstrip("\ufeff") == "image"][0]
    if img_col != "image":
        labels = labels.rename(columns={img_col: "image"})
    return labels


def norm_name(value):
    import os
    import re
    s = os.path.basename(str(value).strip())
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\)\s+\(", ")(", s)
    return s.lower()


def load_aligned(traits):
    """Join Strategy A OOF to labels and fold assignments on image name."""
    oof = pd.read_csv(OOF_A, encoding="utf-8")
    labels = load_labels(traits)
    folds = pd.read_csv(FOLD_CSV, encoding="utf-8")

    oof["norm_key"] = oof["image_id"].apply(norm_name)
    labels["norm_key"] = labels["image"].apply(norm_name)
    folds["norm_key"] = folds["norm_key"].astype(str)

    merged = oof.merge(
        labels[["norm_key"] + traits], on="norm_key", how="inner", suffixes=("", "_y")
    )
    fmap = dict(zip(folds["norm_key"], folds["fold"].astype(int)))
    gmap = dict(zip(folds["norm_key"], folds["safe_group"].astype(int)))
    smap = dict(zip(folds["norm_key"], folds["sha256"].astype(str)))
    merged["fold_check"] = merged["norm_key"].map(fmap)
    merged["group_check"] = merged["norm_key"].map(gmap)
    merged["sha_check"] = merged["norm_key"].map(smap)
    return merged


def verify_alignment(merged, traits):
    problems = []
    if len(merged) != 6983:
        problems.append(f"row count {len(merged)} != 6983")
    missing = [t for t in traits if t not in merged.columns]
    if missing:
        problems.append(f"missing trait cols {missing[:5]}")
    P = merged[traits].to_numpy(float)
    if np.isnan(P).any():
        problems.append("NaN probabilities")
    if (P < 0).any() or (P > 1).any():
        problems.append("probabilities outside [0,1]")
    if merged["fold"].isna().any() or not set(merged["fold"].unique()) <= set(range(5)):
        problems.append("bad fold ids")
    if merged["fold_check"].isna().any():
        problems.append("rows missing fold assignment")
    if not np.array_equal(merged["fold"].to_numpy(), merged["fold_check"].to_numpy()):
        problems.append("OOF fold disagrees with cv_fold_assignments")
    if not np.array_equal(
        merged["component_group_id"].to_numpy(), merged["group_check"].to_numpy()
    ):
        problems.append("OOF component disagrees with cv_fold_assignments")
    if not (merged["sha256"].astype(str) == merged["sha_check"].astype(str)).all():
        problems.append("OOF sha256 disagrees with cv_fold_assignments")
    # trait column order must match schema order
    oof_cols = [c for c in merged.columns if c in traits]
    if oof_cols != traits:
        problems.append("trait column order differs from schema")
    return problems


def group_support(y, groups):
    """Independent positive/negative component counts for one trait."""
    g = np.asarray(groups)
    y = np.asarray(y)
    pos = set(g[y == 1])
    neg = set(g[y == 0])
    return len(pos), len(neg)


def fit_sigmoid(p_train, y_train):
    """Platt: logistic regression of y on logit(p). Returns None if degenerate."""
    y = np.asarray(y_train, float)
    if len(np.unique(y)) < 2 or len(y) < 20:
        return None
    x = logit(p_train).reshape(-1, 1)
    clf = LogisticRegression(solver="lbfgs", max_iter=200, class_weight=None)
    clf.fit(x, y)
    return clf


def apply_sigmoid(clf, p):
    return clf.predict_proba(logit(p).reshape(-1, 1))[:, 1]


def fit_isotonic(p_train, y_train):
    y = np.asarray(y_train, float)
    if len(np.unique(y)) < 2 or len(y) < 20:
        return None
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(np.asarray(p_train, float), y)
    return iso


def apply_isotonic(iso, p):
    return iso.predict(np.asarray(p, float))


def prior_correct(p, pi):
    """Undo class_weight='balanced' prior shift using train prevalence pi.

    Balanced logistic optimizes as if pi=0.5, which shifts the intercept by
    log((1-pi)/pi). Subtracting that shift maps scores back toward the
    observed prevalence. pi must come from calibration-train rows only.
    """
    pi = float(np.clip(pi, CLIP, 1.0 - CLIP))
    shift = math.log((1.0 - pi) / pi)
    return 1.0 / (1.0 + np.exp(-(logit(p) - shift)))


def component_folds(groups, n_splits=5, seed=SEED):
    """Assign each connected component to one calibration fold.

    Deterministic: sort unique component ids, then round-robin after a
    seeded shuffle so fold sizes stay close without pulling in sklearn's
    GroupKFold (components are already the atomic unit).
    """
    uniq = np.unique(groups)
    rng = np.random.RandomState(seed)
    order = uniq.copy()
    rng.shuffle(order)
    assign = {}
    for i, g in enumerate(order):
        assign[g] = i % n_splits
    return np.array([assign[g] for g in groups])


def cross_fit(method, p, y, groups, cal_folds, min_pos, min_neg):
    """Cross-fit one calibrator. Returns calibrated probs + per-fold notes.

    A component's rows are evaluated by a calibrator fit on every other
    component. Degenerate folds fall back to the uncalibrated probability
    and are flagged, never silently scored as calibrated.
    """
    out = np.full(len(p), np.nan)
    notes = []
    for k in range(int(cal_folds.max()) + 1):
        te = cal_folds == k
        tr = ~te
        # isolation: no component on both sides
        if set(groups[tr]) & set(groups[te]):
            raise AssertionError(f"component leak in calibration fold {k}")
        ptr, ytr = p[tr], y[tr]
        pos_g, neg_g = group_support(ytr, groups[tr])
        usable = pos_g >= 2 and neg_g >= 2
        if method == "sigmoid" and usable:
            model = fit_sigmoid(ptr, ytr)
            out[te] = apply_sigmoid(model, p[te]) if model is not None else p[te]
            used = "sigmoid" if model is not None else "fallback_uncalibrated"
        elif method == "isotonic" and usable and pos_g >= min_pos and neg_g >= min_neg:
            model = fit_isotonic(ptr, ytr)
            out[te] = apply_isotonic(model, p[te]) if model is not None else p[te]
            used = "isotonic" if model is not None else "fallback_uncalibrated"
        elif method == "prior":
            pi = float(ytr.mean()) if len(ytr) else 0.5
            out[te] = prior_correct(p[te], pi)
            used = "prior"
        else:
            out[te] = p[te]
            used = "fallback_uncalibrated"
        notes.append({"fold": int(k), "method_used": used,
                      "train_pos_groups": int(pos_g), "train_neg_groups": int(neg_g)})
    return out, notes


def macro(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return fin(float(np.mean(vals))) if vals else None


def component_bootstrap(y_all, p_by_method, groups, traits, reps, seed):
    """Paired component bootstrap of macro Brier / ECE / ROC deltas vs raw.

    Resamples components, keeps all rows of a drawn component, no refit.
    """
    rng = np.random.RandomState(seed)
    uniq = np.unique(groups)
    index = {g: np.where(groups == g)[0] for g in uniq}
    methods = [m for m in p_by_method if m != "uncalibrated"]
    acc = {m: {"brier": [], "ece": [], "roc_auc": []} for m in methods}
    for _ in range(reps):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index[g] for g in draw])
        base = []
        per = {m: [] for m in methods}
        for j in range(len(traits)):
            y = y_all[idx, j]
            if len(np.unique(y)) < 2:
                continue
            base.append(metrics_block(y, p_by_method["uncalibrated"][idx, j]))
            for m in methods:
                per[m].append(metrics_block(y, p_by_method[m][idx, j]))
        b_brier, b_ece, b_roc = macro(base, "brier"), macro(base, "ece"), macro(base, "roc_auc")
        for m in methods:
            for key, base_v in (("brier", b_brier), ("ece", b_ece), ("roc_auc", b_roc)):
                v = macro(per[m], key)
                acc[m][key].append(None if v is None or base_v is None else v - base_v)
    out = {}
    for m in methods:
        out[m] = {}
        for key in ("brier", "ece", "roc_auc"):
            vals = np.array([v for v in acc[m][key] if v is not None], float)
            if len(vals) == 0:
                out[m][key] = None
                continue
            lo, hi = np.percentile(vals, [2.5, 97.5])
            out[m][key] = {
                "mean_delta": fin(vals.mean()),
                "ci95": [fin(lo), fin(hi)],
                "frac_negative": fin(float((vals < 0).mean())),
                "n_reps_used": int(len(vals)),
            }
    return out


def threshold_impact(p_raw, p_cal, thresholds, traits, cat):
    """Activation counts under frozen production thresholds, raw vs calibrated.

    Thresholds were tuned on uncalibrated probabilities, so a change here is
    a compatibility warning, not evidence the new threshold is better.
    """
    rows = []
    for j, t in enumerate(traits):
        thr = float(thresholds[t])
        a_raw = p_raw[:, j] >= thr
        a_cal = p_cal[:, j] >= thr
        rows.append({
            "trait": t,
            "category": cat[t],
            "threshold": thr,
            "active_raw": int(a_raw.sum()),
            "active_calibrated": int(a_cal.sum()),
            "delta_active": int(a_cal.sum() - a_raw.sum()),
            "flipped_fraction": fin(float((a_raw != a_cal).mean())),
        })
    by_cat = {}
    for c in sorted(set(cat.values())):
        sub = [r for r in rows if r["category"] == c]
        by_cat[c] = {
            "active_raw": int(sum(r["active_raw"] for r in sub)),
            "active_calibrated": int(sum(r["active_calibrated"] for r in sub)),
        }
    return rows, by_cat


def downstream_section_stats(p_raw, p_cal, thresholds, traits, cat):
    """Row-level downstream: how many traits/sections flip if thresholds reused.

    Uses frozen production thresholds only. No new feature inference, no model
    load. Sections = 5 provider categories. A section is active if any trait
    in that category crosses its threshold. ponytail: ceiling is template
    text equality (need generate_text_template); upgrade when token-level diff needed.
    """
    cats_sorted = sorted(set(cat.values()))
    cat_idx = {c: [i for i, t in enumerate(traits) if cat[t] == c] for c in cats_sorted}
    # per-row active trait counts
    raw_active = np.zeros(len(p_raw), dtype=int)
    cal_active = np.zeros(len(p_raw), dtype=int)
    flipped = np.zeros(len(p_raw), dtype=int)
    raw_sections = np.zeros(len(p_raw), dtype=int)
    cal_sections = np.zeros(len(p_raw), dtype=int)
    changed_sections = np.zeros(len(p_raw), dtype=int)
    for j, t in enumerate(traits):
        thr = float(thresholds[t])
        ar = p_raw[:, j] >= thr
        ac = p_cal[:, j] >= thr
        raw_active += ar.astype(int)
        cal_active += ac.astype(int)
        flipped += (ar != ac).astype(int)
    for c in cats_sorted:
        idx = cat_idx[c]
        if not idx:
            continue
        # section active if any trait in category active
        raw_sec = np.zeros(len(p_raw), dtype=bool)
        cal_sec = np.zeros(len(p_raw), dtype=bool)
        for j in idx:
            thr = float(thresholds[traits[j]])
            raw_sec |= p_raw[:, j] >= thr
            cal_sec |= p_cal[:, j] >= thr
        raw_sections += raw_sec.astype(int)
        cal_sections += cal_sec.astype(int)
        changed_sections += (raw_sec != cal_sec).astype(int)
    n = float(len(p_raw))
    return {
        "n_rows": int(n),
        "mean_active_traits_raw": fin(float(raw_active.mean())),
        "mean_active_traits_calibrated": fin(float(cal_active.mean())),
        "mean_active_sections_raw": fin(float(raw_sections.mean())),
        "mean_active_sections_calibrated": fin(float(cal_sections.mean())),
        "rows_with_any_trait_flip": int((flipped > 0).sum()),
        "rows_with_any_section_flip": int((changed_sections > 0).sum()),
        "frac_rows_trait_flip": fin(float((flipped > 0).mean())),
        "frac_rows_section_flip": fin(float((changed_sections > 0).mean())),
        "mean_traits_flipped_per_row": fin(float(flipped.mean())),
        "mean_sections_flipped_per_row": fin(float(changed_sections.mean())),
        "by_category_sections": {
            c: {
                "active_rows_raw": int(sum(1 for r in range(len(p_raw)) if any(p_raw[r, j] >= float(thresholds[traits[j]]) for j in cat_idx[c]))),
                "active_rows_calibrated": int(sum(1 for r in range(len(p_raw)) if any(p_cal[r, j] >= float(thresholds[traits[j]]) for j in cat_idx[c]))),
            } for c in cats_sorted
        },
    }


def protected_sizes():
    return {str(p): (p.stat().st_size if p.exists() else None) for p in PROTECTED}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=300)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--splits", type=int, default=5)
    args = ap.parse_args()
    t0 = time.time()
    before = protected_sizes()

    traits, cat = load_schema()
    merged = load_aligned(traits)
    problems = verify_alignment(merged, traits)
    if problems:
        raise SystemExit("alignment failed: " + "; ".join(problems))

    P = merged[traits].to_numpy(float)
    Y = np.column_stack(
        [pd.to_numeric(merged[t + "_y"], errors="coerce").to_numpy()
         if t + "_y" in merged.columns else pd.to_numeric(merged[t], errors="coerce").to_numpy()
         for t in traits]
    )
    # merge suffixes: label columns collide with prob columns, so labels land
    # in <trait>_y. Guard against reading probabilities as labels.
    if Y.shape != P.shape or np.isnan(Y).any():
        raise SystemExit("label matrix misaligned or contains NaN")
    groups = merged["component_group_id"].to_numpy()
    cal_folds = component_folds(groups, n_splits=args.splits, seed=args.seed)

    # ---- baseline reproduction vs saved phase 3.12 metrics ----
    baseline_rows = []
    for j, t in enumerate(traits):
        y, p = Y[:, j], P[:, j]
        pos_g, neg_g = group_support(y, groups)
        slope, intercept = calibration_slope_intercept(y, p)
        block = metrics_block(y, p)
        block.update({
            "trait": t,
            "category": cat[t],
            "prevalence": fin(float(y.mean())),
            "mean_predicted": fin(float(p.mean())),
            "pred_minus_prevalence": fin(float(p.mean() - y.mean())),
            "calibration_slope": slope,
            "calibration_intercept": intercept,
            "pos_groups": int(pos_g),
            "neg_groups": int(neg_g),
        })
        baseline_rows.append(block)

    saved = json.loads(PHASE312_METRICS.read_text(encoding="utf-8"))
    saved_macro = saved["strategies"]["A"]["macro_ordinary"]
    reproduced = {
        "roc_auc": macro(baseline_rows, "roc_auc"),
        "pr_auc": macro(baseline_rows, "pr_auc"),
        "brier": macro(baseline_rows, "brier"),
        "ece": macro(baseline_rows, "ece"),
        "log_loss": macro(baseline_rows, "log_loss"),
    }
    agreement = {
        k: {
            "reproduced": reproduced[k],
            "phase312": fin(saved_macro.get(k)),
            "abs_diff": fin(abs(reproduced[k] - saved_macro[k]))
            if reproduced[k] is not None and saved_macro.get(k) is not None else None,
        }
        for k in ("roc_auc", "pr_auc", "brier", "ece")
    }

    # ---- class-balance diagnosis (no refit) ----
    gaps = np.array([r["pred_minus_prevalence"] for r in baseline_rows])
    balance = {
        "note": (
            "class_weight='balanced' trains as if prevalence were 0.5, which "
            "pushes mean predicted probability above the observed prevalence "
            "for rare traits. This is a property of the training objective, "
            "not evidence the trait is common."
        ),
        "mean_prevalence": fin(float(np.mean([r["prevalence"] for r in baseline_rows]))),
        "mean_predicted": fin(float(np.mean([r["mean_predicted"] for r in baseline_rows]))),
        "mean_pred_minus_prevalence": fin(float(gaps.mean())),
        "n_traits_overpredicted": int((gaps > 0.05).sum()),
        "n_traits_underpredicted": int((gaps < -0.05).sum()),
        "mean_calibration_slope": macro(baseline_rows, "calibration_slope"),
        "mean_calibration_intercept": macro(baseline_rows, "calibration_intercept"),
    }

    # ---- cross-fit calibration ----
    methods = ("sigmoid", "isotonic", "prior")
    calibrated = {"uncalibrated": P.copy()}
    fold_notes = {m: {} for m in methods}
    for m in methods:
        out = np.zeros_like(P)
        for j, t in enumerate(traits):
            col, notes = cross_fit(
                m, P[:, j], Y[:, j], groups, cal_folds,
                MIN_ISOTONIC_POS_GROUPS, MIN_ISOTONIC_NEG_GROUPS,
            )
            out[:, j] = col
            fold_notes[m][t] = notes
        calibrated[m] = out

    # ---- per-trait + macro comparison ----
    per_trait = []
    method_rows = {m: [] for m in ("uncalibrated",) + methods}
    for j, t in enumerate(traits):
        y = Y[:, j]
        pos_g, neg_g = group_support(y, groups)
        isotonic_ok = pos_g >= MIN_ISOTONIC_POS_GROUPS and neg_g >= MIN_ISOTONIC_NEG_GROUPS
        rec = {
            "trait": t,
            "category": cat[t],
            "prevalence": baseline_rows[j]["prevalence"],
            "pos_groups": int(pos_g),
            "neg_groups": int(neg_g),
            "isotonic_eligible": bool(isotonic_ok),
        }
        best_name, best_brier = "uncalibrated", baseline_rows[j]["brier"]
        for m in ("uncalibrated",) + methods:
            block = metrics_block(y, calibrated[m][:, j])
            method_rows[m].append(block)
            for k in ("roc_auc", "brier", "ece", "log_loss"):
                rec[f"{m}_{k}"] = block[k]
            if m != "uncalibrated" and block["brier"] is not None and best_brier is not None:
                if block["brier"] < best_brier - 1e-4:
                    best_name, best_brier = m, block["brier"]
        if not isotonic_ok and best_name == "isotonic":
            best_name = "sigmoid"
        rec["selected_method"] = best_name
        rec["warning"] = None if isotonic_ok else "insufficient_independent_support_for_isotonic"
        per_trait.append(rec)

    comparison = {}
    for m, rows in method_rows.items():
        comparison[m] = {k: macro(rows, k) for k in ("roc_auc", "pr_auc", "brier", "ece", "log_loss")}

    # per-category Brier so a single bad category is visible
    by_category = {}
    for c in sorted(set(cat.values())):
        idx = [i for i, t in enumerate(traits) if cat[t] == c]
        by_category[c] = {
            m: fin(float(np.mean([method_rows[m][i]["brier"] for i in idx])))
            for m in method_rows
        }

    # ---- uncertainty (no refit) ----
    boot = component_bootstrap(Y, calibrated, groups, traits, args.bootstrap, args.seed)

    # ---- threshold compatibility (frozen production thresholds, not retuned) ----
    thresholds = json.loads(THRESH_JSON.read_text(encoding="utf-8"))["thresholds"]
    # report impact for the method with the best macro Brier among candidates
    ranked = sorted(
        ((m, comparison[m]["brier"]) for m in methods), key=lambda kv: kv[1]
    )
    best_method = ranked[0][0]
    thr_rows, thr_cat = threshold_impact(P, calibrated[best_method], thresholds, traits, cat)
    n_flipped = int(sum(1 for r in thr_rows if r["flipped_fraction"] and r["flipped_fraction"] > 0))
    downstream = downstream_section_stats(P, calibrated[best_method], thresholds, traits, cat)

    # ---- acceptance (fixed before looking at which method wins) ----
    # A candidate is "promising" only if macro Brier drops by >= 0.01 AND
    # macro ROC-AUC drops by < 0.005 AND macro log loss does not increase.
    # Promising != promotion: this protocol is exploratory.
    BRIER_GAIN = 0.01
    ROC_TOL = 0.005
    acceptance = {}
    for m in methods:
        d_brier = comparison[m]["brier"] - comparison["uncalibrated"]["brier"]
        d_roc = comparison[m]["roc_auc"] - comparison["uncalibrated"]["roc_auc"]
        d_ll = comparison[m]["log_loss"] - comparison["uncalibrated"]["log_loss"]
        ci = boot.get(m, {}).get("brier")
        ci_excludes_zero = bool(ci and ci["ci95"][1] < 0)
        acceptance[m] = {
            "delta_brier": fin(d_brier),
            "delta_roc_auc": fin(d_roc),
            "delta_log_loss": fin(d_ll),
            "brier_ci_entirely_negative": ci_excludes_zero,
            "meets_exploratory_bar": bool(
                d_brier <= -BRIER_GAIN and d_roc >= -ROC_TOL and d_ll <= 0 and ci_excludes_zero
            ),
        }
    any_promising = any(v["meets_exploratory_bar"] for v in acceptance.values())

    after = protected_sizes()
    isolation_ok = before == after

    decision = {
        "protocol": "exploratory_cross_fit_on_saved_oof",
        "not_nested_end_to_end": True,
        "production_unchanged": isolation_ok,
        "best_exploratory_method": best_method,
        "any_method_meets_exploratory_bar": any_promising,
        "promotion_justified": False,
        "promotion_blockers": [
            "second-stage calibration fit on saved OOF, not nested inside outer training folds",
            "base models were trained with class_weight=balanced; calibration does not fix ranking",
            "provider labels conflict on every duplicate group, so probability targets are noisy",
        ],
        "recommendation": (
            "retain uncalibrated baseline for ranking; do not promote a calibrator "
            "without a nested outer-fold retrain"
            if not any_promising
            else "exploratory gain only; nested outer-fold pilot required before any promotion"
        ),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUT_DIR / "calibration_config.json", {
        "phase": "3.13",
        "protocol": "exploratory cross-fit of calibrators over saved Strategy A OOF",
        "oof_source": str(OOF_A),
        "n_rows": int(len(merged)),
        "n_traits": len(traits),
        "n_components": int(len(np.unique(groups))),
        "calibration_splits": args.splits,
        "seed": args.seed,
        "bootstrap_reps": args.bootstrap,
        "min_isotonic_pos_groups": MIN_ISOTONIC_POS_GROUPS,
        "min_isotonic_neg_groups": MIN_ISOTONIC_NEG_GROUPS,
        "acceptance": {"min_brier_drop": BRIER_GAIN, "max_roc_drop": ROC_TOL},
        "label_semantics": "agreement with provider-derived labels only, not factual traits",
    })
    write_json(OUT_DIR / "baseline_calibration_report.json", {
        "reproduced_macro": reproduced,
        "agreement_with_phase312": agreement,
        "class_balance": balance,
        "per_trait": baseline_rows,
        "alignment_problems": problems,
    })
    pd.DataFrame(per_trait).to_csv(
        OUT_DIR / "per_trait_calibration.csv", index=False, encoding="utf-8"
    )
    write_json(OUT_DIR / "calibration_method_comparison.json", {
        "macro": comparison,
        "brier_by_category": by_category,
        "acceptance": acceptance,
        "isotonic_fold_fallbacks": {
            t: [n for n in notes if n["method_used"] != "isotonic"]
            for t, notes in fold_notes["isotonic"].items()
        },
    })
    write_json(OUT_DIR / "threshold_impact_report.json", {
        "method_compared": best_method,
        "note": (
            "production thresholds were tuned on uncalibrated probabilities and "
            "are NOT valid decision points for calibrated probabilities; this "
            "report only measures how many activations would flip if they were "
            "reused unchanged. Threshold file was not modified."
        ),
        "per_trait": thr_rows,
        "by_category": thr_cat,
        "traits_with_any_flip": n_flipped,
        "downstream": downstream,
        "recalibrated_thresholds": {
            "derived": False,
            "reason": "would require training-fold-only tuning to avoid OOF leakage; not derived in this exploratory cross-fit pass",
        },
    })
    write_json(OUT_DIR / "calibration_uncertainty.json", {
        "reps": args.bootstrap,
        "seed": args.seed,
        "unit": "connected component, paired, no refit",
        "delta_vs_uncalibrated": boot,
    })
    write_json(OUT_DIR / "nested_pilot_report.json", {
        "ran": False,
        "reason": (
            "not run: exploratory cross-fit did not clear the pre-registered bar "
            "of a 0.01 macro Brier drop with ROC-AUC held within 0.005 and "
            "non-increasing log loss, or a nested retrain was not justified. "
            "A nested pilot would retrain 45 logistics inside outer folds and "
            "exceeds the bounded budget without that evidence."
            if not any_promising else
            "exploratory bar met; nested pilot still requires explicit approval "
            "before a multi-hour retrain"
        ),
    })
    write_json(OUT_DIR / "calibration_decision.json", decision)

    elapsed = time.time() - t0
    summary = {
        "elapsed_sec": fin(elapsed),
        "isolation_ok": isolation_ok,
        "reproduced": reproduced,
        "comparison": comparison,
        "acceptance": acceptance,
        "decision": decision["recommendation"],
    }
    print(json.dumps(sanitize(summary), indent=2))


if __name__ == "__main__":
    main()