# -*- coding: utf-8 -*-
"""Phase 3.14 - Nested semantic calibration + threshold compatibility pilot.

Leakage-safe nested CV over the 6983 matched rows:
  outer 5 folds  = pre-assigned `fold` from cv_fold_assignments.csv (grouped by safe_group)
  inner 4 folds  = GroupKFold over outer-train safe_group only
  preprocessing  = median imputer + standard scaler, REFIT per split on train rows only
  base estimator = LogisticRegression(solver=lbfgs, max_iter=500, class_weight=balanced)

Calibrators (no isotonic):
  sigmoid (Platt scaling on the logit) and prior correction.
  Sigmoid fit / threshold tuning are separated: calibrator fit on inner folds [0,1],
  threshold grid search on held-out inner folds [2,3]. Final sigmoid refit on all inner OOF.

Policies (per-trait thresholds):
  A raw probs        + production thresholds (models_semantic/trait_thresholds.json)
  B calibrated probs + production thresholds
  C calibrated probs + thresholds tuned on held-out inner folds only

Downstream: real generate_text_template deterministic generators (calibrated + concise).
Exploratory pilot only. Never overwrites a production artifact (oof_semantic_probs.csv,
trait_thresholds.json). Labels are weak (agreement with provider-derived text).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import math
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
CSV_CLEAN = ROOT / "faces_full_features_clean.csv"
LABELS_CSV = ROOT / "semantic_labels.csv"
SCHEMA_JSON = ROOT / "semantic_schema.json"
FEATURE_COLS_NPY = SRC / "models_semantic" / "feature_cols.npy"
FOLD_CSV = SRC / "experiments" / "hash_grouped_semantic" / "cv_fold_assignments.csv"
THRESH_JSON = SRC / "models_semantic" / "trait_thresholds.json"
OUT_DIR = SRC / "experiments" / "semantic_calibration_phase314"

SEED = 42
N_OUTER = 5
N_INNER = 4
BOOT_REPS = 300
N_BINS = 10
CLIP = 1e-6
EXPECTED_ROWS = 6983
GRID = [round(0.30 + i * 0.05, 4) for i in range(9)]  # 0.30..0.70 step 0.05

# pilot traits fixed before running - chosen to cover prevalence / ROC / support / category
PILOT_TRAITS = ["poised", "learning_rich", "intellectual_rel", "favorable", "adaptable",
                "breakthroughs", "warm", "conservative", "collaborative"]
PILOT_RATIONALE = {
    "poised": "personality low prev 0.10 strong ROC 0.738 medium-low support",
    "learning_rich": "career very low prev 0.05 strong ROC 0.734 low support",
    "intellectual_rel": "relationships very low prev 0.048 strong ROC 0.727 low support",
    "favorable": "future_destiny medium prev 0.37 weak ROC 0.520 high support - worst performer",
    "adaptable": "personality weak ROC 0.535 medium prev, high ECE",
    "breakthroughs": "future_destiny low prev 0.079 weak ROC 0.565",
    "warm": "personality high prev 0.63 strong ROC 0.696 high support - high prevalence anchor",
    "conservative": "wealth low prev 0.073 medium ROC 0.64 low support",
    "collaborative": "career high prev 0.44 medium ROC 0.66 high support",
}


# ---------------------------------------------------------------- numeric helpers
def fin(x):
    try:
        v = float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def sanitize(o):
    if isinstance(o, dict):
        return {k: sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitize(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return fin(float(o))
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, float):
        return fin(o)
    return o


def write_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(sanitize(obj), indent=2, ensure_ascii=False, allow_nan=False),
                 encoding="utf-8")


def clip_prob(p):
    return np.clip(np.asarray(p, float), CLIP, 1 - CLIP)


def logit(p):
    p = clip_prob(p)
    return np.log(p / (1 - p))


def ece_score(y, p, weights=None, n_bins=N_BINS):
    y = np.asarray(y, float); p = np.asarray(p, float)
    if len(y) == 0:
        return None
    if weights is None:
        weights = np.ones(len(y))
    weights = np.asarray(weights, float)
    tot = weights.sum()
    if tot <= 0:
        return None
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p <= hi) if i == n_bins - 1 else (p >= lo) & (p < hi)
        w = weights[m]
        if w.sum() == 0:
            continue
        diff = abs(np.average(y[m], weights=w) - np.average(p[m], weights=w))
        e += diff * w.sum() / tot
    return float(e)


def _both_classes(y):
    return len(np.unique(y)) >= 2


def safe_auc(y, p, w=None):
    if not _both_classes(y):
        return None
    return fin(roc_auc_score(y, p, sample_weight=w))


def safe_ap(y, p, w=None):
    if not _both_classes(y):
        return None
    return fin(average_precision_score(y, p, sample_weight=w))


def safe_brier(y, p, w=None):
    if len(y) == 0:
        return None
    return fin(brier_score_loss(y, np.clip(p, 0, 1), sample_weight=w))


def safe_logloss(y, p, w=None):
    if len(y) == 0:
        return None
    return fin(log_loss(y, clip_prob(p), sample_weight=w, labels=[0, 1]))


def safe_f1(y, pred, w=None):
    return fin(f1_score(y, pred, zero_division=0, sample_weight=w))


def safe_prec(y, pred, w=None):
    return fin(precision_score(y, pred, zero_division=0, sample_weight=w))


def safe_rec(y, pred, w=None):
    return fin(recall_score(y, pred, zero_division=0, sample_weight=w))


def calibration_slope_intercept(y, p, w=None):
    y = np.asarray(y, float)
    if len(y) < 10 or not _both_classes(y):
        return None, None
    x = logit(p).reshape(-1, 1)
    try:
        clf = LogisticRegression(solver="lbfgs", max_iter=200)
        clf.fit(x, y, sample_weight=w)
    except Exception:
        return None, None
    return fin(float(clf.coef_[0, 0])), fin(float(clf.intercept_[0]))


def prior_correct(p, pi):
    p = clip_prob(p)
    pi = float(np.clip(pi, 1e-6, 1 - 1e-6))
    lp = np.log(p / (1 - p))
    adj = np.log((1 - pi) / pi)
    return 1 / (1 + np.exp(-(lp - adj)))


def fit_sigmoid(p_train, y_train):
    x = logit(p_train).reshape(-1, 1)
    try:
        clf = LogisticRegression(solver="lbfgs", max_iter=500, random_state=SEED)
        clf.fit(x, y_train)
        return clf
    except Exception:
        return None


def apply_sigmoid(clf, p):
    return np.clip(clf.predict_proba(logit(p).reshape(-1, 1))[:, 1], 0, 1)


def group_support(y, groups):
    """pos/neg = number of groups containing at least one pos / one neg row."""
    pos = 0
    neg = 0
    for g in np.unique(groups):
        yy = y[groups == g]
        has_pos = yy.max() == 1
        has_neg = yy.min() == 0
        if has_pos:
            pos += 1
        if has_neg:
            neg += 1
    return pos, neg


# ---------------------------------------------------------------- data helpers
def norm_name(v):
    s = os.path.basename(str(v).strip())
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\)\s+\(", " )(", s)
    return s.lower()


def stem_name(v):
    return re.sub(r"\.[^.]+$", "", norm_name(v))


def load_prod_thresholds():
    if not THRESH_JSON.exists():
        return {}, 0.5, "none"
    j = json.loads(THRESH_JSON.read_text(encoding="utf-8"))
    tmap = j.get("thresholds") or j.get("trait_thresholds") or {}
    return tmap, float(j.get("default_threshold", 0.5)), j.get("strategy", "per_trait")


def fit_preprocess(Xtr):
    imp = SimpleImputer(strategy="median")
    Xtr_i = imp.fit_transform(Xtr)
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr_i)
    return imp, sc


def apply_preprocess(imp, sc, Xx):
    return sc.transform(imp.transform(Xx))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outer-folds", type=int, default=N_OUTER)
    ap.add_argument("--inner-folds", type=int, default=N_INNER)
    ap.add_argument("--bootstrap", type=int, default=BOOT_REPS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--traits", nargs="*", default=PILOT_TRAITS)
    ap.add_argument("--quick", action="store_true", help="quick run: bootstrap 30")
    args = ap.parse_args()
    if args.quick and args.bootstrap == BOOT_REPS:
        args.bootstrap = 30
    traits = list(args.traits)
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    acceptance_config = {"min_brier_drop": 0.01, "max_roc_drop": 0.005,
                         "logloss_non_increasing": True, "ci_excludes_zero": True,
                         "threshold_valid_required": True}

    # --- schema / categories
    schema = json.loads(SCHEMA_JSON.read_text(encoding="utf-8"))
    cat_map = {t["name"]: t["category"] for t in schema["traits"]}
    all_cats = list(dict.fromkeys(cat_map[t["name"]] for t in schema["traits"]))
    for t in traits:
        if t not in cat_map:
            raise SystemExit(f"unknown pilot trait {t!r}")

    # --- load data (feature_cols.npy already contains __missing indicators: parity with 3.13)
    folds_df = pd.read_csv(FOLD_CSV, encoding="utf-8")
    df = pd.read_csv(CSV_CLEAN, encoding="utf-8-sig")
    saved_cols = np.load(FEATURE_COLS_NPY, allow_pickle=True).tolist()
    base_cols = [c for c in saved_cols if not c.endswith("__missing")]
    ind_bases = [c[: -len("__missing")] for c in saved_cols if c.endswith("__missing")]

    labels = pd.read_csv(LABELS_CSV, encoding="utf-8-sig")
    img_col = [c for c in labels.columns if c.strip().lower().lstrip("\ufeff") == "image"][0]
    if img_col != "image":
        labels = labels.rename(columns={img_col: "image"})
    labels["norm_key"] = labels["image"].apply(norm_name)
    df["norm_key"] = df["image_id"].apply(norm_name)
    merged = pd.merge(df, labels[["norm_key"] + traits], on="norm_key", how="inner")

    fmap = dict(zip(folds_df["norm_key"].astype(str), folds_df["fold"].astype(int)))
    gmap = dict(zip(folds_df["norm_key"].astype(str), folds_df["safe_group"].astype(int)))
    smap = dict(zip(folds_df["norm_key"].astype(str), folds_df["sha256"].astype(str)))
    stmap = dict(zip(folds_df["norm_key"].astype(str), folds_df["norm_stem"].astype(str)))
    wmap = dict(zip(folds_df["norm_key"].astype(str),
                    folds_df["inverse_hash_weight_global"].astype(float)))
    merged["fold"] = merged["norm_key"].map(fmap)
    merged["safe_group"] = merged["norm_key"].map(gmap)
    merged["sha256"] = merged["norm_key"].map(smap)
    merged["norm_stem"] = merged["norm_key"].map(stmap)
    merged["dup_weight"] = merged["norm_key"].map(wmap).fillna(1.0)
    merged = merged.dropna(subset=["fold", "safe_group"]).reset_index(drop=True)

    n = len(merged)
    groups = merged["safe_group"].to_numpy(int)
    folds = merged["fold"].to_numpy(int)
    weights = merged["dup_weight"].to_numpy(float)

    X = pd.DataFrame(index=merged.index)
    for c in base_cols:
        X[c] = pd.to_numeric(merged[c], errors="coerce") if c in merged.columns else np.nan
    for c in ind_bases:
        X[c + "__missing"] = merged[c].isna().astype(int) if c in merged.columns else 0
    X = X[saved_cols]
    Y = np.column_stack([pd.to_numeric(merged[t], errors="coerce").to_numpy(float) for t in traits])
    T = len(traits)

    # --- fold integrity (group / sha / stem overlap must be 0)
    fold_integrity = []
    leakage_ok = True
    for k in range(args.outer_folds):
        tr = folds != k
        va = folds == k
        og = len(set(groups[tr]) & set(groups[va]))
        osha = len(set(merged.loc[tr, "sha256"]) & set(merged.loc[va, "sha256"]))
        ost = len(set(merged.loc[tr, "norm_stem"]) & set(merged.loc[va, "norm_stem"]))
        ok = (og == 0 and osha == 0 and ost == 0)
        leakage_ok = leakage_ok and ok
        fold_integrity.append({"fold": int(k), "n_train": int(tr.sum()), "n_val": int(va.sum()),
                               "group_overlap": int(og), "sha_overlap": int(osha),
                               "stem_overlap": int(ost), "ok": bool(ok)})
    assign_pairs = sorted(zip(merged["image_id"].astype(str).tolist(), folds.tolist()))
    assign_hash = hashlib.sha256(json.dumps(assign_pairs, separators=(",", ":")).encode("utf-8")).hexdigest()

    old_tmap, old_dflt, old_strat = load_prod_thresholds()
    thr_old = {t: float(old_tmap.get(t, old_dflt)) for t in traits}

    P_raw = np.full((n, T), np.nan)
    P_sig = np.full((n, T), np.nan)
    P_prior = np.full((n, T), np.nan)
    thresholds_C_per_fold = [None] * args.outer_folds
    per_fold_metrics = []

    for k in range(args.outer_folds):
        tr_pos = np.where(folds != k)[0]
        va_pos = np.where(folds == k)[0]
        Xtr = X.iloc[tr_pos]
        Xva = X.iloc[va_pos]
        Ytr = Y[tr_pos]

        # ---- inner OOF over outer-train, preprocessing refit per inner split
        n_tr = len(tr_pos)
        inner_groups = groups[tr_pos]
        inner_fold_of = np.full(n_tr, -1, dtype=int)
        inner_oof = np.full((n_tr, T), np.nan)
        gkf = GroupKFold(n_splits=args.inner_folds)
        for j, (itr, iva) in enumerate(gkf.split(Xtr, Ytr[:, 0], groups=inner_groups)):
            inner_fold_of[iva] = j
            imp, sc = fit_preprocess(Xtr.iloc[itr])
            Xi_s = apply_preprocess(imp, sc, Xtr.iloc[itr])
            Xv_s = apply_preprocess(imp, sc, Xtr.iloc[iva])
            for ti in range(T):
                y_in_tr = Ytr[itr, ti]
                if len(np.unique(y_in_tr)) < 2:
                    inner_oof[iva, ti] = float(np.clip(y_in_tr.mean() if len(y_in_tr) else 0.5, 0, 1))
                else:
                    clf = LogisticRegression(max_iter=500, solver="lbfgs",
                                             class_weight="balanced", random_state=SEED)
                    clf.fit(Xi_s, y_in_tr)
                    inner_oof[iva, ti] = clf.predict_proba(Xv_s)[:, 1]

        # ---- split inner folds: calibrator fit [0..half) / threshold tune [half..end)
        half = max(1, args.inner_folds // 2)
        calib_mask = inner_fold_of < half
        tune_mask = inner_fold_of >= half
        if tune_mask.sum() == 0:
            tune_mask = calib_mask

        sigmoid_final = {}
        prior_pis = {}
        thr_C = {}
        for ti, trait in enumerate(traits):
            y_tr = Ytr[:, ti]
            ok_all = (len(np.unique(y_tr)) == 2)
            # prior correction pi (outer-train prevalence)
            prior_pis[trait] = float(np.clip(y_tr.mean() if len(y_tr) else 0.5, 1e-6, 1 - 1e-6))

            # --- final sigmoid for outer-val, fit on all cross-fitted inner OOF
            m = np.isfinite(inner_oof[:, ti])
            pos_g, neg_g = group_support(y_tr[m], inner_groups[m]) if m.sum() else (0, 0)
            usable = ok_all and pos_g >= 2 and neg_g >= 2 and m.sum() >= 20
            sigmoid_final[trait] = fit_sigmoid(inner_oof[m, ti], y_tr[m]) if usable else None

            # --- Policy C threshold tuned on held-out inner folds (calibrator fit on the other half)
            mc = np.isfinite(inner_oof[:, ti]) & calib_mask
            mt = np.isfinite(inner_oof[:, ti]) & tune_mask
            sig_c = fit_sigmoid(inner_oof[mc, ti], y_tr[mc]) if (ok_all and mc.sum() >= 20) else None
            if sig_c is None:
                p_tune = prior_correct(inner_oof[mt, ti], prior_pis[trait])
            else:
                p_tune = apply_sigmoid(sig_c, inner_oof[mt, ti])
            y_tune = y_tr[mt]
            gp, gn = group_support(y_tune, inner_groups[mt]) if mt.sum() else (0, 0)
            if gp < 5 or gn < 5 or len(np.unique(y_tune)) < 2 or len(y_tune) < 20:
                thr_C[trait] = float(thr_old[trait])  # documented fallback
            else:
                best_f1, best_thr = -1.0, 0.5
                for thr in GRID:
                    f1 = f1_score(y_tune, (p_tune >= thr).astype(int), zero_division=0)
                    if f1 > best_f1:
                        best_f1, best_thr = f1, thr
                thr_C[trait] = float(best_thr)
        thresholds_C_per_fold[k] = thr_C

        # ---- final base model on full outer-train, predict outer-val
        imp, sc = fit_preprocess(Xtr)
        Xtr_s = apply_preprocess(imp, sc, Xtr)
        Xva_s = apply_preprocess(imp, sc, Xva)
        for ti, trait in enumerate(traits):
            y_tr = Ytr[:, ti]
            if len(np.unique(y_tr)) < 2:
                p_va = np.full(len(va_pos), float(np.clip(y_tr.mean(), 0, 1)))
            else:
                clf = LogisticRegression(max_iter=500, solver="lbfgs",
                                         class_weight="balanced", random_state=SEED)
                clf.fit(Xtr_s, y_tr)
                p_va = clf.predict_proba(Xva_s)[:, 1]
            P_raw[va_pos, ti] = p_va
            sig_m = sigmoid_final[trait]
            # unified fallback: if sigmoid unusable, use prior correction (same scale as P_prior)
            P_sig[va_pos, ti] = apply_sigmoid(sig_m, p_va) if sig_m is not None else \
                prior_correct(p_va, prior_pis[trait])
            P_prior[va_pos, ti] = prior_correct(p_va, prior_pis[trait])

        per_fold_metrics.append({"fold": int(k), "n_train": int(len(tr_pos)),
                                 "n_val": int(len(va_pos))})

    P_raw = np.clip(P_raw, 0, 1)
    P_sig = np.clip(P_sig, 0, 1)
    P_prior = np.clip(P_prior, 0, 1)

    # ---- persist fold integrity + OOF (exploratory, under phase314 only)
    write_json(OUT_DIR / "nested_fold_integrity.json",
               {"n_rows": int(n), "n_traits": int(T), "traits": traits,
                "folds": fold_integrity, "leakage_ok": bool(leakage_ok),
                "per_fold": per_fold_metrics, "seed": int(args.seed),
                "outer_folds": int(args.outer_folds), "inner_folds": int(args.inner_folds),
                "fold_assignment_sha256": assign_hash,
                "expected_rows": EXPECTED_ROWS})
    oof_df = pd.DataFrame(P_raw, columns=traits)
    oof_df.insert(0, "image_id", merged["image_id"].astype(str).values)
    oof_df["fold"] = folds
    oof_df["safe_group"] = groups
    oof_df.to_csv(OUT_DIR / "nested_oof_raw.csv", index=False, encoding="utf-8")
    pd.DataFrame(P_sig, columns=traits).assign(image_id=merged["image_id"].astype(str).values)\
        .to_csv(OUT_DIR / "nested_oof_sigmoid.csv", index=False, encoding="utf-8")
    pd.DataFrame(P_prior, columns=traits).assign(image_id=merged["image_id"].astype(str).values)\
        .to_csv(OUT_DIR / "nested_oof_prior.csv", index=False, encoding="utf-8")

    # ---- metric blocks (row-weighted and duplicate-adjusted)
    def trait_metrics(y, p, w):
        slope, intercept = calibration_slope_intercept(y, p, w)
        return {"trait": None, "roc_auc": safe_auc(y, p, w), "pr_auc": safe_ap(y, p, w),
                "brier": safe_brier(y, p, w), "ece": ece_score(y, p, w),
                "log_loss": safe_logloss(y, p, w), "slope": slope, "intercept": intercept,
                "n": int(len(y)), "prevalence": fin(np.mean(y))}

    def macro_of(rows, key):
        vals = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(vals)) if vals else None

    def evaluate_block(P, w):
        rows = []
        for ti, trait in enumerate(traits):
            r = trait_metrics(Y[:, ti], P[:, ti], w)
            r["trait"] = trait
            r["category"] = cat_map[trait]
            rows.append(r)
        macro = {k: macro_of(rows, k) for k in
                 ("roc_auc", "pr_auc", "brier", "ece", "log_loss")}
        return rows, macro

    per_raw, macro_raw = evaluate_block(P_raw, None)
    per_sig, macro_sig = evaluate_block(P_sig, None)
    per_prior, macro_prior = evaluate_block(P_prior, None)
    per_raw_w, macro_raw_w = evaluate_block(P_raw, weights)
    per_sig_w, macro_sig_w = evaluate_block(P_sig, weights)
    per_prior_w, macro_prior_w = evaluate_block(P_prior, weights)

    # ---- threshold policies
    def pred_matrix(P, thr_spec):
        """thr_spec: dict trait->thr (A/B) or list-per-outer-fold (C)."""
        P_pred = np.zeros_like(P, dtype=int)
        for ti, trait in enumerate(traits):
            col = P[:, ti]
            if isinstance(thr_spec, dict):
                P_pred[:, ti] = (col >= thr_spec[trait]).astype(int)
            else:
                for k in range(args.outer_folds):
                    m = folds == k
                    P_pred[m, ti] = (col[m] >= thr_spec[k][trait]).astype(int)
        return P_pred

    def policy_metrics(P_pred, w):
        rows = []
        for ti, trait in enumerate(traits):
            y = Y[:, ti]; pred = P_pred[:, ti]
            rows.append({"trait": trait, "category": cat_map[trait],
                         "precision": safe_prec(y, pred, w), "recall": safe_rec(y, pred, w),
                         "f1": safe_f1(y, pred, w), "pos_rate": fin(pred.mean())})
        return {"macro_precision": macro_of(rows, "precision"),
                "macro_recall": macro_of(rows, "recall"),
                "macro_f1": macro_of(rows, "f1"),
                "active_per_sample": fin(P_pred.sum(axis=1).mean()),
                "pos_rate": fin(P_pred.mean()), "rows": rows}

    A_raw = policy_metrics(pred_matrix(P_raw, thr_old), None)
    B_sig = policy_metrics(pred_matrix(P_sig, thr_old), None)
    B_prior = policy_metrics(pred_matrix(P_prior, thr_old), None)
    C_sig = policy_metrics(pred_matrix(P_sig, thresholds_C_per_fold), None)
    C_prior = policy_metrics(pred_matrix(P_prior, thresholds_C_per_fold), None)

    thr_report = {
        "production_thresholds_used": thr_old,
        "policy_A_raw_old": {"macro_f1": A_raw["macro_f1"], "macro_precision": A_raw["macro_precision"],
                             "macro_recall": A_raw["macro_recall"], "active_per_sample": A_raw["active_per_sample"]},
        "policy_B_sig_old": {"macro_f1": B_sig["macro_f1"], "macro_precision": B_sig["macro_precision"],
                             "macro_recall": B_sig["macro_recall"], "active_per_sample": B_sig["active_per_sample"]},
        "policy_B_prior_old": {"macro_f1": B_prior["macro_f1"], "active_per_sample": B_prior["active_per_sample"]},
        "policy_C_sig_recal": {"thresholds_per_fold": thresholds_C_per_fold, "macro_f1": C_sig["macro_f1"],
                               "macro_precision": C_sig["macro_precision"], "macro_recall": C_sig["macro_recall"],
                               "active_per_sample": C_sig["active_per_sample"]},
        "policy_C_prior_recal": {"macro_f1": C_prior["macro_f1"], "active_per_sample": C_prior["active_per_sample"]},
        "grid": GRID,
        "threshold_stability": {t: [thresholds_C_per_fold[k][t] for k in range(args.outer_folds)] for t in traits},
        "note": ("Policy C thresholds tuned on held-out inner folds only; calibrator fit on the "
                 "complementary inner folds; traits with <5 pos or <5 neg groups over the tune set "
                 "fall back to the production threshold."),
    }

    # ---- downstream: real deterministic template generators
    sys.path.insert(0, str(SRC))
    import generate_text_template as G

    def sections_of(text):
        d = {}
        for line in text.split("\n"):
            if ":" in line:
                title, body = line.split(":", 1)
                d[title.strip()] = body.strip()
        return d

    def downstream_for(P, thr_spec):
        n = len(P)
        cal_active, con_active = [], []
        cal_words, con_words = [], []
        cat_active = {c: 0 for c in all_cats}
        unsupported = 0
        realized_total = 0
        concise_fallback = 0
        sections_by_row = []
        for i in range(n):
            probs = {t: float(P[i, ti]) for ti, t in enumerate(traits)}
            thr_map = ({t: float(thr_spec[t]) for t in traits} if isinstance(thr_spec, dict)
                       else {t: float(thr_spec[folds[i]][t]) for t in traits})
            text_cal, active_cal = G.generate_text(probs, threshold=G.DEFAULT_THRESHOLD, thresholds=thr_map)
            text_con, active_con, trace = G.generate_text_concise(probs, threshold=G.DEFAULT_THRESHOLD,
                                                                 thresholds=thr_map)
            cal_active.append(len(active_cal))
            con_active.append(len(active_con))
            cal_words.append(len(text_cal.split()))
            con_words.append(len(text_con.split()))
            for t in active_cal:
                cat_active[cat_map[t]] += 1
            # unsupported: realized concepts must belong to an active trait (template invariant)
            for cat, tr in trace.items():
                realized_total += len(tr["realized_concepts"])
                unsupported += len([c for c in tr["realized_concepts"] if c is None])
                if tr["fallback_used"]:
                    concise_fallback += 1
            sections_by_row.append(sections_of(text_con))
        empty_sections = 0
        total_sections = n * len(all_cats)
        for i in range(n):
            active_cats = set()
            for ti, t in enumerate(traits):
                thr_map_i = (thr_spec[t] if isinstance(thr_spec, dict) else thr_spec[folds[i]][t])
                if P[i, ti] >= thr_map_i:
                    active_cats.add(cat_map[t])
            empty_sections += (len(all_cats) - len(active_cats))
        return {"mean_active_traits_template": fin(np.mean(cal_active)),
                "mean_active_traits_concise": fin(np.mean(con_active)),
                "active_per_category": cat_active,
                "empty_section_rate": fin(empty_sections / total_sections),
                "concise_fallback_section_rate": fin(concise_fallback / total_sections),
                "mean_words_template": fin(np.mean(cal_words)),
                "mean_words_concise": fin(np.mean(con_words)),
                "unsupported_concept_rate": fin(unsupported / realized_total) if realized_total else 0.0,
                "_sections": sections_by_row}

    ds_A = downstream_for(P_raw, thr_old)
    ds_B = downstream_for(P_sig, thr_old)
    ds_C = downstream_for(P_sig, thresholds_C_per_fold)

    def flip_rate(sa, sb):
        flips = 0
        total = 0
        for i in range(len(sa)):
            for cat in all_cats:
                total += 1
                if sa[i].get(G.CATEGORY_TITLES[cat]) != sb[i].get(G.CATEGORY_TITLES[cat]):
                    flips += 1
        return fin(flips / total) if total else None

    downstream = {}
    for name, ds in (("policy_A_raw_old", ds_A), ("policy_B_sig_old", ds_B), ("policy_C_sig_recal", ds_C)):
        downstream[name] = {k: v for k, v in ds.items() if k != "_sections"}
    downstream["section_flip_rate_B_vs_A"] = flip_rate(ds_A["_sections"], ds_B["_sections"])
    downstream["section_flip_rate_C_vs_A"] = flip_rate(ds_A["_sections"], ds_C["_sections"])
    downstream["note"] = ("Pilot renders only 9 traits; categories without an active pilot trait "
                          "fall back to canned text (not a hallucination). More concepts or longer "
                          "text is not treated as better.")

    # ---- bootstrap over connected components (paired, no refit)
    reps = int(args.bootstrap)
    rng = np.random.RandomState(args.seed)
    uniq_groups = np.unique(groups)
    deltas = {"sigmoid": {k: [] for k in ("brier", "ece", "roc_auc", "log_loss", "f1_C")},
              "prior": {k: [] for k in ("brier", "ece", "roc_auc", "log_loss", "f1_C")}}
    null_reasons = {}

    def macro_metrics(Yb, Pb):
        br, ece, roc, ll = [], [], [], []
        for ti in range(T):
            y = Yb[:, ti]; p = Pb[:, ti]
            b = safe_brier(y, p); e = ece_score(y, p); a = safe_auc(y, p); l = safe_logloss(y, p)
            if b is not None: br.append(b)
            if e is not None: ece.append(e)
            if a is not None: roc.append(a)
            if l is not None: ll.append(l)
        return (float(np.mean(br)) if br else None, float(np.mean(ece)) if ece else None,
                float(np.mean(roc)) if roc else None, float(np.mean(ll)) if ll else None)

    def f1_policy(Yb, Pb, folds_b, thr_spec):
        Ppred = np.zeros_like(Pb, dtype=int)
        for ti, trait in enumerate(traits):
            col = Pb[:, ti]
            if isinstance(thr_spec, dict):
                Ppred[:, ti] = (col >= thr_spec[trait]).astype(int)
            else:
                for k in range(args.outer_folds):
                    m = folds_b == k
                    Ppred[m, ti] = (col[m] >= thr_spec[k][trait]).astype(int)
        f1s = []
        for ti in range(T):
            y = Yb[:, ti]
            if len(np.unique(y)) < 2:
                continue
            f1s.append(f1_score(y, Ppred[:, ti], zero_division=0))
        return float(np.mean(f1s)) if f1s else None

    for _ in range(reps):
        chosen = rng.choice(uniq_groups, size=len(uniq_groups), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in chosen]) if len(chosen) else np.array([], int)
        if len(idx) == 0:
            continue
        Yb = Y[idx]; folds_b = folds[idx]
        Prb = P_raw[idx]; Psb = P_sig[idx]; Ppb = P_prior[idx]
        br, e_r, rr, lr = macro_metrics(Yb, Prb)
        bs, e_s, rs, ls = macro_metrics(Yb, Psb)
        bp, e_p, rp, lp = macro_metrics(Yb, Ppb)
        fA = f1_policy(Yb, Prb, folds_b, thr_old)
        fCs = f1_policy(Yb, Psb, folds_b, thresholds_C_per_fold)
        fCp = f1_policy(Yb, Ppb, folds_b, thresholds_C_per_fold)
        if bs is not None and br is not None: deltas["sigmoid"]["brier"].append(bs - br)
        if e_s is not None and e_r is not None: deltas["sigmoid"]["ece"].append(e_s - e_r)
        if rs is not None and rr is not None: deltas["sigmoid"]["roc_auc"].append(rs - rr)
        if ls is not None and lr is not None: deltas["sigmoid"]["log_loss"].append(ls - lr)
        if fCs is not None and fA is not None: deltas["sigmoid"]["f1_C"].append(fCs - fA)
        if bp is not None and br is not None: deltas["prior"]["brier"].append(bp - br)
        if e_p is not None and e_r is not None: deltas["prior"]["ece"].append(e_p - e_r)
        if rp is not None and rr is not None: deltas["prior"]["roc_auc"].append(rp - rr)
        if lp is not None and lr is not None: deltas["prior"]["log_loss"].append(lp - lr)
        if fCp is not None and fA is not None: deltas["prior"]["f1_C"].append(fCp - fA)

    def ci(arr, label):
        if len(arr) < 10:
            null_reasons[label] = f"only {len(arr)} usable bootstrap reps (need >=10)"
            return [None, None]
        return [fin(float(np.percentile(arr, 2.5))), fin(float(np.percentile(arr, 97.5)))]

    def delta_block(metric):
        arr = deltas["sigmoid"][metric]
        return {"mean": fin(np.mean(arr)) if arr else None, "ci95": ci(arr, f"sigmoid_{metric}"),
                "frac_negative": fin(np.mean([x < 0 for x in arr])) if arr else None,
                "n_usable": len(arr)}

    def delta_block_prior(metric):
        arr = deltas["prior"][metric]
        return {"mean": fin(np.mean(arr)) if arr else None, "ci95": ci(arr, f"prior_{metric}"),
                "n_usable": len(arr)}

    bootstrap_report = {
        "reps_requested": reps, "seed": int(args.seed),
        "unit": "connected component (safe_group), paired, no refit, pilot traits only",
        "deltas": {
            "sigmoid": {m: delta_block(m) for m in ("brier", "ece", "roc_auc", "log_loss", "f1_C")},
            "prior": {m: delta_block_prior(m) for m in ("brier", "ece", "roc_auc", "log_loss", "f1_C")},
        },
        "null_reasons": null_reasons,
    }

    # ---- decision
    def ci_excludes_zero(pair):
        if not pair or pair[0] is None or pair[1] is None:
            return None
        return (pair[0] < 0 and pair[1] < 0) or (pair[0] > 0 and pair[1] > 0)

    brier_drop_sig = (macro_raw["brier"] - macro_sig["brier"]) \
        if macro_raw["brier"] is not None and macro_sig["brier"] is not None else None
    roc_drop_sig = (macro_raw["roc_auc"] - macro_sig["roc_auc"]) \
        if macro_raw["roc_auc"] is not None and macro_sig["roc_auc"] is not None else None
    logloss_improved = (macro_sig["log_loss"] <= macro_raw["log_loss"]) \
        if macro_sig["log_loss"] is not None and macro_raw["log_loss"] is not None else None
    brier_ci = bootstrap_report["deltas"]["sigmoid"]["brier"]["ci95"]
    roc_ci = bootstrap_report["deltas"]["sigmoid"]["roc_auc"]["ci95"]
    thr_valid = all(0.0 <= thr_map[t] <= 1.0 for thr_map in thresholds_C_per_fold for t in traits)

    supported = bool(
        brier_drop_sig is not None and brier_drop_sig >= acceptance_config["min_brier_drop"]
        and roc_drop_sig is not None and roc_drop_sig < acceptance_config["max_roc_drop"]
        and logloss_improved is True
        and ci_excludes_zero(brier_ci) is True
        and thr_valid
    )
    decision = {
        "sigmoid_independently_supported": supported,
        "threshold_compatibility_established": bool(
            C_sig["macro_f1"] is not None and A_raw["macro_f1"] is not None
            and C_sig["macro_f1"] >= A_raw["macro_f1"] - 0.02),
        "ready_for_wider_45_trait_eval": False,
        "ready_for_production_promotion": False,
        "reasons": ["pilot traits only; full 45-trait nested validation still required",
                    "Policy C thresholds low-support traits fall back to production",
                    "labels are weak provider-derived agreement, not ground truth",
                    "no production artifact written or overwritten"],
        "acceptance": acceptance_config,
        "observed": {"brier_drop_sig": fin(brier_drop_sig), "roc_drop_sig": fin(roc_drop_sig),
                     "logloss_improved": logloss_improved, "brier_ci": brier_ci, "roc_ci": roc_ci,
                     "thresholds_valid": bool(thr_valid), "leakage_ok": bool(leakage_ok)},
    }

    write_json(OUT_DIR / "nested_calibration_config.json",
               {"phase": "3.14", "pilot_traits": traits, "rationale": PILOT_RATIONALE,
                "outer_folds": int(args.outer_folds), "inner_folds": int(args.inner_folds),
                "seed": int(args.seed), "bootstrap_reps": reps,
                "base_estimator": "LogisticRegression(solver=lbfgs,max_iter=500,class_weight=balanced)",
                "calibrators": ["sigmoid", "prior_correction"], "isotonic": False,
                "preprocess": "SimpleImputer(median)+StandardScaler refit per split (train only)",
                "acceptance": acceptance_config, "grid": GRID,
                "label_semantics": "agreement with provider-derived weak labels only"})
    write_json(OUT_DIR / "nested_calibration_metrics.json",
               {"per_trait_raw": per_raw, "macro_raw": macro_raw,
                "per_trait_sigmoid": per_sig, "macro_sigmoid": macro_sig,
                "per_trait_prior": per_prior, "macro_prior": macro_prior,
                "macro_raw_duplicate_adjusted": macro_raw_w,
                "macro_sigmoid_duplicate_adjusted": macro_sig_w,
                "macro_prior_duplicate_adjusted": macro_prior_w,
                "threshold_comparison": thr_report, "downstream": downstream})

    rows = []
    for a, b, c, aw, bw in zip(per_raw, per_sig, per_prior, per_raw_w, per_sig_w):
        rows.append({"trait": a["trait"], "category": a["category"], "prevalence": a["prevalence"],
                     "roc_raw": a["roc_auc"], "roc_sig": b["roc_auc"], "roc_prior": c["roc_auc"],
                     "roc_raw_dup_adj": aw["roc_auc"], "roc_sig_dup_adj": bw["roc_auc"],
                     "brier_raw": a["brier"], "brier_sig": b["brier"], "brier_prior": c["brier"],
                     "ece_raw": a["ece"], "ece_sig": b["ece"],
                     "logloss_raw": a["log_loss"], "logloss_sig": b["log_loss"],
                     "slope_raw": a["slope"], "intercept_raw": a["intercept"]})
    pd.DataFrame(rows).to_csv(OUT_DIR / "nested_calibration_per_trait.csv", index=False, encoding="utf-8")
    write_json(OUT_DIR / "nested_threshold_comparison.json", thr_report)
    write_json(OUT_DIR / "nested_downstream_impact.json", downstream)
    write_json(OUT_DIR / "nested_bootstrap_report.json", bootstrap_report)
    write_json(OUT_DIR / "nested_calibration_decision.json", decision)

    print(json.dumps(sanitize({
        "elapsed_sec": fin(time.time() - t0), "leakage_ok": leakage_ok,
        "n_rows": n, "traits": traits,
        "macro_raw": macro_raw, "macro_sig": macro_sig, "macro_prior": macro_prior,
        "policy_A_f1": A_raw["macro_f1"], "policy_C_sig_f1": C_sig["macro_f1"],
        "decision": decision}), indent=2))


if __name__ == "__main__":
    main()