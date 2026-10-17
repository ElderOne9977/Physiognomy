# -*- coding: utf-8 -*-
"""Phase 3.12: training-target strategies A-D under leakage-safe hash+stem grouping.

Isolated experiment. Production models/inference untouched. Never overwrites
faces_full_features_clean.csv, labels_enriched.jsonl, oof_semantic_probs.csv,
Source/models_semantic/, or the 45-trait vocabulary.

Strategies (same estimator, same folds, same leakage-safe split; differ only in
the TRAINING target / weight):
  A  row-level binary target (baseline)
  B  row-level binary target, inverse-hash weight w_i = 1/n_h (normalized mean 1)
  C  hash-group majority-consensus target (ties documented + excluded)
  D  soft target p(h,t)=pos/total via weighted pseudo-observations
     (per-hash total pseudo weight = 1)

Usage:
  python Source/experiment_semantic_noise.py
  python Source/experiment_semantic_noise.py --strategies A,B --bootstrap 200
"""
import argparse
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import warnings
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
CSV_CLEAN = ROOT / "faces_full_features_clean.csv"
LABELS_CSV = ROOT / "semantic_labels.csv"
SCHEMA_JSON = ROOT / "semantic_schema.json"
FEATURE_COLS_NPY = SRC / "models_semantic" / "feature_cols.npy"
HASH_EXP = SRC / "experiments" / "hash_grouped_semantic"
FOLD_CSV = HASH_EXP / "cv_fold_assignments.csv"
OUT_DIR = SRC / "experiments" / "semantic_noise_phase312"

SEED = 42
FOLDS = 5
STRATEGIES = ["A", "B", "C", "D"]
# Subgroup thresholds fixed BEFORE scoring any model (conflict count 0..45).
SUBGROUP_THRESHOLDS = {"low_max": 15, "medium_max": 30}


def fin(x):
    try:
        v = float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def norm_name(value):
    s = os.path.basename(str(value).strip())
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\)\s+\(", ")(", s)
    return s.lower()


def stem_name(value):
    return re.sub(r"\.[^.]+$", "", norm_name(value))


def ece_score(y, p, n_bins=10, sample_weight=None):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    if len(y) == 0:
        return None
    edges = np.linspace(0, 1, n_bins + 1)
    w = np.asarray(sample_weight, float) if sample_weight is not None else None
    tot = w.sum() if w is not None else len(y)
    if tot == 0:
        return None
    e = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        if w is not None:
            wm = w[m].sum()
            if wm == 0:
                continue
            e += abs((y[m] * w[m]).sum() / wm - (p[m] * w[m]).sum() / wm) * wm / tot
        else:
            e += abs(y[m].mean() - p[m].mean()) * m.sum() / len(y)
    return float(e)


def sanitize(o):
    if isinstance(o, dict):
        return {k: sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitize(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return fin(float(o))
    if isinstance(o, bool):
        return bool(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float):
        return fin(o)
    return o


def write_json(path, obj):
    path.write_text(json.dumps(sanitize(obj), indent=2, ensure_ascii=False), encoding="utf-8")


def load_schema():
    sch = json.loads(SCHEMA_JSON.read_text(encoding="utf-8"))
    traits = [t["name"] for t in sch["traits"]]
    cat = {t["name"]: t["category"] for t in sch["traits"]}
    return traits, cat


# --------------------------------------------------------------------------
# Alignment (identical join key to phase 3.9/3.10/3.11; row position never used)
# --------------------------------------------------------------------------
def load_aligned():
    traits, cat = load_schema()
    df = pd.read_csv(CSV_CLEAN, encoding="utf-8-sig")
    labels = pd.read_csv(LABELS_CSV, encoding="utf-8-sig")
    img_col = [c for c in labels.columns if c.strip().lower().lstrip("\ufeff") == "image"][0]
    if img_col != "image":
        labels = labels.rename(columns={img_col: "image"})
    df["norm_key"] = df["image_id"].apply(norm_name)
    df["norm_stem"] = df["image_id"].apply(stem_name)
    labels["norm_key"] = labels["image"].apply(norm_name)
    merged = pd.merge(df, labels, on="norm_key", how="inner", suffixes=("", "_lab"))

    folds = pd.read_csv(FOLD_CSV, encoding="utf-8")
    fmap = dict(zip(folds["norm_key"].astype(str), folds["fold"]))
    gmap = dict(zip(folds["norm_key"].astype(str), folds["safe_group"]))
    smap = dict(zip(folds["norm_key"].astype(str), folds["sha256"].astype(str)))
    stmap = dict(zip(folds["norm_key"].astype(str), folds["norm_stem"].astype(str)))
    merged["fold"] = merged["norm_key"].map(fmap)
    merged["safe_group"] = merged["norm_key"].map(gmap)
    merged["sha256"] = merged["norm_key"].map(smap)
    assert merged["fold"].notna().all(), "some matched rows missing fold assignment"
    merged["fold"] = merged["fold"].astype(int)
    merged["safe_group"] = merged["safe_group"].astype(int)
    # stem comes from the same fold csv so split checks use identical strings
    merged["stem_key"] = merged["norm_key"].map(stmap)
    assert merged["stem_key"].notna().all(), "some matched rows missing stem"

    saved_cols = np.load(FEATURE_COLS_NPY, allow_pickle=True).tolist()
    base_cols = [c for c in saved_cols if not c.endswith("__missing")]
    ind_bases = [c.replace("__missing", "") for c in saved_cols if c.endswith("__missing")]
    X = pd.DataFrame(index=merged.index)
    for c in base_cols:
        X[c] = pd.to_numeric(merged[c], errors="coerce") if c in merged.columns else np.nan
    for c in ind_bases:
        X[c + "__missing"] = merged[c].isna().astype(int) if c in merged.columns else 0
    X = X[saved_cols]
    Y = np.column_stack([pd.to_numeric(merged[t], errors="coerce").values for t in traits])
    return {
        "merged": merged,
        "X": X,
        "Y": Y,
        "traits": traits,
        "cat": cat,
        "fold": merged["fold"].values,
        "groups": merged["safe_group"].values,
        "sha": merged["sha256"].astype(str).values,
        "stem": merged["stem_key"].astype(str).values,
        "image_id": merged["image_id"].values,
    }


# --------------------------------------------------------------------------
# Hash-group target / weight construction (pure, unit-tested)
# --------------------------------------------------------------------------
def group_stats(Y, keys):
    """Per hash-group positive/negative counts, soft p, majority label, conflicts."""
    keys = np.asarray(keys, dtype=object)
    uniq, inv = np.unique(keys, return_inverse=True)
    G = len(uniq)
    T = Y.shape[1]
    pos = np.zeros((G, T))
    neg = np.zeros((G, T))
    for j in range(T):
        col = Y[:, j]
        m = np.isfinite(col)
        pos[:, j] = np.bincount(inv[m], weights=(col[m] == 1).astype(float), minlength=G)
        neg[:, j] = np.bincount(inv[m], weights=(col[m] == 0).astype(float), minlength=G)
    tot = pos + neg
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.divide(pos, tot, out=np.full_like(pos, np.nan), where=tot > 0)
    maj = np.where(pos > neg, 1.0, np.where(pos < neg, 0.0, -1.0))  # -1 == tie/undefined
    maj[tot == 0] = -1.0
    conflict = (pos > 0) & (neg > 0)
    return {
        "uniq": uniq,
        "inv": inv,
        "n_groups": G,
        "pos": pos,
        "neg": neg,
        "tot": tot,
        "p": p,
        "majority": maj,
        "conflict": conflict,
        "conflict_count": conflict.sum(axis=1),
    }


def row_multiplicity(keys):
    keys = np.asarray(keys, dtype=object)
    _, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    return counts[inv].astype(float)


def inverse_hash_weights(keys):
    """w_i = 1/n_h, normalized to mean 1 (same convention as phase 3.9/3.10 expA)."""
    w = 1.0 / row_multiplicity(keys)
    return w * (len(w) / w.sum())


def majority_targets(Y, stats):
    """(n, T) majority label per row; NaN where the hash group is tied."""
    maj = stats["majority"][stats["inv"]]
    out = maj.astype(float).copy()
    out[out < 0] = np.nan
    return out


def soft_targets(stats):
    """(n, T) soft label p(h,t)=pos/total for the row's hash group."""
    return stats["p"][stats["inv"]]


def split_integrity(fold, groups, sha, stem):
    """Per-fold disjointness of train vs validation components / SHA / stems."""
    res = {}
    ok = True
    for k in range(FOLDS):
        tr = fold != k
        va = fold == k
        g_tr, g_va = set(groups[tr]), set(groups[va])
        s_tr = {s for s in sha[tr] if s not in ("nan", "None", "")}
        s_va = {s for s in sha[va] if s not in ("nan", "None", "")}
        t_tr, t_va = set(stem[tr]), set(stem[va])
        go, so, to = len(g_tr & g_va), len(s_tr & s_va), len(t_tr & t_va)
        res[f"fold_{k}"] = {
            "n_train": int(tr.sum()),
            "n_val": int(va.sum()),
            "group_overlap": go,
            "sha_overlap": so,
            "stem_overlap": to,
            "pass": go == 0 and so == 0 and to == 0,
        }
        ok = ok and res[f"fold_{k}"]["pass"]
    return ok, res


# --------------------------------------------------------------------------
# Fold-local preprocessing + per-trait OOF training
# --------------------------------------------------------------------------
def _fit_scaler(Xtr):
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    Ztr = sc.fit_transform(imp.fit_transform(Xtr))
    return imp, sc, Ztr


def _new_clf():
    return LogisticRegression(
        max_iter=500, solver="lbfgs", class_weight="balanced", random_state=SEED
    )


def train_oof(strategy, data):
    X, Y = data["X"], data["Y"]
    fold, sha = data["fold"], data["sha"]
    n, T = Y.shape
    P = np.full((n, T), np.nan)
    stats = group_stats(Y, sha)
    multit = row_multiplicity(sha)

    for k in range(FOLDS):
        tr_idx = np.where(fold != k)[0]
        va_idx = np.where(fold == k)[0]
        Xva = X.iloc[va_idx]

        if strategy in ("A", "B", "C"):
            if strategy == "C":
                tgt = majority_targets(Y, stats)
            else:
                tgt = Y.astype(float)
            imp, sc, Ztr = _fit_scaler(X.iloc[tr_idx])
            Zva = sc.transform(imp.transform(Xva))
            if strategy == "B":
                w_all = inverse_hash_weights(sha[tr_idx])
            else:
                w_all = None
            for j in range(T):
                yj = tgt[tr_idx, j]
                m = np.isfinite(yj)
                if m.sum() < 2 or len(np.unique(yj[m])) < 2:
                    continue
                sw = w_all[m] if w_all is not None else None
                clf = _new_clf()
                clf.fit(Ztr[m], yj[m], sample_weight=sw)
                P[va_idx, j] = clf.predict_proba(Zva)[:, 1]
        else:  # strategy D: soft-label weighted pseudo-observations
            tr_sha = sha[tr_idx]
            uniq, first = np.unique(tr_sha, return_index=True)
            Zg = X.iloc[tr_idx[first]]
            imp, sc, Zg_s = _fit_scaler(Zg)
            Zva = sc.transform(imp.transform(Xva))
            _, tr_inv = np.unique(tr_sha, return_inverse=True)
            g_pos = np.zeros((len(uniq), T))
            g_tot = np.zeros((len(uniq), T))
            for j in range(T):
                col = Y[tr_idx, j]
                mm = np.isfinite(col)
                g_pos[:, j] = np.bincount(tr_inv[mm], weights=(col[mm] == 1).astype(float),
                                          minlength=len(uniq))
                g_tot[:, j] = np.bincount(tr_inv[mm], minlength=len(uniq)).astype(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                gp = np.divide(g_pos, g_tot, out=np.full_like(g_pos, np.nan), where=g_tot > 0)
            for j in range(T):
                p = gp[:, j]
                keep1 = np.isfinite(p) & (p > 0)
                keep0 = np.isfinite(p) & (p < 1)
                if keep1.sum() + keep0.sum() < 2:
                    continue
                Zd = np.vstack([Zg_s[keep1], Zg_s[keep0]])
                yd = np.concatenate([np.ones(int(keep1.sum())), np.zeros(int(keep0.sum()))])
                wd = np.concatenate([p[keep1], 1.0 - p[keep0]])
                if len(np.unique(yd)) < 2:
                    continue
                clf = _new_clf()
                clf.fit(Zd, yd, sample_weight=wd)
                P[va_idx, j] = clf.predict_proba(Zva)[:, 1]
    return P


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def trait_metrics(y, p, w=None):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    mask = np.isfinite(p) & np.isfinite(y) & ((y == 0) | (y == 1))
    if mask.sum() < 20:
        return {"n_valid": int(mask.sum()), "roc_auc": None, "pr_auc": None,
                "brier": None, "ece": None, "f1_0_5": None}
    yv, pv = y[mask], p[mask]
    wv = w[mask] if w is not None else None
    if len(np.unique(yv)) < 2:
        roc = pr = None
    else:
        try:
            roc = fin(roc_auc_score(yv, pv, sample_weight=wv))
        except Exception:
            roc = None
        try:
            pr = fin(average_precision_score(yv, pv, sample_weight=wv))
        except Exception:
            pr = None
    brier = fin(np.average((pv - yv) ** 2, weights=wv) if wv is not None else np.mean((pv - yv) ** 2))
    ece = fin(ece_score(yv, pv, sample_weight=wv))
    pred = (pv >= 0.5).astype(int)
    f1 = fin(f1_score(yv, pred, zero_division=0, sample_weight=wv) if wv is not None
             else f1_score(yv, pred, zero_division=0))
    return {"n_valid": int(mask.sum()), "roc_auc": roc, "pr_auc": pr,
            "brier": brier, "ece": ece, "f1_0_5": f1,
            "prevalence": fin(yv.mean())}


def macro_and_per_trait(P, Y, traits, cat, w=None):
    per_trait = {}
    macro = {"roc_auc": [], "pr_auc": [], "brier": [], "ece": [], "f1_0_5": []}
    for j, t in enumerate(traits):
        tm = trait_metrics(Y[:, j], P[:, j], w=w)
        tm["category"] = cat[t]
        per_trait[t] = tm
        for key in macro:
            if tm.get(key) is not None:
                macro[key].append(tm[key])
    agg = {k: (fin(np.mean(v)) if v else None) for k, v in macro.items()}
    agg["n_traits_scored"] = {k: len(v) for k, v in macro.items()}
    return agg, per_trait


def macro_category(per_trait, cat):
    out = {}
    for c in sorted(set(cat.values())):
        vals = [m["roc_auc"] for t, m in per_trait.items() if cat[t] == c and m.get("roc_auc")]
        briers = [m["brier"] for t, m in per_trait.items() if cat[t] == c and m.get("brier") is not None]
        out[c] = {"n_traits": sum(1 for t in per_trait if cat[t] == c),
                  "macro_roc_auc": fin(np.mean(vals)) if vals else None,
                  "macro_brier": fin(np.mean(briers)) if briers else None}
    return out


def group_level(P, Y, soft, traits):
    """MSE/MAE of predicted probability vs soft hash target q; soft ECE."""
    mse, mae, ece = [], [], []
    per_trait = {}
    for j, t in enumerate(traits):
        q = soft[:, j]
        p = P[:, j]
        mask = np.isfinite(q) & np.isfinite(p)
        if mask.sum() == 0:
            per_trait[t] = {"mse_vs_q": None, "mae_vs_q": None, "ece_soft": None}
            continue
        d = p[mask] - q[mask]
        m = fin(np.mean(d ** 2))
        a = fin(np.mean(np.abs(d)))
        e = fin(ece_score(q[mask], p[mask]))
        per_trait[t] = {"mse_vs_q": m, "mae_vs_q": a, "ece_soft": e}
        mse.append(m)
        mae.append(a)
        ece.append(e)
    return {"macro_mse_vs_q": fin(np.mean(mse)) if mse else None,
            "macro_mae_vs_q": fin(np.mean(mae)) if mae else None,
            "macro_ece_soft": fin(np.mean(ece)) if ece else None}, per_trait


def subgroup_of_row(mult, conflict_count):
    lo, med = SUBGROUP_THRESHOLDS["low_max"], SUBGROUP_THRESHOLDS["medium_max"]
    out = np.empty(len(mult), dtype=object)
    for i in range(len(mult)):
        if mult[i] <= 1:
            out[i] = "singleton"
        elif conflict_count[i] <= lo:
            out[i] = "low"
        elif conflict_count[i] <= med:
            out[i] = "medium"
        else:
            out[i] = "high"
    return out


def subgroup_metrics(P, Y, traits, subgroup, group_idx):
    out = {}
    for sg in ["singleton", "low", "medium", "high"]:
        rows = np.where(subgroup == sg)[0]
        if len(rows) == 0:
            out[sg] = {"n_rows": 0, "macro_roc_auc": None, "macro_brier": None}
            continue
        # group-level aggregation: one value per component to respect dependence
        aucs, briers = [], []
        for j in range(Y.shape[1]):
            y, p = Y[rows, j], P[rows, j]
            mask = np.isfinite(p) & np.isfinite(y) & ((y == 0) | (y == 1))
            if mask.sum() < 20 or len(np.unique(y[mask])) < 2:
                continue
            try:
                aucs.append(fin(roc_auc_score(y[mask], p[mask])))
            except Exception:
                pass
            briers.append(fin(np.mean((p[mask] - y[mask]) ** 2)))
        out[sg] = {"n_rows": int(len(rows)),
                   "n_components": int(len(np.unique(group_idx[rows]))),
                   "macro_roc_auc": fin(np.mean(aucs)) if aucs else None,
                   "macro_brier": fin(np.mean(briers)) if briers else None}
    return out


def component_bootstrap(P_by_strategy, Y, groups, traits, n_rep, seed):
    """Paired component bootstrap (no retrain). Group ids resampled with replacement."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    G = len(uniq)
    names = list(P_by_strategy.keys())
    base = "A" if "A" in P_by_strategy else names[0]
    T = Y.shape[1]
    valid_roc = {k: [] for k in names}
    valid_brier = {k: [] for k in names}
    delta_roc = {k: [] for k in names}
    delta_brier = {k: [] for k in names}
    valid_rep = 0
    for _ in range(n_rep):
        counts = np.bincount(rng.integers(0, G, G), minlength=G)
        w = counts[inv].astype(float)
        mask = w > 0
        if mask.sum() < 20:
            continue
        roc_rep, brier_rep = {}, {}
        for k in names:
            aucs, brs = [], []
            for j in range(T):
                y, p = Y[mask, j], P_by_strategy[k][mask, j]
                m = np.isfinite(p) & np.isfinite(y) & ((y == 0) | (y == 1))
                if m.sum() < 20 or len(np.unique(y[m])) < 2:
                    continue
                try:
                    aucs.append(roc_auc_score(y[m], p[m], sample_weight=w[mask][m]))
                except Exception:
                    pass
                brs.append(np.average((p[m] - y[m]) ** 2, weights=w[mask][m]))
            if aucs:
                roc_rep[k] = float(np.mean(aucs))
                brier_rep[k] = float(np.mean(brs))
        if base not in roc_rep:
            continue
        valid_rep += 1
        for k in names:
            if k in roc_rep:
                valid_roc[k].append(roc_rep[k])
                valid_brier[k].append(brier_rep[k])
                delta_roc[k].append(roc_rep[k] - roc_rep[base])
                delta_brier[k].append(brier_rep[k] - brier_rep[base])

    def summarize(arr):
        if not arr:
            return None
        a = np.asarray(arr)
        return {"mean": fin(a.mean()), "ci95_low": fin(np.percentile(a, 2.5)),
                "ci95_high": fin(np.percentile(a, 97.5))}

    out = {"n_rep_requested": n_rep, "n_rep_valid": valid_rep, "seed": seed,
           "base_strategy": base}
    for k in names:
        out[k] = {"macro_roc_auc": summarize(valid_roc[k]),
                  "macro_brier": summarize(valid_brier[k]),
                  "delta_roc_vs_base": summarize(delta_roc[k]),
                  "delta_brier_vs_base": summarize(delta_brier[k])}
    return out


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    strategies = [s.strip().upper() for s in args.strategies.split(",") if s.strip()]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    data = load_aligned()
    Y, traits, cat = data["Y"], data["traits"], data["cat"]
    n = len(Y)
    assert Y.shape[1] == 45, f"expected 45 traits, got {Y.shape[1]}"
    print(f"Loaded {n} matched rows x {Y.shape[1]} traits; {len(np.unique(data['groups']))} components")

    ok, integ = split_integrity(data["fold"], data["groups"], data["sha"], data["stem"])
    assert ok, f"leakage-safe split FAILED: {integ}"
    print("Split integrity PASS (component/sha/stem overlap = 0 all folds)")

    stats = group_stats(Y, data["sha"])
    soft = soft_targets(stats)
    multit = row_multiplicity(data["sha"])
    conflict_count_row = stats["conflict_count"][stats["inv"]]
    subgroup = subgroup_of_row(multit, conflict_count_row)
    inv_w = inverse_hash_weights(data["sha"])

    # ---- target report (before any model is scored) ----
    maj = majority_targets(Y, stats)
    target_report = {
        "n_rows": int(n),
        "n_components": int(len(np.unique(data["groups"]))),
        "n_unique_sha": int(len(np.unique(data["sha"]))),
        "n_duplicate_sha_groups": int((stats["tot"][:, 0] > 1).sum()),
        "subgroup_thresholds": SUBGROUP_THRESHOLDS,
        "subgroup_row_counts": {sg: int((subgroup == sg).sum()) for sg in ["singleton", "low", "medium", "high"]},
        "per_trait": {},
    }
    tie_total = 0
    excl_total = 0
    prev_a, prev_c = [], []
    for j, t in enumerate(traits):
        yj = Y[:, j]
        valid = np.isfinite(yj)
        ties = int((np.isnan(maj[:, j]) & valid).sum())
        tie_total += ties
        excl_total += ties
        prev_a.append(fin(np.mean(yj[valid] == 1)) if valid.sum() else None)
        cj = maj[:, j]
        prev_c.append(fin(np.nanmean(cj == 1)))
        target_report["per_trait"][t] = {
            "category": cat[t],
            "n_valid": int(valid.sum()),
            "n_tie_rows_excluded": ties,
            "n_conflicting_groups": int(stats["conflict"][:, j].sum()),
            "mean_soft_p": fin(np.nanmean(stats["p"][:, j])),
            "prevalence_A": prev_a[-1],
            "prevalence_C": prev_c[-1],
        }
    target_report["tie_rows_excluded_total"] = tie_total
    target_report["mean_prevalence_A"] = fin(np.mean([v for v in prev_a if v is not None]))
    target_report["mean_prevalence_C"] = fin(np.mean([v for v in prev_c if v is not None]))
    target_report["conflict_group_rate"] = fin(
        float((stats["conflict_count"][stats["tot"][:, 0] > 1] > 0).sum()) / max(1, int((stats["tot"][:, 0] > 1).sum()))
    )

    # ---- train OOF per strategy ----
    oofs = {}
    for strat in strategies:
        ts = time.time()
        print(f"Training strategy {strat} ...")
        P = train_oof(strat, data)
        oofs[strat] = P
        oof_df = pd.DataFrame(P, columns=traits)
        oof_df.insert(0, "image_id", data["image_id"])
        oof_df["fold"] = data["fold"]
        oof_df["component_group_id"] = data["groups"]
        oof_df["sha256"] = data["sha"]
        oof_df.to_csv(OUT_DIR / f"oof_strategy_{strat}.csv", index=False, encoding="utf-8")
        print(f"  strategy {strat} done in {time.time() - ts:.1f}s")

    # ---- metrics (ordinary + inverse-hash validation weight) ----
    metrics = {"strategies": {}, "validation_weights": ["ordinary", "inverse_hash"]}
    per_trait_rows = []
    for strat, P in oofs.items():
        macro_o, pt_o = macro_and_per_trait(P, Y, traits, cat, w=None)
        macro_w, pt_w = macro_and_per_trait(P, Y, traits, cat, w=inv_w)
        gl_o, _ = group_level(P, Y, soft, traits)
        metrics["strategies"][strat] = {
            "macro_ordinary": macro_o,
            "macro_inverse_hash": macro_w,
            "macro_by_category_ordinary": macro_category(pt_o, cat),
            "group_level_vs_q": gl_o,
            "subgroups_ordinary": subgroup_metrics(P, Y, traits, subgroup, data["groups"]),
        }
        for j, t in enumerate(traits):
            per_trait_rows.append({
                "strategy": strat, "trait": t, "category": cat[t],
                "roc_auc": pt_o[t]["roc_auc"], "pr_auc": pt_o[t]["pr_auc"],
                "brier": pt_o[t]["brier"], "ece": pt_o[t]["ece"], "f1_0_5": pt_o[t]["f1_0_5"],
                "roc_auc_inv_w": pt_w[t]["roc_auc"], "brier_inv_w": pt_w[t]["brier"],
                "mse_vs_q": group_level(P, Y, soft, [t])[1][t]["mse_vs_q"],
                "n_valid": pt_o[t]["n_valid"],
            })
    pd.DataFrame(per_trait_rows).to_csv(OUT_DIR / "semantic_noise_per_trait.csv",
                                        index=False, encoding="utf-8")

    # ---- per-trait deltas vs A ----
    if "A" in oofs:
        base_pt = {r["trait"]: r for r in per_trait_rows if r["strategy"] == "A"}
        deltas = {}
        for strat in oofs:
            if strat == "A":
                continue
            d = []
            for r in per_trait_rows:
                if r["strategy"] != strat:
                    continue
                b = base_pt[r["trait"]]
                if r["roc_auc"] is not None and b["roc_auc"] is not None:
                    d.append({"trait": r["trait"], "delta_roc_auc": fin(r["roc_auc"] - b["roc_auc"]),
                              "delta_brier": fin(
                                  (r["brier"] - b["brier"]) if (r["brier"] is not None and b["brier"] is not None) else None)})
            d_sorted = sorted([x for x in d if x["delta_roc_auc"] is not None],
                              key=lambda x: x["delta_roc_auc"])
            deltas[strat] = {
                "mean_delta_roc_auc": fin(np.mean([x["delta_roc_auc"] for x in d_sorted])) if d_sorted else None,
                "n_traits_improved": int(sum(1 for x in d_sorted if x["delta_roc_auc"] > 0)),
                "n_traits_worse": int(sum(1 for x in d_sorted if x["delta_roc_auc"] < 0)),
                "worst": d_sorted[:5],
                "best": d_sorted[-5:],
            }
        metrics["per_trait_delta_vs_A"] = deltas

    write_json(OUT_DIR / "semantic_noise_strategy_metrics.json", metrics)
    write_json(OUT_DIR / "semantic_noise_target_report.json", target_report)

    # ---- paired component bootstrap ----
    boot = component_bootstrap(oofs, Y, data["groups"], traits, args.bootstrap, args.seed)
    boot["macro_roc_auc_point"] = {k: metrics["strategies"][k]["macro_ordinary"]["roc_auc"] for k in oofs}
    boot["macro_brier_point"] = {k: metrics["strategies"][k]["macro_ordinary"]["brier"] for k in oofs}
    write_json(OUT_DIR / "semantic_noise_bootstrap.json", boot)

    manifest = {
        "phase": "3.12",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_rows": int(n),
        "n_traits": int(Y.shape[1]),
        "n_components": int(len(np.unique(data["groups"]))),
        "strategies": list(oofs.keys()),
        "split_integrity_all_pass": bool(ok),
        "bootstrap_reps": args.bootstrap,
        "seed": args.seed,
        "elapsed_sec": fin(time.time() - t0),
    }
    write_json(OUT_DIR / "run_manifest.json", manifest)
    print(f"Wrote outputs to {OUT_DIR} in {time.time() - t0:.1f}s")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()