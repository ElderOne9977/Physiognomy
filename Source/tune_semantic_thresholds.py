"""
Leakage-safe threshold calibration for the semantic trait classifiers.

Nested cross-validation:
  outer GroupKFold -> for each outer fold:
      inner GroupKFold on outer-train -> inner OOF probabilities
      tune thresholds on inner OOF (never sees outer-val labels)
      fit final model on full outer-train, predict outer-val
      apply that fold's tuned thresholds to outer-val

Every row's outer-val prediction therefore comes from a model that never
trained on it, and from thresholds tuned without its labels.

Three strategies compared (objective = macro semantic F1):
    A. global        one threshold for all traits
    B. per-category  one threshold per category
    C. per-trait     independent threshold per trait

Usage:
  python Source/tune_semantic_thresholds.py
  python Source/tune_semantic_thresholds.py --outer-folds 5 --inner-folds 4 ^
      --grid-start 0.30 --grid-stop 0.70 --grid-step 0.05
"""
import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import warnings
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "faces_full_features_clean.csv"
LABELS_CSV = ROOT / "semantic_labels.csv"
SCHEMA = ROOT / "semantic_schema.json"

DEFAULT_REPORT = Path(__file__).parent / "semantic_threshold_report.json"
DEFAULT_CSV_REPORT = Path(__file__).parent / "semantic_threshold_report.csv"
DEFAULT_THRESHOLDS = Path(__file__).parent / "models_semantic" / "trait_thresholds.json"

CATEGORIES = ["personality", "career", "relationships", "future_destiny", "wealth"]
WEAK_AUC = 0.55


# --- helpers mirrored from train_semantic.py (single source of truth there) ---
def is_feature_col(name):
    return name.endswith("_ratio") or "area_ratio" in name or "asymmetry" in name


def norm_name(value):
    s = os.path.basename(str(value).strip())
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\)\s+\(", ")(", s)
    return s.lower()


def run_fold(X_tr, X_va, numeric_cols, high_null_cols):
    """Per-fold missing-indicator + median impute + standardize (train-only fit)."""

    def add_ind(df):
        df = df.copy()
        for c in high_null_cols:
            if c in df.columns:
                df[c + "__missing"] = df[c].isna().astype(int)
        return df

    Xt = add_ind(X_tr)
    Xv = add_ind(X_va)
    cols = list(Xt.columns)
    imp = SimpleImputer(strategy="median")
    Xt_imp = imp.fit_transform(Xt[cols])
    Xv_imp = imp.transform(Xv[cols])
    scaler = StandardScaler()
    return scaler.fit_transform(Xt_imp), scaler.transform(Xv_imp)


def make_grid(start, stop, step):
    n = int(round((stop - start) / step)) + 1
    return [round(start + i * step, 4) for i in range(n)]


def macro_f1(y_true, y_pred):
    return float(np.mean([f1_score(y_true[i], y_pred[i], zero_division=0)
                          for i in range(y_true.shape[0])]))


def macro_p(y_true, y_pred):
    return float(np.mean([precision_score(y_true[i], y_pred[i], zero_division=0)
                          for i in range(y_true.shape[0])]))


def macro_r(y_true, y_pred):
    return float(np.mean([recall_score(y_true[i], y_pred[i], zero_division=0)
                          for i in range(y_true.shape[0])]))


def load_data(features):
    df = pd.read_csv(CSV, encoding="utf-8-sig")
    if features == "slim":
        df = df[[c for c in df.columns if c == "image_id" or is_feature_col(c)]].copy()
    df["norm_key"] = df["image_id"].apply(norm_name)
    labels = pd.read_csv(LABELS_CSV, encoding="utf-8-sig")
    img_col = [c for c in labels.columns if c.strip().lower().lstrip("\ufeff") == "image"][0]
    if img_col != "image":
        labels = labels.rename(columns={img_col: "image"})
    labels["norm_key"] = labels["image"].apply(norm_name)
    merged = pd.merge(df, labels, on="norm_key", how="inner")

    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    trait_names = [t["name"] for t in schema["traits"]]
    trait_cat = {t["name"]: t["category"] for t in schema["traits"]}
    feature_cols = [c for c in merged.columns
                    if c not in ("image_id", "norm_key", "image") + tuple(trait_names)]
    X = merged[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = merged[trait_names].values.T.astype(int)
    groups = merged["norm_key"].values
    ids = merged["image_id"].values
    return merged, X, y, groups, ids, trait_names, trait_cat, feature_cols, schema


def main():
    ap = argparse.ArgumentParser(description="Leakage-safe semantic threshold calibration")
    ap.add_argument("--features", choices=["slim", "full"], default="slim")
    ap.add_argument("--outer-folds", type=int, default=5)
    ap.add_argument("--inner-folds", type=int, default=4)
    ap.add_argument("--grid-start", type=float, default=0.30)
    ap.add_argument("--grid-stop", type=float, default=0.70)
    ap.add_argument("--grid-step", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--baseline-threshold", type=float, default=0.5)
    ap.add_argument("--report", default=str(DEFAULT_REPORT))
    ap.add_argument("--csv-report", default=str(DEFAULT_CSV_REPORT))
    ap.add_argument("--save-thresholds", default=str(DEFAULT_THRESHOLDS))
    ap.add_argument("--no-save-thresholds", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    np.random.seed(args.seed)
    grid = make_grid(args.grid_start, args.grid_stop, args.grid_step)
    print(f"grid: {len(grid)} values {grid[0]}..{grid[-1]} step {args.grid_step}")

    merged, X, y, groups, ids, trait_names, trait_cat, feature_cols, schema = load_data(args.features)
    n_traits = len(trait_names)
    n_rows = X.shape[0]
    n_groups = len(set(groups))
    null_rate = X.isna().mean()
    high_null_cols = [c for c in X.columns if null_rate.get(c, 0) > 0.05]
    cat_idx = {c: [i for i, t in enumerate(trait_names) if trait_cat[t] == c]
               for c in CATEGORIES}
    print(f"data {n_rows}x{X.shape[1]} feats, {n_traits} traits, {n_groups} groups, "
          f"high_null {len(high_null_cols)}")

    def fit_predict(tr_idx, va_idx):
        """Fit 45 logreg on tr_idx, return (n_traits, len(va_idx)) probs for va_idx."""
        Xt_s, Xv_s = run_fold(X.iloc[tr_idx], X.iloc[va_idx], feature_cols, high_null_cols)
        P = np.full((n_traits, len(va_idx)), np.nan)
        for ti in range(n_traits):
            yt = y[ti, tr_idx]
            if len(set(yt)) < 2:
                continue
            clf = LogisticRegression(max_iter=500, solver="lbfgs",
                                     class_weight="balanced", random_state=args.seed)
            clf.fit(Xt_s, yt)
            P[ti] = clf.predict_proba(Xv_s)[:, 1]
        return P

    outer = list(GroupKFold(n_splits=args.outer_folds).split(X, groups=groups))
    oof_probs = np.full((n_traits, n_rows), np.nan)
    row_assign = np.zeros(n_rows, dtype=int)
    per_fold = []
    fold_trait_thr = defaultdict(list)
    fold_cat_thr = defaultdict(list)
    fold_global_thr = []

    for fi, (tr, va) in enumerate(outer):
        g_tr, g_va = set(groups[tr]), set(groups[va])
        overlap = g_tr & g_va
        assert not overlap, f"outer fold {fi}: group overlap {len(overlap)}"
        assert not (set(tr) & set(va)), f"outer fold {fi}: row overlap"
        row_assign[va] += 1

        # inner OOF on outer-train
        g_inner = groups[tr]
        inner_oof = np.full((n_traits, len(tr)), np.nan)
        for itr, iva in GroupKFold(n_splits=args.inner_folds).split(X.iloc[tr], groups=g_inner):
            ig_tr, ig_va = set(g_inner[itr]), set(g_inner[iva])
            assert not (ig_tr & ig_va), f"inner fold overlap {len(ig_tr & ig_va)}"
            inner_oof[:, iva] = fit_predict(tr[itr], tr[iva])
        assert not np.isnan(inner_oof).any(), f"inner OOF gaps in outer fold {fi}"

        y_tr = y[:, tr]
        # tune on inner OOF only (outer-val labels never touched)
        g_best = max(grid, key=lambda t: macro_f1(y_tr, (inner_oof >= t).astype(int)))
        c_best = {c: max(grid, key=lambda t: macro_f1(y_tr[cat_idx[c]],
                                                       (inner_oof[cat_idx[c]] >= t).astype(int)))
                  for c in CATEGORIES}
        t_best = [max(grid, key=lambda t: f1_score(y_tr[i], (inner_oof[i] >= t).astype(int),
                                                   zero_division=0))
                  for i in range(n_traits)]

        fold_global_thr.append(g_best)
        for c in CATEGORIES:
            fold_cat_thr[c].append(c_best[c])
        for i, t in enumerate(trait_names):
            fold_trait_thr[t].append(t_best[i])

        # predict outer-val with model fit on full outer-train
        P_va = fit_predict(tr, va)
        assert np.isfinite(P_va).all(), f"non-finite prob outer fold {fi}"
        assert ((P_va >= 0) & (P_va <= 1)).all(), f"prob out of [0,1] outer fold {fi}"
        oof_probs[:, va] = P_va
        y_va = y[:, va]

        pred_base = (P_va >= args.baseline_threshold).astype(int)
        pred_a = (P_va >= g_best).astype(int)
        pred_b = np.zeros_like(pred_a)
        for c in CATEGORIES:
            pred_b[cat_idx[c]] = (P_va[cat_idx[c]] >= c_best[c]).astype(int)
        pred_c = np.zeros_like(pred_a)
        for i in range(n_traits):
            pred_c[i] = (P_va[i] >= t_best[i]).astype(int)

        def m(pred):
            return {
                "macro_f1": round(macro_f1(y_va, pred), 4),
                "macro_precision": round(macro_p(y_va, pred), 4),
                "macro_recall": round(macro_r(y_va, pred), 4),
                "micro_f1": round(float(f1_score(y_va.ravel(), pred.ravel(), zero_division=0)), 4),
                "avg_predicted_traits_per_sample": round(float(pred.sum(0).mean()), 4),
            }

        fold_rec = {
            "fold": fi,
            "n_train": int(len(tr)),
            "n_val": int(len(va)),
            "outer_group_overlap": int(len(overlap)),
            "global_threshold": float(g_best),
            "category_thresholds": {c: float(c_best[c]) for c in CATEGORIES},
            "trait_thresholds": {trait_names[i]: float(t_best[i]) for i in range(n_traits)},
            "baseline_0.5": m(pred_base),
            "global": m(pred_a),
            "per_category": m(pred_b),
            "per_trait": m(pred_c),
        }
        per_fold.append(fold_rec)
        print(f"fold {fi}: base {fold_rec['baseline_0.5']['macro_f1']} "
              f"A {fold_rec['global']['macro_f1']} "
              f"B {fold_rec['per_category']['macro_f1']} "
              f"C {fold_rec['per_trait']['macro_f1']}")

    # --- integrity ---
    assert not np.isnan(oof_probs).any(), "OOF missing predictions"
    assert (row_assign == 1).all(), "duplicate/missing row assignment across outer folds"

    # strictly leakage-safe cross-fitted metrics = mean over outer folds
    def agg_leakage_safe(policy_key):
        keys = ["macro_f1", "macro_precision", "macro_recall", "micro_f1",
                "avg_predicted_traits_per_sample"]
        out = {}
        for k in keys:
            vals = [f[policy_key][k] for f in per_fold]
            out[k + "_mean"] = round(float(np.mean(vals)), 4)
            out[k + "_std"] = round(float(np.std(vals)), 4)
        return out

    lf_safe = {k: agg_leakage_safe(k) for k in ["baseline_0.5", "global", "per_category", "per_trait"]}
    best_key = max(lf_safe, key=lambda k: lf_safe[k]["macro_f1_mean"])
    print("\n=== leakage-safe cross-fitted (mean over outer folds) ===")
    for k in ["baseline_0.5", "global", "per_category", "per_trait"]:
        print(f"{k:14s} F1={lf_safe[k]['macro_f1_mean']:.4f}+/-{lf_safe[k]['macro_f1_std']:.4f} "
              f"P={lf_safe[k]['macro_precision_mean']:.4f} "
              f"R={lf_safe[k]['macro_recall_mean']:.4f} "
              f"micro={lf_safe[k]['micro_f1_mean']:.4f} "
              f"avg_pred={lf_safe[k]['avg_predicted_traits_per_sample_mean']}")
    print(f"selected strategy: {best_key}")

    # --- final policy = median of per-fold tuned thresholds ---
    final_global = round(float(np.median(fold_global_thr)), 4)
    final_cat = {c: round(float(np.median(fold_cat_thr[c])), 4) for c in CATEGORIES}
    final_trait = {t: round(float(np.median(fold_trait_thr[t])), 4) for t in trait_names}

    def apply_policy(policy):
        pred = np.zeros_like(oof_probs, dtype=int)
        if policy == "baseline_0.5":
            return (oof_probs >= args.baseline_threshold).astype(int)
        if policy == "global":
            return (oof_probs >= final_global).astype(int)
        if policy == "per_category":
            for c in CATEGORIES:
                pred[cat_idx[c]] = (oof_probs[cat_idx[c]] >= final_cat[c]).astype(int)
            return pred
        for i, t in enumerate(trait_names):
            pred[i] = (oof_probs[i] >= final_trait[t]).astype(int)
        return pred

    preds = {k: apply_policy(k) for k in ["baseline_0.5", "global", "per_category", "per_trait"]}

    def full_metrics(pred):
        return {
            "macro_precision": round(macro_p(y, pred), 4),
            "macro_recall": round(macro_r(y, pred), 4),
            "macro_f1": round(macro_f1(y, pred), 4),
            "micro_f1": round(float(f1_score(y.ravel(), pred.ravel(), zero_division=0)), 4),
            "avg_predicted_traits_per_sample": round(float(pred.sum(0).mean()), 4),
            "avg_true_traits_per_sample": round(float(y.sum(0).mean()), 4),
        }

    oof_metrics = {k: full_metrics(v) for k, v in preds.items()}
    print("\n=== full OOF aggregate (median-of-folds policy, reference only) ===")
    for k, v in oof_metrics.items():
        print(f"{k:14s} F1={v['macro_f1']:.4f} P={v['macro_precision']:.4f} "
              f"R={v['macro_recall']:.4f} avg_pred={v['avg_predicted_traits_per_sample']}")

    best_pred = preds[best_key]

    # --- per-trait detail under best policy ---
    per_trait = {}
    for i, t in enumerate(trait_names):
        yt, yp, pp = y[i], best_pred[i], oof_probs[i]
        tp = int(((yp == 1) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())
        tn = int(((yp == 0) & (yt == 0)).sum())
        try:
            auc = round(float(roc_auc_score(yt, pp)), 4)
        except Exception:
            auc = None
        try:
            prauc = round(float(average_precision_score(yt, pp)), 4)
        except Exception:
            prauc = None
        thr = (final_trait[t] if best_key == "per_trait"
               else final_global if best_key == "global"
               else final_cat[trait_cat[t]] if best_key == "per_category"
               else args.baseline_threshold)
        per_trait[t] = {
            "category": trait_cat[t],
            "prevalence": round(float(yt.mean()), 4),
            "roc_auc": auc,
            "pr_auc": prauc,
            "threshold": round(float(thr), 4),
            "precision": round(precision_score(yt, yp, zero_division=0), 4),
            "recall": round(recall_score(yt, yp, zero_division=0), 4),
            "f1": round(f1_score(yt, yp, zero_division=0), 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "predicted_positive_rate": round(float(yp.mean()), 4),
            "weak_signal": bool(auc is not None and auc < WEAK_AUC),
        }

    # --- per-category metrics under best policy ---
    cat_metrics = {}
    for c in CATEGORIES:
        idx = cat_idx[c]
        ys, ps = y[idx], best_pred[idx]
        aucs = [per_trait[trait_names[i]]["roc_auc"] for i in idx]
        prs = [per_trait[trait_names[i]]["pr_auc"] for i in idx]
        cat_metrics[c] = {
            "trait_count": len(idx),
            "macro_roc_auc": round(float(np.mean([a for a in aucs if a is not None])), 4),
            "macro_pr_auc": round(float(np.mean([p for p in prs if p is not None])), 4),
            "semantic_precision": round(macro_p(ys, ps), 4),
            "semantic_recall": round(macro_r(ys, ps), 4),
            "semantic_f1": round(macro_f1(ys, ps), 4),
            "avg_true_traits_per_sample": round(float(ys.sum(0).mean()), 4),
            "avg_predicted_traits_per_sample": round(float(ps.sum(0).mean()), 4),
        }

    # --- trait-count distribution ---
    def count_dist(v):
        c = v.sum(0)
        d = {str(k): int((c == k).sum()) for k in [0, 1, 2, 3, 4]}
        d["5+"] = int((c >= 5).sum())
        return {"histogram": d, "mean": round(float(c.mean()), 3),
                "median": round(float(np.median(c)), 3),
                "p90": round(float(np.percentile(c, 90)), 3)}

    dists = {k: count_dist(v) for k, v in preds.items()}
    dists = {"true": count_dist(y), **dists}

    weak = [t for t in per_trait if per_trait[t]["weak_signal"]]
    ranked = sorted([t for t in per_trait if per_trait[t]["roc_auc"] is not None],
                    key=lambda t: -per_trait[t]["roc_auc"])

    report = {
        "generated_by": "tune_semantic_thresholds.py",
        "cv_method": "nested GroupKFold (outer=model fit, inner=threshold tuning)",
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "seed": args.seed,
        "group_key": "norm_key",
        "threshold_grid": grid,
        "selection_objective": "macro_semantic_f1",
        "features": args.features,
        "n_rows": int(n_rows),
        "n_groups": n_groups,
        "n_traits": n_traits,
        "high_null_columns": len(high_null_cols),
        "leakage_safe_results": lf_safe,
        "selected_strategy": best_key,
        "oof_aggregate_results": oof_metrics,
        "selected_thresholds": {
            "global": final_global,
            "per_category": final_cat,
            "per_trait": final_trait,
        },
        "per_trait": per_trait,
        "per_category": cat_metrics,
        "trait_count_distribution": dists,
        "weak_signal_traits": weak,
        "weak_signal_criteria": f"roc_auc < {WEAK_AUC}",
        "top_5_strongest": ranked[:5],
        "top_5_weakest": ranked[-5:][::-1],
        "per_fold": per_fold,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    rp = Path(args.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {rp}")

    rows = [{
        "trait": t,
        "category": per_trait[t]["category"],
        "prevalence": per_trait[t]["prevalence"],
        "roc_auc": per_trait[t]["roc_auc"],
        "pr_auc": per_trait[t]["pr_auc"],
        "threshold": per_trait[t]["threshold"],
        "precision": per_trait[t]["precision"],
        "recall": per_trait[t]["recall"],
        "f1": per_trait[t]["f1"],
        "predicted_positive_rate": per_trait[t]["predicted_positive_rate"],
        "weak_signal": per_trait[t]["weak_signal"],
    } for t in trait_names]
    cp = Path(args.csv_report)
    pd.DataFrame(rows).to_csv(cp, index=False)
    print(f"wrote {cp}")

    if not args.no_save_thresholds:
        policy = {
            "version": 1,
            "strategy": best_key,
            "selection_metric": "cross_fitted_semantic_macro_f1",
            "default_threshold": 0.5,
            "thresholds": (final_trait if best_key == "per_trait"
                           else final_cat if best_key == "per_category"
                           else {"global": final_global} if best_key == "global"
                           else {"global": args.baseline_threshold}),
            "trait_thresholds": final_trait,
            "category_thresholds": final_cat,
            "global_threshold": final_global,
            "schema_version": schema.get("version", 1),
            "trait_order": trait_names,
            "metadata": {
                "seed": args.seed,
                "grid": grid,
                "cv_method": f"nested GroupKFold outer={args.outer_folds} inner={args.inner_folds}",
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "selected_strategy": best_key,
                "leakage_safe_macro_f1": lf_safe[best_key]["macro_f1_mean"],
                "baseline_macro_f1": lf_safe["baseline_0.5"]["macro_f1_mean"],
            },
        }
        sp = Path(args.save_thresholds)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(policy, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {sp} (strategy={best_key})")

    print(f"done in {report['elapsed_sec']}s")


if __name__ == "__main__":
    main()