"""
Train semantic predictions (Stage B) on faces_full_features_clean.csv + semantic_labels.csv.

One LogisticRegression per trait, group-aware CV, fold-local preprocessing, deterministic seed.
Output: Source/semantic_train_report.json, models under Source/models_semantic/, OOF data.

Usage:
  python Source/train_semantic.py --folds 5 --seed 42 --save
"""
import argparse
import json
import os
import pickle
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import warnings
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "faces_full_features_clean.csv"
LABELS_CSV = ROOT / "semantic_labels.csv"
SCHEMA = ROOT / "semantic_schema.json"
OUTPUT_REPORT = Path(__file__).parent / "semantic_train_report.json"
OUT_DIR = Path(__file__).parent / "models_semantic"
OOF_PROBS_CSV = ROOT / "oof_semantic_probs.csv"
OOF_PROBS_NPY = ROOT / "oof_semantic_probs.npy"
OOF_META_JSON = ROOT / "oof_semantic_meta.json"

# Feature filter: keep only ratio/area_ratio/asymmetry columns (matches train.py).
def is_feature_col(name):
    return name.endswith("_ratio") or "area_ratio" in name or "asymmetry" in name


def norm_name(value):
    s = os.path.basename(str(value).strip())
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\)\s+\(", ")(", s)
    return s.lower()


def load_dataset(features="slim"):
    df = pd.read_csv(CSV, encoding="utf-8-sig")
    if features == "slim":
        cols = [c for c in df.columns if c == "image_id" or is_feature_col(c)]
        df = df[cols].copy()
    df["norm_key"] = df["image_id"].apply(norm_name)
    labels = pd.read_csv(LABELS_CSV, encoding="utf-8-sig")
    # first column may carry BOM (\"image\"); normalize its name
    img_col = [c for c in labels.columns if c.strip().lower().lstrip("\ufeff") == "image"][0]
    if img_col != "image":
        labels = labels.rename(columns={img_col: "image"})
    labels["norm_key"] = labels["image"].apply(norm_name)
    merged = pd.merge(df, labels, on="norm_key", how="inner")
    unmatched = len(df) - len(merged)
    print(f"Merged: {len(merged)} rows, {unmatched} unmatched from feature side")
    return merged, unmatched


def run_fold(X_tr, X_va, y_tr, y_va, numeric_cols, high_null_cols=None):
    if high_null_cols is None:
        tr_rate = X_tr.isna().mean()
        high_null_cols = [c for c in numeric_cols if tr_rate.get(c, 0) > 0.05]
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
    Xt_s = scaler.fit_transform(Xt_imp)
    Xv_s = scaler.transform(Xv_imp)
    return Xt_s, Xv_s, cols, imp, scaler, high_null_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=["slim", "full"], default="slim")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    np.random.seed(args.seed)

    merged, unmatched = load_dataset(args.features)
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    trait_names = [t["name"] for t in schema["traits"]]
    n_traits = len(trait_names)
    print(f"Loaded {len(merged)} rows x {n_traits} traits")

    # feature matrix
    feature_cols = [c for c in merged.columns if c not in ("image_id", "norm_key", "image") + tuple(trait_names)]
    X = merged[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(np.nan)
    groups = merged["norm_key"].values
    ids = merged["image_id"].values
    y = merged[trait_names].values.T
    print(f"X shape: {X.shape}, Y shape: {y.shape}, groups: {len(set(groups))}")

    # find high-null cols
    null_rate = X.isna().mean()
    high_null_cols = [c for c in X.columns if null_rate.get(c, 0) > 0.05]
    print(f"High-null cols: {len(high_null_cols)}")

    gkf = GroupKFold(n_splits=args.folds)
    split_iter = gkf.split(X, groups=groups)
    split_list = [(tr, va) for tr, va in split_iter]

    all_oof_probs = np.full((len(X), n_traits), np.nan)
    fold_of_row = np.full(len(X), -1, dtype=int)
    per_trait_aucs = {t: [] for t in trait_names}
    per_trait_pr = {t: [] for t in trait_names}
    per_trait_f1 = {t: [] for t in trait_names}
    per_trait_brier = {t: [] for t in trait_names}

    for fold_idx, (tr, va) in enumerate(split_list):
        print(f"Fold {fold_idx + 1}/{args.folds}")
        # group-leak guard: a normalized image key must never span two folds
        g_tr, g_va = set(groups[tr]), set(groups[va])
        assert not (g_tr & g_va), f"group leak in fold {fold_idx}: {len(g_tr & g_va)} shared keys"
        fold_of_row[va] = fold_idx
        Xt_s, Xv_s, _, _, _, _ = run_fold(
            X.iloc[tr], X.iloc[va], y[:, tr], y[:, va], feature_cols, high_null_cols
        )
        y_tr = y[:, tr]
        y_va = y[:, va]
        for ti, t in enumerate(trait_names):
            yt_tr = y_tr[ti]
            yt_va = y_va[ti]
            pos_neg = len(set(yt_tr))
            if pos_neg < 2 or sum(yt_va) < 1:
                continue
            clf = LogisticRegression(max_iter=500, solver="lbfgs", class_weight="balanced", random_state=args.seed)
            clf.fit(Xt_s, yt_tr)
            prob = clf.predict_proba(Xv_s)[:, 1]
            pred = (prob >= 0.5).astype(int)
            try:
                auc = roc_auc_score(yt_va, prob)
                pr = average_precision_score(yt_va, prob)
                f1 = f1_score(yt_va, pred, zero_division=0)
                brier = ((prob - yt_va) ** 2).mean()
            except Exception:
                continue
            per_trait_aucs[t].append(auc)
            per_trait_pr[t].append(pr)
            per_trait_f1[t].append(f1)
            per_trait_brier[t].append(brier)
            all_oof_probs[va, ti] = prob

    # OOF integrity: every row predicted at most once, coverage recorded
    row_pred_counts = (~np.isnan(all_oof_probs)).sum(axis=1)
    assert row_pred_counts.max() <= n_traits, "row received more than one prediction per trait"
    assert len(set(ids)) == len(ids), "duplicate image ids in joined dataset"

    # Save OOF CSV/NPY
    oof_df = pd.DataFrame(all_oof_probs, columns=trait_names)
    OOF_PROBS_CSV.parent.mkdir(parents=True, exist_ok=True)
    oof_df.insert(0, "image_id", ids)
    oof_df.to_csv(OOF_PROBS_CSV, index=False)
    np.save(OOF_PROBS_NPY, all_oof_probs)
    cov = float((~np.isnan(all_oof_probs)).sum(axis=1).mean())
    meta = {
        "n_rows": int(len(X)),
        "n_traits": n_traits,
        "trait_names": trait_names,
        "oof_coverage": round(cov, 4),
        "rows_with_all_traits": int((row_pred_counts == n_traits).sum()),
        "rows_with_any_trait": int((row_pred_counts > 0).sum()),
        "fold_of_row_sum": int(fold_of_row[fold_of_row >= 0].size),
        "image_ids": "column 0 of oof_semantic_probs.csv, row order identical to .npy",
    }
    OOF_META_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote OOF CSV, NPY, JSON ({OOF_PROBS_CSV})")

    # Build report
    summary = {}
    for ti, t in enumerate(trait_names):
        aucs = [x for x in per_trait_aucs[t] if not np.isnan(x)]
        prs = [x for x in per_trait_pr[t] if not np.isnan(x)]
        f1s = per_trait_f1[t]
        briers = per_trait_brier[t]
        summary[t] = {
            "auc_mean": float(np.mean(aucs)) if aucs else None,
            "auc_std": float(np.std(aucs)) if aucs else None,
            "pr_mean": float(np.mean(prs)) if prs else None,
            "f1_mean": float(np.mean(f1s)) if f1s else None,
            "brier_mean": float(np.mean(briers)) if briers else None,
            "prevalence": float(y[ti].mean()),
            "folds_scored": len(aucs),
        }
    macro_auc = float(np.mean([s["auc_mean"] for s in summary.values() if s["auc_mean"]]))
    report = {
        "dataset_size": int(len(X)),
        "group_count": int(len(set(groups))),
        "feature_count": int(X.shape[1]),
        "trait_count": n_traits,
        "trait_names": trait_names,
        "cv_method": "GroupKFold(norm_key)",
        "traits_skipped": [t for t, s in summary.items() if s["auc_mean"] is None],
        "folds": args.folds,
        "seed": args.seed,
        "per_trait": summary,
        "macro_auc": macro_auc,
        "high_null_columns": high_null_cols,
        "unmatched_dropped": unmatched,
    }
    OUTPUT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote report {OUTPUT_REPORT}, macro_auc={macro_auc:.3f}")

    if args.save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        full_X = X.copy()
        for c in high_null_cols:
            if c in full_X.columns:
                full_X[c + "__missing"] = full_X[c].isna().astype(int)
        full_cols = list(full_X.columns)
        imp = SimpleImputer(strategy="median")
        Xt_imp = imp.fit_transform(full_X[full_cols])
        scaler = StandardScaler()
        Xt_s = scaler.fit_transform(Xt_imp)
        np.save(OUT_DIR / "feature_cols.npy", np.array(full_cols))
        with open(OUT_DIR / "imputer.pkl", "wb") as f:
            pickle.dump(imp, f)
        with open(OUT_DIR / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        np.save(OUT_DIR / "trait_names.npy", np.array(trait_names))
        for t in trait_names:
            idx = trait_names.index(t)
            y_t = y[idx]
            clf = LogisticRegression(max_iter=500, solver="lbfgs", class_weight="balanced", random_state=args.seed)
            clf.fit(Xt_s, y_t)
            with open(OUT_DIR / f"logreg_{t}.pkl", "wb") as f:
                pickle.dump(clf, f)
        shutil.copy(SCHEMA, OUT_DIR / "semantic_schema.json")
        (OUT_DIR / "missing_indicator_cols.json").write_text(
            json.dumps(high_null_cols, indent=2), encoding="utf-8"
        )
        print(f"Saved models to {OUT_DIR}")


if __name__ == "__main__":
    main()