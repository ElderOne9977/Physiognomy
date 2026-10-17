"""
Train physiognomy probe on faces_full_features_clean.csv + labels_enriched.jsonl.

No global imputation. Impute per-fold (median fit on train only) + missing indicator
for high-null cols (>5%). GroupKFold by stem to avoid leakage from ext-variant dupes.
Labels = multi-label binary traits mined from personality text.

Usage:
  python Source/train.py
  python Source/train.py --features slim --folds 5

Outputs: Source/train_report.json, models under Source/models/ if --save
"""
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import warnings
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    StratifiedGroupKFold = None
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

ROOT = Path(__file__).resolve().parents[1]
CLEAN_CSV = ROOT / "faces_full_features_clean.csv"
FALLBACK_CSV = ROOT / "faces_full_features.csv"
JSONL = ROOT / "labels_enriched.jsonl"
REPORT = Path(__file__).parent / "train_report.json"

PRED_CATS = ["personality", "career", "relationships", "future_destiny", "wealth"]
# ponytail: vocab 35 traits threshold 300-5000 would be ideal. Now 7 core traits only; upgrade when probe passes AUC>0.60.
TRAITS = ["warm", "calm", "confident", "perceptive", "resilient", "reliable", "practical"]
# stem map for trait keyword (handles warm/warmth, confident/confidence ..)
TRAIT_PAT = {t: re.compile(rf"\b{re.escape(t)}", re.I) for t in TRAITS}
# ponytail: exact word too strict for warm/warmth split; use lemma. Upgrade: use Trait lexicon with aliases.
TRAIT_PAT["warm"] = re.compile(r"\bwarm", re.I)
TRAIT_PAT["confident"] = re.compile(r"confiden", re.I)
TRAIT_PAT["resilient"] = re.compile(r"resilien", re.I)
TRAIT_PAT["perceptive"] = re.compile(r"percept", re.I)
TRAIT_PAT["reliable"] = re.compile(r"reliab", re.I)


def norm_name(v: str) -> str:
    s = os.path.basename(str(v).strip())
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\)\s+\(", ")(", s)
    return s.lower()


def stem(name: str) -> str:
    return norm_name(name).rsplit(".", 1)[0]


def is_feature_col(name: str) -> bool:
    return name.endswith("_ratio") or "area_ratio" in name or "asymmetry" in name


def load_joined(feature_set="slim"):
    csv_path = CLEAN_CSV if CLEAN_CSV.exists() else FALLBACK_CSV
    df = pd.read_csv(csv_path)  # keeps NaN for empty strings
    # ponytail: full join uses norm_key; unmatched 5 rows (3111/4087/4568/4641/img331) dropped here. Upgrade: backfill those 5 labels.
    # labels map
    labels = {}
    with open(JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            k = norm_name(rec.get("image", ""))
            if k:
                labels[k] = rec
    df["norm_key"] = df["image_id"].apply(norm_name)
    df["stem"] = df["image_id"].apply(stem)
    # keep only matched
    mask = df["norm_key"].isin(labels)
    unmatched = int((~mask).sum())
    df_m = df[mask].copy().reset_index(drop=True)
    # extract personality text and trait labels
    texts = []
    for _, r in df_m.iterrows():
        rec = labels[r["norm_key"]]
        vals = rec.get("predictions", {}).get("personality") or []
        texts.append(" ".join(vals))
    df_m["personality_text"] = texts
    y = pd.DataFrame({t: df_m["personality_text"].str.contains(pat, na=False).astype(int) for t, pat in TRAIT_PAT.items()})
    # feature cols
    all_cols = [c for c in df_m.columns if c not in ("image_id", "norm_key", "stem", "personality_text")]
    # numeric only (csv is all numeric except ids)
    numeric_cols = [c for c in df.columns if c not in ("image_id", "norm_key", "stem")]
    if feature_set == "slim":
        numeric_cols = [c for c in numeric_cols if is_feature_col(c)]
    # ensure existence in df_m
    numeric_cols = [c for c in numeric_cols if c in df_m.columns]
    X = df_m[numeric_cols].apply(pd.to_numeric, errors="coerce")
    groups = df_m["stem"].values
    ids = df_m["image_id"].values
    return X, y, groups, ids, numeric_cols, unmatched, df_m["personality_text"]


def run_fold(X_train, X_val, y_train, y_val, numeric_cols, high_null_cols=None):
    # per-fold threshold: fit on train only, no global leak
    if high_null_cols is None:
        tr_rate = X_train.isna().mean()
        high_null_cols = [c for c in X_train.columns if tr_rate.get(c, 0) > 0.05]
    def add_ind(df):
        df = df.copy()
        for c in high_null_cols:
            if c in df.columns:
                df[c + "__missing"] = df[c].isna().astype(int)
        return df
    Xt = add_ind(X_train)
    Xv = add_ind(X_val)
    # order cols deterministic
    cols = list(Xt.columns)
    # median impute on train numeric+indicator? indicator has no NaN, keep
    imp = SimpleImputer(strategy="median")
    Xt_imp = imp.fit_transform(Xt[cols])
    Xv_imp = imp.transform(Xv[cols])
    scaler = StandardScaler()
    Xt_s = scaler.fit_transform(Xt_imp)
    Xv_s = scaler.transform(Xv_imp)
    return Xt_s, Xv_s, cols, imp, scaler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=["slim", "full"], default="slim")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--save", action="store_true", help="save final model per trait on full data")
    ap.add_argument("--drop-ear", action="store_true", help="ablation: drop ear cols (MNAR check)")
    ap.add_argument("--exclude", default="", help="comma-separated regex substrings to drop from features")
    ap.add_argument("--shuffle-labels", type=int, default=0, help="permute labels N times (null test); report mean macro_auc")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    ap.add_argument("--report", default=str(REPORT), help="report json path (default Source/train_report.json)")
    args = ap.parse_args()

    X, y, groups, ids, numeric_cols, unmatched, texts = load_joined(args.features)
    if args.drop_ear:
        keep = [c for c in numeric_cols if '_ear_' not in c]
        print(f'ablation drop-ear: {len(numeric_cols)} -> {len(keep)}')
        numeric_cols = keep
        X = X[numeric_cols]
    if args.exclude:
        pats = [re.compile(p.strip()) for p in args.exclude.split(",") if p.strip()]
        keep = [c for c in numeric_cols if not any(p.search(c) for p in pats)]
        print(f"exclude {[p.pattern for p in pats]}: {len(numeric_cols)} -> {len(keep)}")
        numeric_cols = keep
        X = X[numeric_cols]
    null_rate = X.isna().mean()
    high_null_cols = [c for c in numeric_cols if null_rate.get(c, 0) > 0.05]
    print(f"rows {len(X)} unmatched_dropped {unmatched} features {len(numeric_cols)} high_null {len(high_null_cols)}")
    print(f"traits prevalence: {y.mean().to_dict()}")
    print(f"groups {len(set(groups))} dup_stem {len(X) - len(set(groups))} (stem leak risk)")

    if StratifiedGroupKFold is not None:
        strat = y.sum(axis=1).astype(str)  # n_labels 0..6, min n=9, avoids 118-combo singletons
        # ponytail: sum proxy, not full iterative stratification (Sechidis 2011); upgrade when n_trait>10
        gkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        split_iter = gkf.split(X, strat, groups)
        split_name = f"StratifiedGroupKFold(shuffle={args.seed}) by n_labels"
    else:
        gkf = GroupKFold(n_splits=args.folds)
        split_iter = gkf.split(X, y, groups)
        split_name = "GroupKFold"
    print(f"split: {split_name}")
    # materialize fold indices so null test can reuse identical splits
    fold_idx = [(tr, va) for tr, va in split_iter]
    per_trait_aucs = {t: [] for t in TRAITS}
    per_trait_pr = {t: [] for t in TRAITS}
    per_trait_f1 = {t: [] for t in TRAITS}

    for fold, (tr, va) in enumerate(fold_idx):
        Xt_s, Xv_s, cols, _, _ = run_fold(X.iloc[tr], X.iloc[va], y.iloc[tr], y.iloc[va], numeric_cols, high_null_cols)
        y_tr, y_va = y.iloc[tr].values, y.iloc[va].values
        for ti, t in enumerate(TRAITS):
            yt_tr = y_tr[:, ti]
            yt_va = y_va[:, ti]
            if len(set(yt_tr)) < 2 or len(set(yt_va)) < 2:
                continue
            # ponytail: single-model-for-all would be better; now one logistic per trait (shortest diff).
            clf = LogisticRegression(max_iter=500, solver="lbfgs", class_weight="balanced")
            clf.fit(Xt_s, yt_tr)
            prob = clf.predict_proba(Xv_s)[:, 1]
            pred = (prob >= 0.5).astype(int)
            try:
                auc = roc_auc_score(yt_va, prob)
            except ValueError:
                auc = float("nan")
            try:
                pr = average_precision_score(yt_va, prob)
            except ValueError:
                pr = float("nan")
            f1 = f1_score(yt_va, pred, zero_division=0)
            per_trait_aucs[t].append(auc)
            per_trait_pr[t].append(pr)
            per_trait_f1[t].append(f1)
        print(f"fold {fold+1}/{args.folds} done")

    # null test: permute labels N times, reuse same folds
    null_macros = []
    if args.shuffle_labels > 0:
        rng = np.random.default_rng(args.seed)
        for si in range(args.shuffle_labels):
            y_shuf = y.copy()
            for c in y.columns:
                y_shuf[c] = rng.permutation(y_shuf[c].values)
            shuf_aucs = {t: [] for t in TRAITS}
            for tr, va in fold_idx:
                Xt_s, Xv_s, cols, _, _ = run_fold(X.iloc[tr], X.iloc[va], y_shuf.iloc[tr], y_shuf.iloc[va], numeric_cols, high_null_cols)
                for ti, t in enumerate(TRAITS):
                    yt_tr = y_shuf.iloc[tr].values[:, ti]
                    yt_va = y_shuf.iloc[va].values[:, ti]
                    if len(set(yt_tr)) < 2 or len(set(yt_va)) < 2:
                        continue
                    clf = LogisticRegression(max_iter=500, solver="lbfgs", class_weight="balanced")
                    clf.fit(Xt_s, yt_tr)
                    prob = clf.predict_proba(Xv_s)[:, 1]
                    try:
                        shuf_aucs[t].append(roc_auc_score(yt_va, prob))
                    except ValueError:
                        pass
            m = float(np.mean([np.mean(v) for v in shuf_aucs.values() if v]))
            null_macros.append(m)
            print(f"shuffle {si+1}/{args.shuffle_labels} macro {m:.3f}")
        print(f"null_test: mean_macro={float(np.mean(null_macros)):.4f} +/- {float(np.std(null_macros)):.4f}  (n={len(null_macros)})")

    summary = {}
    for t in TRAITS:
        aucs = [x for x in per_trait_aucs[t] if not np.isnan(x)]
        prs = [x for x in per_trait_pr[t] if not np.isnan(x)]
        f1s = per_trait_f1[t]
        summary[t] = {
            "auc_mean": float(np.mean(aucs)) if aucs else None,
            "auc_std": float(np.std(aucs)) if aucs else None,
            "pr_mean": float(np.mean(prs)) if prs else None,
            "f1_mean": float(np.mean(f1s)) if f1s else None,
            "prevalence": float(y[t].mean()),
            "folds": len(aucs),
        }
    macro_auc = float(np.mean([v["auc_mean"] for v in summary.values() if v["auc_mean"] is not None]))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"macro_auc {macro_auc:.3f}")
    assert macro_auc > 0.5, f"probe failed: macro_auc {macro_auc:.3f} <=0.5"  # ponytail: single threshold; upgrade to per-trait CI when n_trait>10

    # ponytail: no imputation artefact analysis now; upgrade: ablation drop ear vs keep+indicator.
    report = {
        "csv": str(CLEAN_CSV if CLEAN_CSV.exists() else FALLBACK_CSV),
        "rows": int(len(X)),
        "unmatched_dropped": int(unmatched),
        "features": args.features,
        "n_features": int(len(numeric_cols)),
        "high_null_cols": high_null_cols,
        "null_top": {c: float(null_rate[c]) for c in high_null_cols[:10]},
        "traits": TRAITS,
        "folds": args.folds,
        "per_trait": summary,
        "macro_auc": macro_auc,
        "group_leakage_guard": split_name,
        "drop_ear": bool(args.drop_ear),
        "stratification": split_name,
        "shuffle_labels": args.shuffle_labels,
        "null_macro_mean": float(np.mean(null_macros)) if null_macros else None,
        "null_macro_std": float(np.std(null_macros)) if null_macros else None,
        "imputation": "median fit on train only + missing indicator for >5% null",
    }
    rep_path = Path(args.report)
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {rep_path}")

    if args.save:
        out_dir = Path(__file__).parent / "models"
        out_dir.mkdir(exist_ok=True)
        # fit final model on full data
        Xt_full = X.copy()
        for c in high_null_cols:
            if c in Xt_full.columns:
                Xt_full[c + "__missing"] = Xt_full[c].isna().astype(int)
        cols = list(Xt_full.columns)
        imp = SimpleImputer(strategy="median")
        Xt_imp = imp.fit_transform(Xt_full[cols])
        scaler = StandardScaler()
        Xt_s = scaler.fit_transform(Xt_imp)
        np.save(out_dir / "feature_cols.npy", np.array(cols))
        # save imputer/scaler via joblib or pickle
        import pickle
        pickle.dump(imp, open(out_dir / "imputer.pkl", "wb"))
        pickle.dump(scaler, open(out_dir / "scaler.pkl", "wb"))
        for ti, t in enumerate(TRAITS):
            clf = LogisticRegression(max_iter=500, solver="lbfgs", class_weight="balanced")
            clf.fit(Xt_s, y[t].values)
            pickle.dump(clf, open(out_dir / f"logreg_{t}.pkl", "wb"))
        print(f"saved models to {out_dir}")


if __name__ == "__main__":
    main()
