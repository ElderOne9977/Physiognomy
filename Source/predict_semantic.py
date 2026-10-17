"""
Predict semantic trait probabilities for new faces.

Loads Source/models_semantic/ (feature_cols, imputer, scaler, logreg_*.pkl)
and emits JSONL with per-image trait probs.

Usage:
  python Source/predict_semantic.py --csv faces_full_features_clean.csv --output semantic_predictions.jsonl
  python Source/predict_semantic.py --csv new_features.csv --output out.jsonl
"""
import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(__file__).parent / "models_semantic"


def load_models():
    feat_cols = np.load(MODEL_DIR / "feature_cols.npy", allow_pickle=True).tolist()
    trait_names = np.load(MODEL_DIR / "trait_names.npy", allow_pickle=True).tolist()
    with open(MODEL_DIR / "imputer.pkl", "rb") as f:
        imp = pickle.load(f)
    with open(MODEL_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    # missing indicator cols (for reference only; feat_cols already includes __missing)
    clfs = {}
    for t in trait_names:
        p = MODEL_DIR / f"logreg_{t}.pkl"
        with open(p, "rb") as f:
            clfs[t] = pickle.load(f)
    return feat_cols, trait_names, imp, scaler, clfs


def prepare_features(df_raw, feat_cols):
    # df_raw has slim feature columns; missing-indicator cols are in feat_cols
    # base feature names without suffix
    base_cols = [c for c in feat_cols if not c.endswith("__missing")]
    miss_cols = [c for c in feat_cols if c.endswith("__missing")]
    # ensure all base cols present (fill NaN if missing)
    for c in base_cols:
        if c not in df_raw.columns:
            df_raw[c] = np.nan
    # build full matrix in feat_cols order
    out = pd.DataFrame(index=df_raw.index)
    for c in base_cols:
        out[c] = pd.to_numeric(df_raw[c], errors="coerce")
    for c in miss_cols:
        base = c.replace("__missing", "")
        if base in df_raw.columns:
            out[c] = df_raw[base].isna().astype(int)
        else:
            out[c] = 0
    # reorder to feat_cols
    out = out[feat_cols]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="input CSV with facial features (must contain image_id)")
    ap.add_argument("--output", required=True, help="output JSONL path")
    args = ap.parse_args()

    feat_cols, trait_names, imp, scaler, clfs = load_models()
    print(f"Loaded {len(feat_cols)} feature cols, {len(trait_names)} traits from {MODEL_DIR}")

    df = pd.read_csv(args.csv, encoding="utf-8-sig")
    if "image_id" not in df.columns:
        raise ValueError("input CSV must contain image_id column")
    ids = df["image_id"].astype(str).tolist()

    X_raw = prepare_features(df, feat_cols)
    X_imp = imp.transform(X_raw[feat_cols])
    X_s = scaler.transform(X_imp)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        for i, img_id in enumerate(ids):
            row = X_s[i:i+1]
            probs = {}
            for t in trait_names:
                clf = clfs[t]
                p = float(clf.predict_proba(row)[0, 1])
                probs[t] = round(p, 4)
            rec = {"image": img_id, "probs": probs}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(ids)} predictions to {out_path}")
    # ponytail: single-row inference path auto-tested; upgrade: batch streaming for >100k rows


if __name__ == "__main__":
    main()