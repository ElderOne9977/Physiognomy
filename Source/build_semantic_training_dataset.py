"""
Build semantic_training_dataset.csv + semantic_join_report.json.

Joins faces_full_features_clean.csv (slim 95) with semantic_labels.csv (45 traits)
on normalized basename. Reuses norm_name logic from build_training_dataset.py.
Outputs:
  - semantic_training_dataset.csv : image, group_id, 95 features, 45 traits
  - semantic_join_report.json      : counts, unmatched, dups, coverage

Usage:
  python Source/build_semantic_training_dataset.py
"""
import json
import os
import re
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "faces_full_features_clean.csv"
LABELS = ROOT / "semantic_labels.csv"
SCHEMA = ROOT / "semantic_schema.json"
OUT_CSV = ROOT / "semantic_training_dataset.csv"
OUT_REPORT = Path(__file__).parent / "semantic_join_report.json"


def norm_name(v):
    s = os.path.basename(str(v).strip())
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\)\s+\(", ")(", s)
    return s.lower()


def stem(v):
    return norm_name(v).rsplit(".", 1)[0]


def is_feature_col(n):
    return n.endswith("_ratio") or "area_ratio" in n or "asymmetry" in n


def main():
    df = pd.read_csv(CSV, encoding="utf-8-sig")
    n_csv = len(df)
    feature_cols = [c for c in df.columns if c == "image_id" or is_feature_col(c)]
    print(f"CSV rows {n_csv}, slim cols {len(feature_cols)}")

    lab = pd.read_csv(LABELS, encoding="utf-8-sig")
    # handle BOM column name
    c0 = lab.columns[0]
    if c0.strip().lower().lstrip("\ufeff") == "image" and c0 != "image":
        lab = lab.rename(columns={c0: "image"})
    n_lab = len(lab)
    trait_cols = [c for c in lab.columns if c != "image"]
    print(f"Labels rows {n_lab}, traits {len(trait_cols)}")

    df["norm_key"] = df["image_id"].apply(norm_name)
    lab["norm_key"] = lab["image"].apply(norm_name)
    df["group_id"] = df["image_id"].apply(stem)
    lab["group_id_lab"] = lab["image"].apply(stem)

    # dup checks
    dup_ids = df[df.duplicated("norm_key", keep=False)].sort_values("norm_key")["image_id"].tolist()
    dup_stems = df[df.duplicated("group_id", keep=False)].sort_values("group_id")["image_id"].tolist()
    # unmatched
    csv_keys = set(df["norm_key"])
    lab_keys = set(lab["norm_key"])
    matched_keys = csv_keys & lab_keys
    unmatched_csv = sorted(csv_keys - lab_keys)
    unmatched_lab = sorted(lab_keys - csv_keys)
    # map for join
    merged = pd.merge(df, lab, on="norm_key", how="inner", suffixes=("", "_lab"))
    # drop helper cols duplicates if any
    # keep image_id from df, group_id, feature cols, traits
    keep_cols = ["image_id", "group_id"] + [c for c in feature_cols if c != "image_id"] + trait_cols
    out = merged[keep_cols].copy()
    # ensure trait order matches schema
    try:
        import json as _j
        schema = _j.loads(SCHEMA.read_text(encoding="utf-8"))
        ordered = [t["name"] for t in schema["traits"]]
        # reorder trait columns to schema order (features keep original order)
        trait_ordered = [c for c in ordered if c in trait_cols]
        keep_ordered = ["image_id", "group_id"] + [c for c in feature_cols if c != "image_id"] + trait_ordered
        out = out[keep_ordered]
        trait_cols = trait_ordered
    except Exception:
        pass

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Wrote {OUT_CSV} shape {out.shape}")

    report = {
        "csv_path": str(CSV),
        "labels_path": str(LABELS),
        "csv_rows": int(n_csv),
        "label_rows": int(n_lab),
        "feature_rows": int(n_csv),
        "matched": int(len(out)),
        "unmatched_csv": unmatched_csv,
        "unmatched_lab_count": len(unmatched_lab),
        "unmatched_csv_count": len(unmatched_csv),
        "unmatched_csv_sample": unmatched_csv[:10],
        "unmatched_lab_sample": unmatched_lab[:10],
        "dup_image_ids": dup_ids[:20],
        "dup_stems_count": len(set(df["group_id"])) ,
        "unique_groups_csv": int(df["group_id"].nunique()),
        "unique_groups_matched": int(out["group_id"].nunique()),
        "feature_count": int(len([c for c in feature_cols if c != "image_id"])),
        "trait_count": int(len(trait_cols)),
        "trait_names": trait_cols,
        "columns": list(out.columns)[:10],
        "out_csv": str(OUT_CSV),
    }
    OUT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()