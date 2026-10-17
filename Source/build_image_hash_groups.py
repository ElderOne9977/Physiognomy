# -*- coding: utf-8 -*-
# Stage A+B: image hash manifest + leakage-safe connected components
# Usage: py Source/build_image_hash_groups.py
import hashlib
import json
import os
import re
import time
import math
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
CSV_CLEAN = ROOT / "faces_full_features_clean.csv"
LABELS_CSV = ROOT / "semantic_labels.csv"
HUMANS = ROOT / "Humans"
OUT_DIR = SRC / "experiments" / "hash_grouped_semantic"
CHUNK = 1024 * 1024
def norm_name(v):
    s = os.path.basename(str(v).strip())
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\)\s+\(", ")(", s)
    return s.lower()
def stem_name(v):
    return re.sub(r"\.[^.]+$", "", norm_name(v))
def fin(x):
    try:
        v = float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None
def hash_file(path, algo="sha256"):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_CLEAN, encoding="utf-8-sig") if CSV_CLEAN.exists() else pd.DataFrame()
    labels = pd.read_csv(LABELS_CSV, encoding="utf-8-sig") if LABELS_CSV.exists() else pd.DataFrame()
    if not labels.empty:
        img_col = [c for c in labels.columns if c.strip().lower().lstrip("\ufeff") == "image"][0]
        if img_col != "image":
            labels = labels.rename(columns={img_col: "image"})
        labels["norm_key"] = labels["image"].apply(norm_name)
        labels["norm_stem"] = labels["image"].apply(stem_name)
    if not df.empty:
        df["norm_key"] = df["image_id"].apply(norm_name)
        df["norm_stem"] = df["image_id"].apply(stem_name)
    by_lower = {}
    all_files = []
    if HUMANS.exists():
        for p in HUMANS.iterdir():
            if p.is_file():
                all_files.append(p)
                by_lower[p.name.lower()] = p
    feature_norm_set = set(df["norm_key"].tolist()) if not df.empty else set()
    label_norm_set = set(labels["norm_key"].tolist()) if not labels.empty else set()
    rows = []
    md5_to_files = defaultdict(list)
    sha_to_files = defaultdict(list)
    unreadable = []
    for p in sorted(all_files, key=lambda x: x.name.lower()):
        name = p.name
        norm = norm_name(name)
        stem = stem_name(name)
        try:
            sz = p.stat().st_size
        except Exception:
            sz = None
        try:
            md5 = hash_file(p, "md5")
            sha256 = hash_file(p, "sha256")
        except Exception as e:
            unreadable.append({"file": name, "error": str(e)[:300]})
            md5 = None
            sha256 = None
        rows.append({
            "image_id": name,
            "original_filename": name,
            "resolved_image_path": str(p.resolve()),
            "normalized_filename": norm,
            "normalized_stem": stem,
            "md5": md5,
            "sha256": sha256,
            "byte_size": sz,
            "has_feature_row": norm in feature_norm_set,
            "has_label": norm in label_norm_set,
            "feature_rows_matching": int((df["norm_key"] == norm).sum()) if not df.empty else 0,
            "label_rows_matching": int((labels["norm_key"] == norm).sum()) if not labels.empty else 0,
        })
        if md5:
            md5_to_files[md5].append(name)
        if sha256:
            sha_to_files[sha256].append(name)
    md5_groups = {k: v for k, v in md5_to_files.items() if len(v) > 1}
    sha_groups = {k: v for k, v in sha_to_files.items() if len(v) > 1}
    file_to_md5 = {}
    for md5, files in md5_to_files.items():
        for f in files:
            file_to_md5[f] = md5
    file_to_sha = {}
    for sha, files in sha_to_files.items():
        for f in files:
            file_to_sha[f] = sha
    disagreements = []
    for md5, files in md5_groups.items():
        shas = set(file_to_sha.get(f) for f in files)
        if len(shas) > 1:
            disagreements.append({"md5": md5, "files": files[:5], "distinct_sha256": len(shas)})
    for sha, files in sha_groups.items():
        md5s = set(file_to_md5.get(f) for f in files)
        if len(md5s) > 1:
            disagreements.append({"sha256": sha, "files": files[:5], "distinct_md5": len(md5s)})
    feature_hashes = {}
    for iid in df["image_id"].astype(str).tolist() if not df.empty else []:
        p = by_lower.get(iid.lower())
        if p is None:
            cands = [x for x in all_files if stem_name(x.name) == stem_name(iid)]
            p = cands[0] if cands else None
        if p is not None:
            try:
                feature_hashes[iid] = hash_file(p, "md5")
            except Exception:
                pass
    # Build leakage-safe groups via union-find on feature rows only
    # Nodes = feature row indices (0..n-1) where file exists; use SHA256 + norm_stem equivalence
    n_feat = len(df)
    sha_of_row = {}
    stem_of_row = {}
    for idx, iid in enumerate(df["image_id"].astype(str).tolist() if not df.empty else []):
        p = by_lower.get(iid.lower())
        if p is None:
            cands = [x for x in all_files if stem_name(x.name) == stem_name(iid)]
            p = cands[0] if cands else None
        if p is not None:
            try:
                sha_of_row[idx] = hash_file(p, "sha256")
            except Exception:
                sha_of_row[idx] = None
        else:
            sha_of_row[idx] = None
        stem_of_row[idx] = stem_name(iid)
    parent = list(range(n_feat))
    rank = [0]*n_feat
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]
            x=parent[x]
        return x
    def union(a,b):
        ra, rb = find(a), find(b)
        if ra==rb:
            return
        if rank[ra]<rank[rb]:
            parent[ra]=rb
        elif rank[ra]>rank[rb]:
            parent[rb]=ra
        else:
            parent[rb]=ra
            rank[ra]+=1
    # union by SHA
    sha_to_indices = defaultdict(list)
    for idx, sha in sha_of_row.items():
        if sha is not None:
            sha_to_indices[sha].append(idx)
    for idxs in sha_to_indices.values():
        for a in idxs[1:]:
            union(idxs[0], a)
    # union by stem
    stem_to_indices = defaultdict(list)
    for idx, stem in stem_of_row.items():
        stem_to_indices[stem].append(idx)
    for idxs in stem_to_indices.values():
        for a in idxs[1:]:
            union(idxs[0], a)
    comp_map = {}
    comp_id_of_row = []
    for i in range(n_feat):
        r = find(i)
        if r not in comp_map:
            comp_map[r] = len(comp_map)
        comp_id_of_row.append(comp_map[r])
    n_components = len(comp_map)
    comp_sizes = Counter(comp_id_of_row)
    largest = max(comp_sizes.values()) if comp_sizes else 0
    # stats for stage A audit
    n_sha_groups_feat = len([v for v in sha_to_indices.values() if len(v)>1])
    n_stem_groups_feat = len([v for v in stem_to_indices.values() if len(v)>1])
    # save safe groups csv (feature rows only)
    safe_groups_path = OUT_DIR / "semantic_leakage_safe_groups.csv"
    df_out = df[["image_id"]].copy() if not df.empty else pd.DataFrame()
    if not df.empty:
        df_out["norm_key"] = df["norm_key"]
        df_out["norm_stem"] = df["norm_stem"]
        df_out["sha256"] = [sha_of_row.get(i) for i in range(n_feat)]
        df_out["md5"] = [hash_file(by_lower.get(df["image_id"].astype(str).tolist()[i].lower()), "md5") if by_lower.get(df["image_id"].astype(str).tolist()[i].lower()) is not None else None for i in range(n_feat)] if False else None
        # compute md5 per row for export
        md5s = []
        for idx, iid in enumerate(df["image_id"].astype(str).tolist()):
            p = by_lower.get(iid.lower())
            if p is None:
                cands = [x for x in all_files if stem_name(x.name) == stem_name(iid)]
                p = cands[0] if cands else None
            if p is not None:
                try:
                    md5s.append(hash_file(p, "md5"))
                except Exception:
                    md5s.append(None)
            else:
                md5s.append(None)
        df_out["md5"] = md5s
        df_out["leakage_safe_group_id"] = comp_id_of_row
        df_out["has_feature_row"] = True
        df_out.to_csv(safe_groups_path, index=False, encoding="utf-8")
    print(f"Wrote {safe_groups_path} components={n_components} largest={largest}")
    # manifest CSV
    manifest_path = OUT_DIR / "image_hash_manifest.csv"
    import csv
    fieldnames = ["image_id","original_filename","resolved_image_path","normalized_filename","normalized_stem","md5","sha256","byte_size","has_feature_row","has_label","feature_rows_matching","label_rows_matching"]
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {manifest_path} ({len(rows)} rows)")
    audit = {
        "stage": "A-B image hash manifest + leakage-safe groups",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "inputs": {
            "humans_dir": str(HUMANS),
            "humans_file_count": len(all_files),
            "feature_rows": int(len(df)),
            "label_rows": int(len(labels)),
        },
        "hashing": {
            "chunk_bytes": CHUNK,
            "images_hashed": len(rows),
            "unreadable_images": len(unreadable),
            "unique_md5": len(md5_to_files),
            "unique_sha256": len(sha_to_files),
            "duplicate_md5_groups": len(md5_groups),
            "duplicate_sha256_groups": len(sha_groups),
            "rows_in_duplicate_md5_groups": sum(len(v) for v in md5_groups.values()),
            "rows_in_duplicate_sha256_groups": sum(len(v) for v in sha_groups.values()),
            "md5_sha256_disagreements": len(disagreements),
            "disagreement_samples": disagreements[:3],
        },
        "feature_subset_hashing": {
            "feature_ids_hashed": len(feature_hashes),
            "feature_unique_md5": len(set(feature_hashes.values())) if feature_hashes else 0,
            "feature_duplicate_md5_groups": len([h for h,c in Counter(feature_hashes.values()).items() if c>1]) if feature_hashes else 0,
        },
        "leakage_safe_groups": {
            "n_feature_rows": n_feat,
            "n_sha256_groups_feature": n_sha_groups_feat,
            "n_stem_groups_feature": n_stem_groups_feat,
            "n_connected_components": n_components,
            "largest_component_size": int(largest),
            "component_size_distribution": dict(sorted(Counter(comp_sizes.values()).items())),
            "duplicate_row_coverage": int(sum(1 for c,s in comp_sizes.items() if s>1 and c in comp_sizes) if False else sum(v for v in comp_sizes.values() if v>1)),
            "num_components_size_gt1": int(sum(1 for v in comp_sizes.values() if v>1)),
            "rows_in_components_gt1": int(sum(v for v in comp_sizes.values() if v>1)),
        },
        "elapsed_sec": fin(time.time() - t0),
    }
    # correct duplicate coverage = rows in components >1 = sum sizes >1
    audit["leakage_safe_groups"]["duplicate_row_coverage"] = audit["leakage_safe_groups"]["rows_in_components_gt1"]
    audit_path = OUT_DIR / "image_hash_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2))
if __name__ == "__main__":
    main()