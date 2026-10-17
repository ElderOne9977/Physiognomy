# -*- coding: utf-8 -*-
# Stages C-K: duplicate label conflicts, safe CV, OOF training, strategy comparison
# Usage: py Source/evaluate_hash_grouped_semantic.py
import json
import math
import time
import re
import os
import hashlib
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
CSV_CLEAN = ROOT / "faces_full_features_clean.csv"
LABELS_CSV = ROOT / "semantic_labels.csv"
SCHEMA_JSON = ROOT / "semantic_schema.json"
HUMANS = ROOT / "Humans"
EXP_DIR = SRC / "experiments" / "hash_grouped_semantic"
SAFE_GROUPS_CSV = EXP_DIR / "semantic_leakage_safe_groups.csv"
CHUNK = 1024*1024
SEED = 42
FOLDS = 5
def norm_name(v):
    s = os.path.basename(str(v).strip()); s = re.sub(r"\s+", " ", s); s = re.sub(r"\)\s+\(", ")(", s); return s.lower()
def stem_name(v):
    return re.sub(r"\.[^.]+$", "", norm_name(v))
def fin(x):
    try:
        v = float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None
def ece_score(y,p,n_bins=10, sample_weight=None):
    y=np.asarray(y,float); p=np.asarray(p,float)
    if len(y)==0: return None
    bins=np.linspace(0,1,n_bins+1); e=0.0
    w=np.asarray(sample_weight,float) if sample_weight is not None else None
    tot = w.sum() if w is not None else len(y)
    for i in range(n_bins):
        lo,hi=bins[i],bins[i+1]
        m=(p>=lo)&(p<hi) if i<n_bins-1 else (p>=lo)&(p<=hi)
        if m.sum()==0: continue
        if w is not None:
            wm=w[m].sum()
            if wm==0: continue
            e+= abs((y[m]*w[m]).sum()/wm - (p[m]*w[m]).sum()/wm) * wm/tot
        else:
            e+= abs(y[m].mean()-p[m].mean())*m.sum()/len(y)
    return float(e)
def load_schema():
    sch=json.loads(SCHEMA_JSON.read_text(encoding="utf-8")); names=[t["name"] for t in sch["traits"]]; cat={t["name"]:t["category"] for t in sch["traits"]}; return names, cat, sch
def is_feature_col(name):
    return name.endswith("_ratio") or "area_ratio" in name or "asymmetry" in name
def hash_file(p, algo="sha256"):
    h=hashlib.new(algo)
    with open(p,"rb") as f:
        while True:
            b=f.read(CHUNK)
            if not b: break
            h.update(b)
    return h.hexdigest()
def ensure_safe_groups():
    if SAFE_GROUPS_CSV.exists():
        return pd.read_csv(SAFE_GROUPS_CSV, encoding="utf-8")
    raise FileNotFoundError(f"missing {SAFE_GROUPS_CSV} run build_image_hash_groups.py first")
def main():
    t0=time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    traits, cat_map, schema = load_schema()
    safe_df = ensure_safe_groups()
    # Load aligned dataset (feature + labels + safe group)
    df = pd.read_csv(CSV_CLEAN, encoding="utf-8-sig")
    labels = pd.read_csv(LABELS_CSV, encoding="utf-8-sig")
    img_col=[c for c in labels.columns if c.strip().lower().lstrip("\ufeff")=="image"][0]
    if img_col!="image": labels=labels.rename(columns={img_col:"image"})
    df["norm_key"]=df["image_id"].apply(norm_name)
    df["norm_stem"]=df["image_id"].apply(stem_name)
    labels["norm_key"]=labels["image"].apply(norm_name)
    labels["norm_stem"]=labels["image"].apply(stem_name)
    # merge labels onto df (inner)
    merged=pd.merge(df, labels, on="norm_key", how="inner", suffixes=("","_lab"))
    # attach safe group
    # safe_df has image_id -> group
    grp_map=dict(zip(safe_df["image_id"].astype(str), safe_df["leakage_safe_group_id"]))
    # also map via norm_key fallback if image_id mismatched case
    norm_grp=dict(zip(safe_df["norm_key"] if "norm_key" in safe_df.columns else safe_df["image_id"].apply(norm_name), safe_df["leakage_safe_group_id"]))
    merged["safe_group"]=merged["image_id"].map(grp_map)
    # fallback via norm_key
    mask=merged["safe_group"].isna()
    if mask.any():
        merged.loc[mask,"safe_group"]=merged.loc[mask,"norm_key"].map(norm_grp)
    # any still missing => assign new group id (should not happen for matched rows)
    if merged["safe_group"].isna().any():
        missing=merged["safe_group"].isna().sum()
        print(f"WARNING: {missing} merged rows missing safe_group, assigning new ids")
        max_g=int(safe_df["leakage_safe_group_id"].max()) if len(safe_df) else 0
        merged.loc[merged["safe_group"].isna(),"safe_group"]=range(max_g+1, max_g+1+missing)
    merged["safe_group"]=merged["safe_group"].astype(int)
    # also attach sha for weighting
    sha_map=dict(zip(safe_df["image_id"].astype(str), safe_df["sha256"])) if "sha256" in safe_df.columns else {}
    norm_sha=dict(zip(safe_df["norm_key"] if "norm_key" in safe_df.columns else [], safe_df["sha256"])) if "sha256" in safe_df.columns else {}
    merged["sha256"]=merged["image_id"].map(sha_map)
    m2=merged["sha256"].isna()
    if m2.any() and norm_sha:
        merged.loc[m2,"sha256"]=merged.loc[m2,"norm_key"].map(norm_sha)
    # feature cols
    saved_cols=None
    try:
        saved_cols=np.load(SRC/"models_semantic"/"feature_cols.npy", allow_pickle=True).tolist()
    except Exception:
        saved_cols=[c for c in df.columns if c!="image_id" and c!="norm_key" and c!="norm_stem" and is_feature_col(c)][:95]
        # add missing indicators if exist
        try:
            rates=df[saved_cols].isna().mean()
            ind=[c for c in saved_cols if rates.get(c,0)>0.05]
            saved_cols = saved_cols + [c+"__missing" for c in ind]
        except Exception:
            pass
    base_cols=[c for c in saved_cols if not c.endswith("__missing")]
    ind_bases=[c.replace("__missing","") for c in saved_cols if c.endswith("__missing")]
    # Build X matrix (pre-indicator, impute later per fold)
    # We'll build full X with missing indicators already added (like production)
    X_full=pd.DataFrame(index=merged.index)
    for c in base_cols:
        if c in merged.columns:
            X_full[c]=pd.to_numeric(merged[c], errors="coerce")
        else:
            X_full[c]=np.nan
    for c in ind_bases:
        X_full[c+"__missing"]=merged[c].isna().astype(int) if c in merged.columns else 0
    X_full=X_full[saved_cols]
    Y=np.column_stack([pd.to_numeric(merged[t], errors="coerce").values for t in traits])
    groups=merged["safe_group"].values
    # ---- Stage C: duplicate label conflicts ----
    # Group by sha256 among merged rows
    sha_groups=defaultdict(list)
    for idx, sha in enumerate(merged["sha256"].astype(str).tolist()):
        if sha and sha!="nan" and sha!="None":
            sha_groups[sha].append(idx)
    dup_sha_groups={k:v for k,v in sha_groups.items() if len(v)>1}
    stem_groups=defaultdict(list)
    for idx, stem in enumerate(merged["norm_stem"].tolist()):
        stem_groups[stem].append(idx)
    dup_stem_groups={k:v for k,v in stem_groups.items() if len(v)>1}
    # Different bytes linked by stem
    stem_linked_diff_bytes=0
    stem_linked_details=[]
    for stem, idxs in dup_stem_groups.items():
        shas=set(merged.iloc[idxs]["sha256"].astype(str).tolist())
        # filter nan
        shas={s for s in shas if s not in ("nan","None","")}
        if len(shas)>1:
            stem_linked_diff_bytes+=1
            stem_linked_details.append({"stem":stem,"n_rows":len(idxs),"distinct_sha":len(shas)})
    # Label conflicts per trait
    conflict_per_trait=Counter()
    conflict_group_count=0
    exact_dup_identical=0
    exact_dup_conflicting=0
    per_category_disagreement={}
    cat_conflict_counts=Counter()
    cat_total_groups=Counter()
    trait_disagreement_rates={}
    total_dup_groups=len(dup_sha_groups)
    for sha, idxs in dup_sha_groups.items():
        # stack labels for these rows
        sub=Y[idxs,:]  # rows x traits
        # check if vectors identical across rows
        identical = np.all(sub == sub[0:1,:])
        if identical:
            exact_dup_identical+=1
        else:
            exact_dup_conflicting+=1
            conflict_group_count+=1
            for j,t in enumerate(traits):
                vals=set(sub[:,j].tolist())
                if len(vals)>1:
                    conflict_per_trait[t]+=1
                    cat_conflict_counts[cat_map[t]]+=1
        for j,t in enumerate(traits):
            cat_total_groups[cat_map[t]]+=1  # this overcounts, fix below
    # Correct per-category total: number of dup groups contributes to each category once per group
    cat_total={c: total_dup_groups for c in set(cat_map.values())}
    # per-category: groups where at least one trait in category disagrees
    per_category_disagreement={}
    for cat in sorted(set(cat_map.values())):
        cat_traits=[t for t in traits if cat_map[t]==cat]
        cat_traits_idx=[traits.index(t) for t in cat_traits]
        n_conf_cat=0
        for sha, idxs in dup_sha_groups.items():
            sub=Y[idxs,:]
            # check if any trait in category disagrees within group
            if any(len(set(sub[:,j].tolist()))>1 for j in cat_traits_idx):
                n_conf_cat+=1
        per_category_disagreement[cat]={"conflicting_groups": int(n_conf_cat), "total_dup_groups": int(total_dup_groups), "disagreement_rate": fin(n_conf_cat/total_dup_groups) if total_dup_groups else None}
    for t in traits:
        trait_disagreement_rates[t]=fin(conflict_per_trait.get(t,0)/total_dup_groups) if total_dup_groups else None
    # Also compute number of conflicting trait labels per group distribution
    conflicting_traits_per_group=[]
    for sha, idxs in dup_sha_groups.items():
        sub=Y[idxs,:]
        n_conf=int(sum(1 for j in range(len(traits)) if len(set(sub[:,j].tolist()))>1))
        conflicting_traits_per_group.append(n_conf)
    hist=Counter(conflicting_traits_per_group)
    # try load provider text for raw check (labels_enriched.jsonl)
    provider_text_conflicts=None
    try:
        import json as js
        enrich_path=ROOT/"labels_enriched.jsonl"
        if enrich_path.exists():
            txt_map={}
            for line in open(enrich_path, encoding="utf-8"):
                try:
                    rec=js.loads(line)
                    txt_map[norm_name(rec.get("image",""))] = rec.get("raw_output","")[:500]
                except Exception:
                    continue
            text_conflicts=0
            for sha, idxs in dup_sha_groups.items():
                texts=set(txt_map.get(merged.iloc[i]["norm_key"],"") for i in idxs)
                if len(texts)>1:
                    text_conflicts+=1
            provider_text_conflicts={"duplicate_sha_groups": total_dup_groups, "groups_with_differing_raw_text": text_conflicts, "rate": fin(text_conflicts/total_dup_groups) if total_dup_groups else None}
    except Exception as e:
        provider_text_conflicts={"error": str(e)[:200]}
    conflict_payload={
        "stage": "C duplicate label conflicts",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "inputs": {"merged_rows": int(len(merged)), "unique_sha": int(len(sha_groups)), "duplicate_sha_groups": int(total_dup_groups), "duplicate_stem_groups": int(len(dup_stem_groups)), "stem_linked_diff_bytes_groups": int(stem_linked_diff_bytes)},
        "duplicate_image_groups": {"total_dup_sha_groups": int(total_dup_groups), "exact_duplicate_identical_labels": int(exact_dup_identical), "exact_duplicate_conflicting_labels": int(exact_dup_conflicting), "identical_rate": fin(exact_dup_identical/total_dup_groups) if total_dup_groups else None, "conflicting_rate": fin(exact_dup_conflicting/total_dup_groups) if total_dup_groups else None},
        "stem_linked": {"groups_with_same_stem_but_different_bytes": int(stem_linked_diff_bytes), "sample": stem_linked_details[:5]},
        "conflict_per_trait": {t: int(conflict_per_trait.get(t,0)) for t in traits},
        "trait_disagreement_rate": trait_disagreement_rates,
        "per_category_disagreement": per_category_disagreement,
        "conflicting_traits_per_group_histogram": {str(k): int(v) for k,v in sorted(hist.items())},
        "mean_conflicting_traits_per_dup_group": fin(np.mean(conflicting_traits_per_group)) if conflicting_traits_per_group else None,
        "max_conflicting_traits_in_group": int(max(conflicting_traits_per_group)) if conflicting_traits_per_group else 0,
        "provider_text_conflicts": provider_text_conflicts,
        "ceiling_note": "Conflicting labels among byte-identical images create intrinsic ceiling <1.0; provider labels are text-derived weak targets, not ground truth.",
        "interpretation": "Do not majority-vote; report conflict rates first. Representative selection for dedup would arbitrarily discard disagreement."
    }
    (EXP_DIR/"duplicate_label_conflicts.json").write_text(json.dumps(conflict_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Conflicts:", json.dumps(conflict_payload["duplicate_image_groups"], indent=2))
    # ---- Stage D: safe CV assignments + assertions ----
    gkf=GroupKFold(n_splits=FOLDS)
    fold_assign=np.full(len(merged), -1, dtype=int)
    for fold_idx, (_, va) in enumerate(gkf.split(X_full, Y[:,0], groups=groups)):
        fold_assign[va]=fold_idx
    # Verify overlaps
    overlap_results={}
    for k in range(FOLDS):
        tr_mask=fold_assign!=k; va_mask=fold_assign==k
        tr_groups=set(groups[tr_mask]); va_groups=set(groups[va_mask])
        tr_sha=set(merged.loc[tr_mask,"sha256"].dropna().astype(str).tolist()); va_sha=set(merged.loc[va_mask,"sha256"].dropna().astype(str).tolist())
        # filter nan strings
        tr_sha={s for s in tr_sha if s not in ("nan","None","")}
        va_sha={s for s in va_sha if s not in ("nan","None","")}
        tr_stems=set(merged.loc[tr_mask,"norm_stem"].tolist()); va_stems=set(merged.loc[va_mask,"norm_stem"].tolist())
        overlap_results[f"fold_{k}"]={
            "n_train": int(tr_mask.sum()), "n_val": int(va_mask.sum()),
            "unique_train_groups": int(len(tr_groups)), "unique_val_groups": int(len(va_groups)),
            "group_overlap": int(len(tr_groups & va_groups)),
            "sha_overlap": int(len(tr_sha & va_sha)),
            "stem_overlap": int(len(tr_stems & va_stems)),
            "pass_group": len(tr_groups & va_groups)==0,
            "pass_sha": len(tr_sha & va_sha)==0,
            "pass_stem": len(tr_stems & va_stems)==0
        }
    all_pass=all(v["pass_group"] and v["pass_sha"] and v["pass_stem"] for v in overlap_results.values())
    print("Fold overlap pass:", all_pass, json.dumps(overlap_results, indent=2))
    # Prevalence diagnostics per fold
    fold_prev={}
    for k in range(FOLDS):
        va_mask=fold_assign==k
        y_va=Y[va_mask]
        pos_counts=(y_va==1).sum(axis=0)
        neg_counts=(y_va==0).sum(axis=0)
        zero_pos=[traits[i] for i,c in enumerate(pos_counts) if c==0]
        zero_neg=[traits[i] for i,c in enumerate(neg_counts) if c==0]
        fold_prev[f"fold_{k}"]={"n_val": int(va_mask.sum()), "traits_zero_pos": zero_pos, "traits_zero_neg": zero_neg, "n_zero_pos": len(zero_pos), "n_zero_neg": len(zero_neg)}
    # Save assignments
    assign_df=merged[["image_id","norm_key","norm_stem","sha256","safe_group"]].copy()
    assign_df["fold"]=fold_assign
    # add duplicate weight columns for reference
    sha_counts=Counter(merged["sha256"].astype(str).tolist())
    assign_df["sha_multiplicity"]=assign_df["sha256"].astype(str).map(lambda s: sha_counts.get(s,1))
    assign_df["inverse_hash_weight_global"]=assign_df["sha_multiplicity"].apply(lambda c: 1.0/c if c else 1.0)
    assign_df.to_csv(EXP_DIR/"cv_fold_assignments.csv", index=False, encoding="utf-8")
    # ---- Stage E/F: OOF training for 3 strategies ----
    def train_oof(strategy="A"):
        # strategy A: all rows safe grouped (already)
        # B: inverse duplicate weighting (train weights)
        # C: hash-deduplicated training (one per sha per train fold)
        traits_local=traits
        n=len(X_full); P=np.full((n, len(traits_local)), np.nan)
        P_weights=np.full((n, len(traits_local)), np.nan)  # not needed
        # global sha counts for weighting
        global_sha_counts=Counter(merged["sha256"].astype(str).tolist())
        for fold_idx in range(FOLDS):
            tr_mask=fold_assign!=fold_idx; va_mask=fold_assign==fold_idx
            X_tr_full=X_full.loc[tr_mask]; X_va_full=X_full.loc[va_mask]
            Y_tr=Y[tr_mask]; Y_va=Y[va_mask]
            # For strategy C: deduplicate training rows by sha
            train_indices=np.where(tr_mask)[0]
            if strategy=="C":
                # select one per sha deterministic: smallest image_id lexicographically within each sha group
                sha_to_rows=defaultdict(list)
                for idx in train_indices:
                    sha=str(merged.iloc[idx]["sha256"])
                    sha_to_rows[sha].append(idx)
                dedup_indices=[]
                for sha, idxs in sha_to_rows.items():
                    if len(idxs)==1:
                        dedup_indices.append(idxs[0])
                    else:
                        # conflict handling: pick smallest image_id; if labels conflict, we arbitrarily drop - documented
                        idxs_sorted=sorted(idxs, key=lambda i: str(merged.iloc[i]["image_id"]).lower())
                        dedup_indices.append(idxs_sorted[0])
                train_indices=np.array(sorted(dedup_indices))
                X_tr_full=X_full.iloc[train_indices]
                Y_tr=Y[train_indices]
            # Prepare fold-local impute+scale
            imp=SimpleImputer(strategy="median")
            X_tr_imp=imp.fit_transform(X_tr_full)
            X_va_imp=imp.transform(X_va_full)
            sc=StandardScaler()
            X_tr_s=sc.fit_transform(X_tr_imp)
            X_va_s=sc.transform(X_va_imp)
            # sample weights for strategy B
            sample_weight=None
            if strategy=="B":
                # inverse multiplicity based on global sha counts (or train fold counts)
                # use global to be deterministic
                w=np.array([1.0/global_sha_counts.get(str(merged.iloc[i]["sha256"]),1) for i in train_indices], dtype=float)
                # normalize to keep sum = n_train_dedup
                if w.sum()>0:
                    w=w * len(w)/w.sum()
                sample_weight=w
            for j,t in enumerate(traits_local):
                ytr=Y_tr[:,j] if strategy!="C" else Y[train_indices][:,j]
                # check degenerate
                uniq=np.unique(ytr[np.isfinite(ytr)])
                if len(uniq)<2:
                    continue
                clf=LogisticRegression(max_iter=500, solver="lbfgs", class_weight="balanced", random_state=SEED)
                try:
                    if sample_weight is not None:
                        clf.fit(X_tr_s, ytr, sample_weight=sample_weight)
                    else:
                        clf.fit(X_tr_s, ytr)
                    P[np.where(va_mask)[0], j]=clf.predict_proba(X_va_s)[:,1]
                except Exception as e:
                    continue
        return P
    # Run strategies sequentially (one at a time to save RAM)
    results={}
    for strat in ["A","B","C"]:
        print(f"Training strategy {strat} ...")
        P_strat=train_oof(strat)
        # Save OOF for strat A as main
        if strat=="A":
            oof_path=EXP_DIR/"oof_semantic_probs_hash_grouped.csv"
            oof_df=pd.DataFrame(P_strat, columns=traits)
            oof_df.insert(0, "image_id", merged["image_id"].values)
            # also include fold, group, sha for traceability
            oof_df["fold"]=fold_assign
            oof_df["safe_group"]=groups
            oof_df["sha256"]=merged["sha256"].values
            oof_df.to_csv(oof_path, index=False, encoding="utf-8")
            print(f"Wrote {oof_path}")
        results[strat]=P_strat
        # small pause to free
    # ---- Stage G: metrics ----
    def compute_metrics(P, Y, sample_weight=None):
        # per trait
        per_trait={}
        aucs=[]; prs=[]; briers=[]; eces=[]; f1s05=[]
        for j,t in enumerate(traits):
            y=Y[:,j]; p=P[:,j]
            mask=np.isfinite(p)&np.isfinite(y)&((y==0)|(y==1))
            if mask.sum()<20:
                per_trait[t]={"n_valid": int(mask.sum()), "roc_auc": None, "pr_auc": None, "brier": None, "ece": None, "f1_0_5": None, "prevalence": fin(y[mask].mean()) if mask.sum() else None}
                continue
            yv=y[mask]; pv=p[mask]
            w=sample_weight[mask] if sample_weight is not None else None
            # ROC/PR need both classes
            if len(np.unique(yv))<2:
                roc=None; pr=None
            else:
                try: roc=fin(roc_auc_score(yv, pv, sample_weight=w))
                except Exception: roc=None
                try: pr=fin(average_precision_score(yv, pv, sample_weight=w))
                except Exception: pr=None
            # brier weighted
            if w is not None:
                brier=fin(np.average((pv-yv)**2, weights=w))
            else:
                brier=fin(np.mean((pv-yv)**2))
            ece=fin(ece_score(yv, pv, sample_weight=w))
            # f1 @0.5
            pred=(pv>=0.5).astype(int)
            f1=fin(f1_score(yv, pred, zero_division=0, sample_weight=w) if w is not None else f1_score(yv, pred, zero_division=0))
            per_trait[t]={"n_valid": int(mask.sum()), "roc_auc": roc, "pr_auc": pr, "brier": brier, "ece": ece, "f1_0_5": f1, "prevalence": fin(yv.mean()), "category": cat_map[t]}
            if roc is not None: aucs.append(roc)
            if pr is not None: prs.append(pr)
            if brier is not None: briers.append(brier)
            if ece is not None: eces.append(ece)
            if f1 is not None: f1s05.append(f1)
        # micro: flatten?
        # compute micro F1 across all traits (requires valid rows)
        # use sample_weight expanded
        overall={
            "n_valid_traits_roc_auc": len(aucs),
            "macro_roc_auc": fin(np.mean(aucs)) if aucs else None,
            "macro_pr_auc": fin(np.mean(prs)) if prs else None,
            "macro_brier": fin(np.mean(briers)) if briers else None,
            "macro_ece": fin(np.mean(eces)) if eces else None,
            "macro_f1_0_5": fin(np.mean(f1s05)) if f1s05 else None,
        }
        return per_trait, overall
    # Row-weighted metrics for each strategy
    metrics_all={}
    for strat, P in results.items():
        per_row, overall_row=compute_metrics(P, Y, sample_weight=None)
        # duplicate-adjusted: weight per row = 1 / multiplicity of its sha (global)
        sha_counts=Counter(merged["sha256"].astype(str).tolist())
        w_dup=np.array([1.0/sha_counts.get(str(s),1) for s in merged["sha256"].astype(str).tolist()], float)
        # normalize not needed for sklearn (relative)
        per_dup, overall_dup=compute_metrics(P, Y, sample_weight=w_dup)
        metrics_all[strat]={"row_weighted": {"per_trait": per_row, "overall": overall_row}, "duplicate_adjusted": {"per_trait": per_dup, "overall": overall_dup}, "weighting": {"description": "duplicate_adjusted: each distinct sha equal total weight (1/multiplicity)"}}
    # Save per-trait CSV for strategy A row-weighted (primary)
    per_trait_csv=EXP_DIR/"semantic_hash_grouped_per_trait.csv"
    per_a_row=metrics_all["A"]["row_weighted"]["per_trait"]
    per_a_dup=metrics_all["A"]["duplicate_adjusted"]["per_trait"]
    with open(per_trait_csv,"w", encoding="utf-8", newline="") as f:
        import csv
        w=csv.writer(f)
        w.writerow(["trait","category","n_valid","roc_auc_row","pr_auc_row","brier_row","ece_row","f1_0_5_row","roc_auc_dup","pr_auc_dup","prevalence"])
        for t in traits:
            r=per_a_row.get(t,{}); d=per_a_dup.get(t,{})
            w.writerow([t, cat_map.get(t), r.get("n_valid"), r.get("roc_auc"), r.get("pr_auc"), r.get("brier"), r.get("ece"), r.get("f1_0_5"), d.get("roc_auc"), d.get("pr_auc"), r.get("prevalence")])
    # Build comparison with historical leaked metrics
    hist_path=SRC/"semantic_baseline_metrics.json"
    hist=None
    try:
        hist=json.loads(hist_path.read_text(encoding="utf-8"))
        hist_overall=hist.get("overall",{})
        hist_macro_roc=hist_overall.get("macro_roc_auc")
        hist_macro_pr=hist_overall.get("macro_pr_auc")
        hist_brier=hist_overall.get("macro_brier")
        hist_ece=hist_overall.get("macro_ece")
        hist_f1=hist_overall.get("macro_f1_threshold_0_5")
    except Exception:
        hist_macro_roc=hist_macro_pr=hist_brier=hist_ece=hist_f1=None
    # per-category aggregates
    def by_category(per_trait):
        out={}
        for c in sorted(set(cat_map.values())):
            ts=[t for t in traits if cat_map[t]==c]
            aucs=[per_trait[t]["roc_auc"] for t in ts if per_trait[t].get("roc_auc") is not None]
            out[c]={"n_traits": len(ts), "macro_roc_auc": fin(np.mean(aucs)) if aucs else None, "traits_scored": len(aucs)}
        return out
    hash_grouped_payload={
        "stage": "E/G corrected hash-grouped evaluation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "label_semantics": "Agreement with provider-derived semantic labels only, not validated personality/wealth/destiny. Leakage-safe grouping via SHA-256 + stem transitive closure.",
        "leakage_checks": {"all_folds_pass": all_pass, "per_fold": overlap_results, "fold_prevalence": fold_prev},
        "dataset": {"n_rows": int(len(merged)), "n_traits": len(traits), "n_safe_groups": int(len(set(groups))), "n_unique_sha": int(len(set(merged['sha256'].dropna().astype(str).tolist()))), "feature_cols": len(saved_cols), "preprocessing": "median impute + StandardScaler fitted per training fold only", "schema_version": schema.get("version")},
        "historical_filename_grouped_OOF_hash_leakage_present": {"macro_roc_auc": hist_macro_roc, "macro_pr_auc": hist_macro_pr, "macro_brier": hist_brier, "macro_ece": hist_ece, "macro_f1_0_5": hist_f1, "note": "Historical OOF used GroupKFold on norm_key only, ~69.1% val rows had byte-identical twin in train"},
        "corrected_hash_and_stem_group_safe_OOF": {
            "strategy_A_all_rows_safe_grouped": metrics_all["A"]["row_weighted"]["overall"],
            "strategy_A_duplicate_adjusted": metrics_all["A"]["duplicate_adjusted"]["overall"],
            "strategy_B_inverse_hash_weighting_row": metrics_all["B"]["row_weighted"]["overall"] if "B" in metrics_all else None,
            "strategy_B_inverse_hash_weighting_dup": metrics_all["B"]["duplicate_adjusted"]["overall"] if "B" in metrics_all else None,
            "strategy_C_hash_dedup_training_row": metrics_all["C"]["row_weighted"]["overall"] if "C" in metrics_all else None,
            "strategy_C_hash_dedup_training_dup": metrics_all["C"]["duplicate_adjusted"]["overall"] if "C" in metrics_all else None,
            "gap_row_weighted_macro_roc": fin(metrics_all["A"]["row_weighted"]["overall"]["macro_roc_auc"] - hist_macro_roc) if metrics_all["A"]["row_weighted"]["overall"]["macro_roc_auc"] and hist_macro_roc else None,
            "gap_note": "Observed evaluation-protocol gap; not exact causal leakage estimate due to different fold assignments/populations"
        },
        "per_category_row_weighted": by_category(metrics_all["A"]["row_weighted"]["per_trait"]),
        "per_category_duplicate_adjusted": by_category(metrics_all["A"]["duplicate_adjusted"]["per_trait"]),
        "fold_variability": {},  # filled after computing per-fold metrics if needed
        "duplicate_strategies": {
            "A_all_rows_safe": metrics_all["A"],
            "B_inverse_weight": metrics_all.get("B"),
            "C_dedup_train": metrics_all.get("C"),
            "best_supported": None
        },
        "elapsed_sec": fin(time.time()-t0)
    }
    # Determine best supported strategy by macro ROC row-weighted
    best=None; best_score=-1
    for k in ["A","B","C"]:
        sc=metrics_all.get(k,{}).get("row_weighted",{}).get("overall",{}).get("macro_roc_auc")
        if sc is not None and sc>best_score:
            best_score=sc; best=k
    hash_grouped_payload["duplicate_strategies"]["best_supported"]=f"Strategy {best} macro_roc={fin(best_score)}" if best else None
    # Also add threshold safety note
    hash_grouped_payload["threshold_and_calibration"]={"threshold_used": 0.5, "note": "Historical per-trait thresholds were tuned on leaked OOF; not used. Proper threshold tuning requires inner group-safe CV (not done). Calibration not re-tuned."}
    # Uncertainty: fold variability - compute per-fold macro ROC for strategy A
    # Need per-fold predictions breakdown
    # Recompute per-fold metrics quickly from P_A
    P_A=results["A"]
    fold_metrics=[]
    for k in range(FOLDS):
        va_mask=fold_assign==k
        per, overall=compute_metrics(P_A[va_mask], Y[va_mask], sample_weight=None)
        fold_metrics.append({"fold": k, "n_val": int(va_mask.sum()), "macro_roc": overall["macro_roc_auc"], "macro_pr": overall["macro_pr_auc"]})
    hash_grouped_payload["fold_variability"]={"per_fold": fold_metrics, "macro_roc_mean": fin(np.mean([f["macro_roc"] for f in fold_metrics if f["macro_roc"] is not None])), "macro_roc_std": fin(np.std([f["macro_roc"] for f in fold_metrics if f["macro_roc"] is not None])), "n_independent_groups": int(len(set(groups))), "traits_with_zero_pos_in_any_fold": fold_prev}
    # Identify performance collapse per trait vs historical
    try:
        hist_per=json.loads(hist_path.read_text(encoding="utf-8")).get("per_trait",{})
        deltas=[]
        for t in traits:
            h=hist_per.get(t,{}).get("roc_auc") if isinstance(hist_per.get(t), dict) else None
            c=metrics_all["A"]["row_weighted"]["per_trait"].get(t,{}).get("roc_auc")
            if h is not None and c is not None:
                deltas.append({"trait": t, "category": cat_map[t], "historical_roc": h, "corrected_roc": c, "delta": fin(c-h)})
        deltas_sorted=sorted(deltas, key=lambda x: x["delta"])
        hash_grouped_payload["trait_deltas_vs_historical"]={"largest_decreases": deltas_sorted[:10], "largest_increases": sorted(deltas, key=lambda x: -x["delta"])[:10], "n_compared": len(deltas)}
        # categorize reliable vs weak
        reliable=[d for d in deltas if d["corrected_roc"] and d["corrected_roc"]>=0.60]
        near_chance=[d for d in deltas if d["corrected_roc"] and d["corrected_roc"]<0.55]
        hash_grouped_payload["trait_analysis"]={
            "strongest_corrected": sorted(deltas, key=lambda x: -(x["corrected_roc"] or 0))[:10],
            "weakest_corrected": sorted(deltas, key=lambda x: (x["corrected_roc"] or 0))[:10],
            "reliable_ge_0_60": len(reliable),
            "near_chance_lt_0_55": len(near_chance)
        }
    except Exception as e:
        hash_grouped_payload["trait_deltas_vs_historical"]={"error": str(e)[:300]}
    # Write payloads
    (EXP_DIR/"semantic_hash_grouped_metrics.json").write_text(json.dumps(hash_grouped_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    # Comparison JSON
    comp={
        "stage": "F/G duplicate strategy comparison",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "row_weighted": {k: v["row_weighted"]["overall"] for k,v in metrics_all.items()},
        "duplicate_adjusted": {k: v["duplicate_adjusted"]["overall"] for k,v in metrics_all.items()},
        "per_trait_row": {k: v["row_weighted"]["per_trait"] for k,v in metrics_all.items()},
        "note": "Row-weighted = standard; duplicate_adjusted = each distinct sha equal weight. Strategy C deduplication within train folds only, evaluated on original validation rows. If conflicting labels high, C arbitrarily discards disagreement.",
        "label_conflict_summary": {"duplicate_sha_groups": conflict_payload["duplicate_image_groups"], "per_trait_conflicts": conflict_payload["conflict_per_trait"]}
    }
    (EXP_DIR/"semantic_duplicate_strategy_comparison.json").write_text(json.dumps(comp, indent=2, ensure_ascii=False), encoding="utf-8")
    # Also ensure finite JSON (no NaN)
    print("Corrected macro ROC row:", hash_grouped_payload["corrected_hash_and_stem_group_safe_OOF"]["strategy_A_all_rows_safe_grouped"]["macro_roc_auc"])
    print("Historical macro ROC:", hist_macro_roc)
    print("Gap:", hash_grouped_payload["corrected_hash_and_stem_group_safe_OOF"]["gap_row_weighted_macro_roc"])
    print("All folds pass:", all_pass)
    # ---- Stage I: cluster-aware bootstrap at leakage-safe group level (seed fixed) ----
    try:
        rng=np.random.default_rng(SEED)
        # build group -> row indices
        group_rows=defaultdict(list)
        for i,g in enumerate(groups):
            group_rows[g].append(i)
        group_ids=list(group_rows.keys())
        n_boot=200
        boot_aucs=[]
        P_a=results["A"]
        for _ in range(n_boot):
            samp_ids=[group_ids[j] for j in rng.integers(0,len(group_ids),size=len(group_ids))]
            idx=[]
            for g in samp_ids:
                idx.extend(group_rows[g])
            idx=np.array(idx)
            trait_aucs=[]
            for j,t in enumerate(traits):
                y=Y[idx,j]; p=P_a[idx,j]
                mask=np.isfinite(p)&np.isfinite(y)&((y==0)|(y==1))
                if mask.sum()<20: continue
                if len(np.unique(y[mask]))<2: continue
                try: trait_aucs.append(roc_auc_score(y[mask], p[mask]))
                except Exception: continue
            if trait_aucs:
                boot_aucs.append(float(np.mean(trait_aucs)))
        if boot_aucs:
            lo=float(np.percentile(boot_aucs,2.5)); hi=float(np.percentile(boot_aucs,97.5))
            hash_grouped_payload["uncertainty"]={
                "method":"cluster-aware bootstrap at leakage-safe group level",
                "n_bootstrap":n_boot, "n_groups":len(group_ids),
                "macro_roc_ci95":[fin(lo),fin(hi)],
                "macro_roc_boot_mean":fin(np.mean(boot_aucs)),
                "note":"Resamples whole connected components, not rows; one-class bootstrap samples skipped. Fixed seed 42.",
            }
            (EXP_DIR/"semantic_hash_grouped_metrics.json").write_text(json.dumps(hash_grouped_payload, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Bootstrap CI95 macro ROC: [{lo:.4f}, {hi:.4f}] over {n_boot} resamples")
    except Exception as e:
        print("bootstrap failed:", str(e)[:200])
    # cluster bootstrap placeholder (not heavy)
    print("Done stages C-K")
if __name__=="__main__":
    main()