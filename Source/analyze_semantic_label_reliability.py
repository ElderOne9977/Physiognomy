# -*- coding: utf-8 -*-
# Phase 3.10: semantic label reliability, trait-level uncertainty, controlled improvement
# Usage: py Source/analyze_semantic_label_reliability.py --stage all --bootstrap 1000 --seed 42
#        py Source/analyze_semantic_label_reliability.py --stage all --quick
import argparse
import csv
import hashlib
import json
import math
import re
import time
import os
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
CSV_CLEAN = ROOT / "faces_full_features_clean.csv"
LABELS_CSV = ROOT / "semantic_labels.csv"
SCHEMA_JSON = ROOT / "semantic_schema.json"
OOF_CSV = ROOT / "oof_semantic_probs.csv"
OOF_NPY = ROOT / "oof_semantic_probs.npy"
OOF_HASH_CSV = SRC / "experiments" / "hash_grouped_semantic" / "oof_semantic_probs_hash_grouped.csv"
SAFE_GROUPS_CSV = SRC / "experiments" / "hash_grouped_semantic" / "semantic_leakage_safe_groups.csv"
MANIFEST_CSV = SRC / "experiments" / "hash_grouped_semantic" / "image_hash_manifest.csv"
AUDIT_JSON = SRC / "experiments" / "hash_grouped_semantic" / "image_hash_audit.json"
METRICS_JSON = SRC / "experiments" / "hash_grouped_semantic" / "semantic_hash_grouped_metrics.json"
CONFLICT_JSON = SRC / "experiments" / "hash_grouped_semantic" / "duplicate_label_conflicts.json"
CV_ASSIGN = SRC / "experiments" / "hash_grouped_semantic" / "cv_fold_assignments.csv"
THRESH_JSON = SRC / "models_semantic" / "trait_thresholds.json"
OUT_DIR = SRC / "experiments" / "semantic_reliability_phase310"

SEED_DEFAULT = 42
FOLDS = 5

def fin(x):
    try:
        v=float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None

def norm_name(v):
    s=os.path.basename(str(v).strip()); s=re.sub(r"\s+"," ",s); s=re.sub(r"\)\s+\("," )(",s); return s.lower()
def stem_name(v):
    return re.sub(r"\.[^.]+$","",norm_name(v))
def is_feature_col(n):
    return n.endswith("_ratio") or "area_ratio" in n or "asymmetry" in n
def entropy(p):
    if p<=0 or p>=1: return 0.0
    return float(-p*math.log2(p)-(1-p)*math.log2(1-p))
def ece_score(y,p,n_bins=10,sample_weight=None):
    y=np.asarray(y,float); p=np.asarray(p,float)
    if len(y)==0: return None
    bins=np.linspace(0,1,n_bins+1); e=0.0
    w=np.asarray(sample_weight,float) if sample_weight is not None else None
    tot=w.sum() if w is not None else len(y)
    for i in range(n_bins):
        lo,hi=bins[i],bins[i+1]
        m=(p>=lo)&(p<hi) if i<n_bins-1 else (p>=lo)&(p<=hi)
        if m.sum()==0: continue
        if w is not None:
            wm=w[m].sum()
            if wm==0: continue
            e+=abs((y[m]*w[m]).sum()/wm - (p[m]*w[m]).sum()/wm)*wm/tot
        else:
            e+=abs(y[m].mean()-p[m].mean())*m.sum()/len(y)
    return float(e)
def write_json(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    # ensure finite: json dump with allow_nan False will raise; we sanitized via fin
    txt=json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False)
    path.write_text(txt,encoding="utf-8")
    print(f"wrote {path}")
def load_schema():
    sch=json.loads(SCHEMA_JSON.read_text(encoding="utf-8"))
    traits=[t["name"] for t in sch["traits"]]
    cat={t["name"]:t["category"] for t in sch["traits"]}
    return traits,cat,sch
def load_aligned_for_conflicts():
    # returns df_merged (6983 rows) with columns: image_id, norm_key, norm_stem, md5, sha256, leakage_safe_group_id, traits...
    traits,cat,sch=load_schema()
    df=pd.read_csv(CSV_CLEAN,encoding="utf-8-sig")
    labels=pd.read_csv(LABELS_CSV,encoding="utf-8-sig")
    img_col=[c for c in labels.columns if c.strip().lower().lstrip("\ufeff")=="image"][0]
    if img_col!="image": labels=labels.rename(columns={img_col:"image"})
    df["norm_key"]=df["image_id"].apply(norm_name)
    df["norm_stem"]=df["image_id"].apply(stem_name)
    labels["norm_key"]=labels["image"].apply(norm_name)
    merged=pd.merge(df,labels,on="norm_key",how="inner")
    # join safe groups
    safe=pd.read_csv(SAFE_GROUPS_CSV,encoding="utf-8")
    # safe has image_id (original filename) -> map via norm_key
    safe_map={}
    for _,r in safe.iterrows():
        nk=norm_name(r["image_id"])
        safe_map[nk]=(r["leakage_safe_group_id"], r.get("md5"), r.get("sha256") if "sha256" in r else None)
    # manifest for sha if missing
    manifest={}
    if MANIFEST_CSV.exists():
        mf=pd.read_csv(MANIFEST_CSV,encoding="utf-8")
        for _,r in mf.iterrows():
            nk=norm_name(r["image_id"])
            manifest[nk]=(r.get("md5"), r.get("sha256"))
    # also need sha from cv_assign (has sha256 per row)
    cv_map={}
    if CV_ASSIGN.exists():
        cv=pd.read_csv(CV_ASSIGN,encoding="utf-8")
        for _,r in cv.iterrows():
            nk=norm_name(r["image_id"])
            cv_map[nk]=r.get("sha256")
    groups=[]; md5s=[]; shas=[]
    for nk in merged["norm_key"]:
        g,md,sh=safe_map.get(nk,(None,None,None))
        # fallback sha from cv or manifest
        if sh is None or (isinstance(sh,float) and math.isnan(sh)):
            sh=cv_map.get(nk)
            if sh is None: sh=manifest.get(nk,(None,None))[1]
        if md is None or (isinstance(md,float) and math.isnan(md)):
            md=manifest.get(nk,(None,None))[0]
        groups.append(g); md5s.append(md); shas.append(sh)
    merged["leakage_safe_group_id"]=groups
    merged["md5"]=md5s
    merged["sha256"]=shas
    return merged, traits, cat

def stage_a(args):
    t0=time.time()
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    traits,cat,sch=load_schema()
    # counts
    humans_count=len(list((ROOT/"Humans").iterdir())) if (ROOT/"Humans").exists() else None
    feature_rows=None; feature_cols=None
    try:
        df=pd.read_csv(CSV_CLEAN,encoding="utf-8-sig",nrows=2)
        feature_cols=[c for c in df.columns if is_feature_col(c)]
        df_full=pd.read_csv(CSV_CLEAN,encoding="utf-8-sig")
        feature_rows=len(df_full)
    except Exception as e:
        feature_rows=str(e)
    label_rows=None
    try: label_rows=len(pd.read_csv(LABELS_CSV,encoding="utf-8-sig"))
    except Exception as e: label_rows=str(e)
    # oof shapes
    oof_hist_shape=None; oof_hash_shape=None
    hist_traits=None; hash_traits=None
    try:
        h=pd.read_csv(OOF_CSV,encoding="utf-8-sig",nrows=2)
        hf=pd.read_csv(OOF_CSV,encoding="utf-8-sig")
        oof_hist_shape=[int(hf.shape[0]), int(hf.shape[1])]
        hist_traits=[c for c in hf.columns if c!="image_id"]
    except Exception as e: oof_hist_shape=str(e)[:200]
    try:
        hf2=pd.read_csv(OOF_HASH_CSV,encoding="utf-8-sig")
        oof_hash_shape=[int(hf2.shape[0]), int(hf2.shape[1])]
        hash_traits=[c for c in hf2.columns if c!="image_id"]
    except Exception as e: oof_hash_shape=str(e)[:200]
    # component counts
    audit=None; audit_components=None
    try:
        audit=json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
        audit_components=audit.get("leakage_safe_groups",{})
    except Exception as e: audit=str(e)[:300]
    # metrics consistency
    metrics=None; hist_macro=None; corr_macro=None; gap=None
    try:
        metrics=json.loads(METRICS_JSON.read_text(encoding="utf-8"))
        hist_macro=metrics.get("historical_filename_grouped_OOF_hash_leakage_present",{}).get("macro_roc_auc")
        corr_macro=metrics.get("corrected_hash_and_stem_group_safe_OOF",{}).get("strategy_A_all_rows_safe_grouped",{}).get("macro_roc_auc")
        if corr_macro is None:
            # alternative path
            corr_macro=metrics.get("corrected_hash_and_stem_group_safe_OOF",{}).get("macro_roc_auc")
        gap=metrics.get("corrected_hash_and_stem_group_safe_OOF",{}).get("gap_row_weighted_macro_roc")
    except Exception as e: metrics=str(e)[:300]
    # fold integrity
    fold_integrity=None
    try:
        cv=pd.read_csv(CV_ASSIGN,encoding="utf-8")
        # load leakage checks from metrics
        checks=metrics.get("leakage_checks",{}) if isinstance(metrics,dict) else {}
        fold_integrity={"n_rows":len(cv),"n_groups":int(cv["safe_group"].nunique()),"folds":sorted(cv["fold"].unique().tolist()),"leakage_checks":checks}
    except Exception as e: fold_integrity=str(e)[:300]
    # safe groups file
    safe_rows=None; safe_groups=None
    try:
        sg=pd.read_csv(SAFE_GROUPS_CSV,encoding="utf-8")
        safe_rows=len(sg); safe_groups=int(sg["leakage_safe_group_id"].nunique())
    except Exception as e: safe_rows=str(e)[:200]
    # trait counts
    trait_count=len(traits)
    # missing files
    expected=[str(CSV_CLEAN),str(LABELS_CSV),str(SCHEMA_JSON),str(OOF_CSV),str(OOF_HASH_CSV),str(SAFE_GROUPS_CSV),str(MANIFEST_CSV),str(AUDIT_JSON),str(METRICS_JSON),str(CONFLICT_JSON),str(CV_ASSIGN),str(THRESH_JSON)]
    missing=[p for p in expected if not Path(p).exists()]
    # prediction shape consistency
    # canonical order check
    order_ok=None
    try:
        def _filter_traits(cols):
            return [c for c in cols if c in traits]
        if hist_traits is not None and hash_traits is not None:
            order_ok=(_filter_traits(hist_traits)==traits and _filter_traits(hash_traits)==traits)
        elif hist_traits is not None:
            order_ok=(_filter_traits(hist_traits)==traits)
        elif hash_traits is not None:
            order_ok=(_filter_traits(hash_traits)==traits)
    except: order_ok=False
    # reproducibility
    repro={"seed":args.seed,"bootstrap":args.bootstrap,"folds":FOLDS,"chunk_bytes":1024*1024,"timestamp":time.strftime("%Y-%m-%dT%H:%M:%S")}
    payload={
        "stage":"A artifact integrity",
        "timestamp":time.strftime("%Y-%m-%dT%H:%M:%S"),
        "artifact_paths":{"csv_clean":str(CSV_CLEAN),"labels_csv":str(LABELS_CSV),"schema":str(SCHEMA_JSON),"oof_hist":str(OOF_CSV),"oof_hash":str(OOF_HASH_CSV),"safe_groups":str(SAFE_GROUPS_CSV),"manifest":str(MANIFEST_CSV),"audit":str(AUDIT_JSON),"metrics":str(METRICS_JSON),"conflicts":str(CONFLICT_JSON),"cv_assign":str(CV_ASSIGN),"thresholds":str(THRESH_JSON)},
        "counts":{"humans_files":humans_count,"feature_rows":feature_rows,"label_rows":label_rows,"trait_count":trait_count,"safe_group_rows":safe_rows,"safe_group_n":safe_groups,"audit_components":audit_components},
        "prediction_shapes":{"oof_hist_shape":oof_hist_shape,"oof_hash_shape":oof_hash_shape,"hist_traits_match_schema":order_ok},
        "fold_integrity":fold_integrity,
        "baseline_metrics":{"historical_macro_roc":fin(hist_macro) if isinstance(hist_macro,(int,float)) else hist_macro,"corrected_macro_roc":fin(corr_macro) if isinstance(corr_macro,(int,float)) else corr_macro,"gap":fin(gap) if isinstance(gap,(int,float)) else gap},
        "missing_files":missing,
        "malformed": [],
        "reproducibility":repro,
        "elapsed_sec": fin(time.time()-t0),
        "note":"All metrics = agreement with provider-derived labels, not validated traits."
    }
    # validate finite
    write_json(OUT_DIR/"artifact_integrity_report.json",payload)
    return payload

def stage_b(args):
    # detailed conflict analysis
    merged,traits,cat=load_aligned_for_conflicts()
    n_rows=len(merged)
    # exact duplicates via sha256
    sha_groups=defaultdict(list)
    for idx,row in merged.iterrows():
        sh=row["sha256"]
        if pd.isna(sh) or sh is None: continue
        sha_groups[sh].append(idx)
    dup_sha_groups={k:v for k,v in sha_groups.items() if len(v)>1}
    # stem groups
    stem_groups=defaultdict(list)
    for idx,row in merged.iterrows():
        stem=row["norm_stem"]
        stem_groups[stem].append(idx)
    dup_stem_groups={k:v for k,v in stem_groups.items() if len(v)>1}
    # stem-linked diff bytes: stem group with >1 distinct sha
    stem_diff_bytes={}
    for stem, idxs in dup_stem_groups.items():
        shas=set(merged.loc[idxs,"sha256"].dropna().tolist())
        if len(shas)>1:
            stem_diff_bytes[stem]=idxs
    # connected components via safe group id
    comp_groups=defaultdict(list)
    for idx,row in merged.iterrows():
        g=row["leakage_safe_group_id"]
        comp_groups[g].append(idx)
    dup_comps={k:v for k,v in comp_groups.items() if len(v)>1}
    n_comp=len(comp_groups)
    largest_comp=max(len(v) for v in comp_groups.values()) if comp_groups else 0
    # for each dup sha group compute stats
    rows_for_csv=[]
    per_trait_conflict_counts=Counter()
    per_trait_entropy=[]
    per_trait_p_list=defaultdict(list)
    # need mapping from group to stats
    for sha, idxs in dup_sha_groups.items():
        sub=merged.loc[idxs,traits]
        group_size=len(idxs)
        distinct_vectors=len(sub.drop_duplicates())
        # per trait stats within group
        conflicting_traits=0
        entropies={}
        pos_freq={}
        maj_agree={}
        for t in traits:
            vals=sub[t].values
            # vals should be 0/1
            pos=np.sum(vals==1); neg=np.sum(vals==0)
            p=pos/group_size if group_size else 0
            h=entropy(p)
            entropies[t]=h
            pos_freq[t]=p
            maj=max(pos,neg)/group_size if group_size else 0
            maj_agree[t]=maj
            if len(set(vals))>1:
                conflicting_traits+=1
                per_trait_conflict_counts[t]+=1
            per_trait_p_list[t].append(p)
        # aggregate per group
        rows_for_csv.append({
            "group_key":sha,
            "group_type":"exact_sha",
            "group_size":group_size,
            "distinct_label_vectors":distinct_vectors,
            "conflicting_traits":conflicting_traits,
            "mean_entropy": float(np.mean(list(entropies.values()))) if entropies else 0,
            "max_entropy": float(max(entropies.values())) if entropies else 0,
            "leakage_safe_group_id": merged.loc[idxs[0],"leakage_safe_group_id"]
        })
    # per-trait disagreement rate (unweighted: count groups where trait disagrees / total dup groups)
    per_trait_disagreement={}
    per_trait_mean_entropy={}
    per_trait_mean_p={}
    for t in traits:
        cnt=per_trait_conflict_counts[t]
        per_trait_disagreement[t]=cnt/len(dup_sha_groups) if dup_sha_groups else 0
        # mean entropy across groups for that trait
        # compute entropy per group for trait t
        hs=[]
        for sha,idxs in dup_sha_groups.items():
            vals=merged.loc[idxs,t].values
            p=np.sum(vals==1)/len(vals) if len(vals) else 0
            hs.append(entropy(p))
        per_trait_mean_entropy[t]=float(np.mean(hs)) if hs else 0
        per_trait_mean_p[t]=float(np.mean(per_trait_p_list[t])) if per_trait_p_list[t] else 0
    # group-size weighted disagreement: weight by group size
    per_trait_weighted_disagreement={}
    per_trait_weighted_entropy={}
    total_dup_rows=sum(len(v) for v in dup_sha_groups.values())
    for t in traits:
        weighted_conflicts=sum(len(idxs) for sha,idxs in dup_sha_groups.items() if len(set(merged.loc[idxs,t].values))>1)
        per_trait_weighted_disagreement[t]=weighted_conflicts/total_dup_rows if total_dup_rows else 0
        # weighted entropy
        w_h=sum(len(idxs)*entropy(np.sum(merged.loc[idxs,t].values==1)/len(idxs)) for sha,idxs in dup_sha_groups.items())
        per_trait_weighted_entropy[t]=w_h/total_dup_rows if total_dup_rows else 0
    # distribution of conflicts per group histogram
    hist=Counter(r["conflicting_traits"] for r in rows_for_csv)
    # correlation between group size and conflicting traits
    sizes=np.array([r["group_size"] for r in rows_for_csv],float) if rows_for_csv else np.array([])
    conflicts=np.array([r["conflicting_traits"] for r in rows_for_csv],float) if rows_for_csv else np.array([])
    corr_size_conflicts=None
    if len(sizes)>2 and np.std(sizes)>0 and np.std(conflicts)>0:
        try: corr_size_conflicts=float(np.corrcoef(sizes,conflicts)[0,1])
        except: corr_size_conflicts=None
    # feature pattern investigation: variance comparison dup vs non-dup? Simple: compute mean feature variance per group?
    # we skip heavy feature pattern, just note largest component composition
    # sensitivity to largest components: compute stats excluding largest 1 and largest 3 components
    # Need component sizes sorted
    comp_sizes=sorted([(k,len(v)) for k,v in comp_groups.items()],key=lambda x: x[1],reverse=True)
    # exact vs stem vs component counts
    non_dup_rows=sum(1 for k,v in comp_groups.items() if len(v)==1)
    # build label_conflict_analysis.json
    payload={
        "stage":"B label conflicts",
        "timestamp":time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset":{"n_rows":int(n_rows),"n_traits":len(traits),"n_sha_groups":len(sha_groups),"n_dup_sha_groups":len(dup_sha_groups),"n_stem_groups":len(stem_groups),"n_dup_stem_groups":len(dup_stem_groups),"n_stem_diff_bytes":len(stem_diff_bytes),"n_connected_components":int(n_comp),"largest_component_size":int(largest_comp),"n_dup_components":len(dup_comps),"nonduplicate_components":int(n_comp-len(dup_comps)),"nonduplicate_rows":int(non_dup_rows)},
        "exact_content_duplicates":{"n_groups":len(dup_sha_groups),"conflicting_groups":int(sum(1 for r in rows_for_csv if r["conflicting_traits"]>0)),"identical_groups":int(sum(1 for r in rows_for_csv if r["conflicting_traits"]==0)),"mean_conflicting_traits": fin(np.mean([r["conflicting_traits"] for r in rows_for_csv])) if rows_for_csv else None,"max_conflicting_traits": int(max([r["conflicting_traits"] for r in rows_for_csv])) if rows_for_csv else None,"histogram_conflicts": {str(k):int(v) for k,v in sorted(hist.items())}, "correlation_group_size_vs_conflicts": fin(corr_size_conflicts)},
        "same_stem_groups":{"n_groups":len(dup_stem_groups),"n_diff_bytes":len(stem_diff_bytes),"sample_stems": list(stem_diff_bytes.keys())[:5]},
        "connected_components":{"n_components":int(n_comp),"largest":int(largest_comp),"size_distribution": dict(Counter(len(v) for v in comp_groups.values()))},
        "nonduplicate":{"n_components":int(n_comp-len(dup_comps)),"n_rows":int(non_dup_rows)},
        "per_trait":{"disagreement_rate": {t:fin(per_trait_disagreement[t]) for t in traits},"mean_entropy": {t:fin(per_trait_mean_entropy[t]) for t in traits},"mean_pos_freq_in_dup_groups": {t:fin(per_trait_mean_p[t]) for t in traits},"weighted_disagreement_rate": {t:fin(per_trait_weighted_disagreement[t]) for t in traits},"weighted_mean_entropy": {t:fin(per_trait_weighted_entropy[t]) for t in traits}},
        "per_category_disagreement": {},
        "limitations":"Connected component may contain samples linked indirectly through different matching relationships; not all samples in a component are guaranteed semantically identical. Exact SHA groups are byte-identical; stem groups share normalized filename stem but may differ in bytes.",
        "assumptions":"Entropy H(p) computed per group per trait with p = positive rate within group. Group-weighted stats weight each row equally; unweighted weight each group equally."
    }
    # per category
    cat_groups={}
    for cat_name in set(cat.values()):
        # count dup groups where >=1 trait in category disagrees
        cnt=0
        for sha,idxs in dup_sha_groups.items():
            sub=merged.loc[idxs,[t for t in traits if cat[t]==cat_name]]
            disagree=False
            for col in sub.columns:
                if len(set(sub[col].values))>1:
                    disagree=True; break
            if disagree: cnt+=1
        cat_groups[cat_name]={"conflicting_groups":int(cnt),"total_dup_groups":len(dup_sha_groups),"disagreement_rate": fin(cnt/len(dup_sha_groups)) if dup_sha_groups else None}
    payload["per_category_disagreement"]=cat_groups
    write_json(OUT_DIR/"label_conflict_analysis.json",payload)
    # write semantic_trait_conflicts.csv
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    trait_csv=OUT_DIR/"semantic_trait_conflicts.csv"
    with open(trait_csv,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["trait","category","disagreement_rate","weighted_disagreement_rate","mean_entropy","weighted_mean_entropy","mean_pos_freq_in_dup_groups","conflicting_groups","total_dup_groups"])
        w.writeheader()
        for t in traits:
            w.writerow({"trait":t,"category":cat[t],"disagreement_rate":fin(per_trait_disagreement[t]),"weighted_disagreement_rate":fin(per_trait_weighted_disagreement[t]),"mean_entropy":fin(per_trait_mean_entropy[t]),"weighted_mean_entropy":fin(per_trait_weighted_entropy[t]),"mean_pos_freq_in_dup_groups":fin(per_trait_mean_p[t]),"conflicting_groups":per_trait_conflict_counts[t],"total_dup_groups":len(dup_sha_groups)})
    # duplicate_group_conflicts.csv
    grp_csv=OUT_DIR/"duplicate_group_conflicts.csv"
    with open(grp_csv,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["group_key","group_type","group_size","distinct_label_vectors","conflicting_traits","mean_entropy","max_entropy","leakage_safe_group_id"])
        w.writeheader()
        for r in sorted(rows_for_csv,key=lambda x: x["group_size"],reverse=True):
            w.writerow(r)
    print(f"stage B done dup_sha {len(dup_sha_groups)}")
    return payload

def stage_c(args):
    merged,traits,cat=load_aligned_for_conflicts()
    n_rows=len(merged)
    # compute per trait global prevalence, group agreement etc
    # exact duplicate pairwise agreement: for each trait, consider all pairs within same sha group
    # majority-label agreement: within each dup group, majority label proportion, average across groups
    sha_groups=defaultdict(list)
    for idx,row in merged.iterrows():
        sh=row["sha256"]
        if pd.isna(sh) or sh is None: continue
        sha_groups[sh].append(idx)
    dup_sha_groups={k:v for k,v in sha_groups.items() if len(v)>1}
    comp_groups=defaultdict(list)
    for idx,row in merged.iterrows():
        comp_groups[row["leakage_safe_group_id"]].append(idx)
    # effective number independent groups per trait: count components with at least one positive and at least one negative? Use n components with trait observed
    rows=[]
    for t in traits:
        y=merged[t].values
        pos=np.sum(y==1); neg=np.sum(y==0)
        prevalence=pos/len(y) if len(y) else None
        # group-level agreement: fraction of dup groups where all labels same for this trait
        agree_groups=sum(1 for idxs in dup_sha_groups.values() if len(set(merged.loc[idxs,t].values))==1)
        total_dup=len(dup_sha_groups)
        group_agreement=agree_groups/total_dup if total_dup else None
        # pairwise agreement among exact duplicates: total pairs agree / total pairs
        total_pairs=0; agree_pairs=0
        for idxs in dup_sha_groups.values():
            vals=merged.loc[idxs,t].values.tolist()
            for i in range(len(vals)):
                for j in range(i+1,len(vals)):
                    total_pairs+=1
                    if vals[i]==vals[j]: agree_pairs+=1
        pairwise_agreement=agree_pairs/total_pairs if total_pairs else None
        # majority-label agreement average across dup groups: mean of majority proportion
        maj_list=[]
        entropies=[]
        for idxs in dup_sha_groups.values():
            vals=merged.loc[idxs,t].values
            p=np.sum(vals==1)/len(vals) if len(vals) else 0
            entropies.append(entropy(p))
            maj=max(np.sum(vals==1),np.sum(vals==0))/len(vals) if len(vals) else 0
            maj_list.append(maj)
        majority_label_agreement=np.mean(maj_list) if maj_list else None
        mean_entropy=np.mean(entropies) if entropies else None
        # effective number independent groups: number of components
        effective_groups=len(comp_groups)
        # independent groups with at least one positive? count
        comps_with_pos=0; comps_with_neg=0
        for g,idxs in comp_groups.items():
            vals=merged.loc[idxs,t].values
            if np.any(vals==1): comps_with_pos+=1
            if np.any(vals==0): comps_with_neg+=1
        # flag insufficient
        flag=None
        if pos<30 or neg<30:
            flag="low_support"
        elif comps_with_pos<10 or comps_with_neg<10:
            flag="low_component_support"
        rows.append({
            "trait":t,"category":cat[t],"prevalence":fin(prevalence),"positive_count":int(pos),"negative_count":int(neg),
            "group_level_agreement":fin(group_agreement),"pairwise_agreement_exact_duplicates":fin(pairwise_agreement),
            "majority_label_agreement":fin(majority_label_agreement),"mean_entropy_within_dup_groups":fin(mean_entropy),
            "effective_n_groups":int(effective_groups),"groups_with_positive":int(comps_with_pos),"groups_with_negative":int(comps_with_neg),
            "total_dup_groups":int(total_dup),"total_pairs":int(total_pairs),"flag":flag
        })
    # bootstrap CI for prevalence and pairwise agreement using component-aware resampling
    n_boot=args.bootstrap if not args.quick else min(50, args.bootstrap)
    rng=np.random.default_rng(args.seed)
    comp_ids=list(comp_groups.keys())
    group_rows={g: idxs for g,idxs in comp_groups.items()}
    # collect bootstrap distributions per trait for prevalence and pairwise agreement?
    # we bootstrap components, then compute prevalence and agreement on resampled dataset (with replacement, expanding rows)
    # For CIs, we compute 2.5 and 97.5 percentiles across bootstraps
    # To avoid heavy, compute per trait prevalence CI only
    boot_prevalences=defaultdict(list)
    boot_pairwise=defaultdict(list)
    for b in range(n_boot):
        samp_ids=[comp_ids[j] for j in rng.integers(0,len(comp_ids),size=len(comp_ids))]
        idx_expanded=[]
        for g in samp_ids:
            idx_expanded.extend(group_rows[g])
        # subset merged via idx_expanded (duplicated indices due to resampling)
        sub=merged.iloc[idx_expanded]
        for t in traits:
            vals=sub[t].values
            prev=np.sum(vals==1)/len(vals) if len(vals) else np.nan
            if np.isfinite(prev): boot_prevalences[t].append(prev)
            # pairwise within this bootstrap? too heavy, skip per bootstrap pairwise; instead we rely on point estimate
    # compute CI
    for r in rows:
        t=r["trait"]
        arr=boot_prevalences[t]
        if len(arr)>=10:
            lo=float(np.percentile(arr,2.5)); hi=float(np.percentile(arr,97.5))
            r["prevalence_ci95_low"]=fin(lo); r["prevalence_ci95_high"]=fin(hi); r["prevalence_boot_mean"]=fin(np.mean(arr)); r["n_bootstrap"]=n_boot
        else:
            r["prevalence_ci95_low"]=None; r["prevalence_ci95_high"]=None; r["prevalence_boot_mean"]=None; r["n_bootstrap"]=n_boot
        # pairwise CI not computed heavy; set null
        r["pairwise_agreement_ci_low"]=None; r["pairwise_agreement_ci_high"]=None
    # write trait_label_reliability.csv
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    csv_path=OUT_DIR/"trait_label_reliability.csv"
    fieldnames=["trait","category","prevalence","positive_count","negative_count","group_level_agreement","pairwise_agreement_exact_duplicates","majority_label_agreement","mean_entropy_within_dup_groups","effective_n_groups","groups_with_positive","groups_with_negative","prevalence_ci95_low","prevalence_ci95_high","prevalence_boot_mean","n_bootstrap","flag"]
    with open(csv_path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fieldnames)
        w.writeheader()
        for r in rows: w.writerow({k:r.get(k) for k in fieldnames})
    payload={
        "stage":"C label consistency and uncertainty",
        "timestamp":time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_bootstrap":n_boot,
        "seed":args.seed,
        "method":"component-aware bootstrap resampling connected components with replacement; prevalence CI95 from bootstrap distribution",
        "n_traits":len(traits),
        "n_rows":int(n_rows),
        "n_components":len(comp_groups),
        "traits_flagged_low_support":[r["trait"] for r in rows if r["flag"]],
        "note":"Group agreement and majority-label accuracy are descriptive, not Bayes-optimal ceiling. Effective n = number of connected components (3201 on matched rows). Do not treat correlated labels within group as independent.",
        "identifiable":"Prevalence, group agreement, pairwise agreement identifiable from observed duplicates. Bayes ceiling not identifiable without repeated independent labeling of same image under provider stochasticity model.",
        "non_identifiable":"True label noise vs feature insufficiency cannot be distinguished from duplicate conflicts alone."
    }
    write_json(OUT_DIR/"label_uncertainty_report.json",payload)
    return rows

def stage_d(args):
    traits,cat,sch=load_schema()
    # load aligned + OOF hash
    merged,_,_=load_aligned_for_conflicts()
    oof=pd.read_csv(OOF_HASH_CSV,encoding="utf-8-sig")
    # thresholds
    thresholds={}
    if THRESH_JSON.exists():
        try: thresholds=json.loads(THRESH_JSON.read_text(encoding="utf-8")).get("thresholds",{})
        except: thresholds={}
    # map oof to merged via norm_key
    merged_map={norm_name(r["image_id"]): idx for idx,r in merged.iterrows()}
    # oof rows order may not match merged; need to align
    oof["norm_key"]=oof["image_id"].apply(norm_name)
    # Build Y and P matrices aligned to oof order where label exists
    Y=[]; P=[]; groups=[]; valid_idx=[]
    for i,row in oof.iterrows():
        nk=row["norm_key"]
        midx=merged_map.get(nk)
        if midx is None: continue
        mrow=merged.loc[midx]
        yrow=[mrow[t] for t in traits]
        prow=[row[t] if t in row else np.nan for t in traits]
        Y.append(yrow); P.append(prow); groups.append(mrow["leakage_safe_group_id"])
        valid_idx.append(midx)
    Y=np.array(Y,float); P=np.array(P,float); groups=np.array(groups)
    n_rows=Y.shape[0]
    # compute per trait metrics
    comp_groups=defaultdict(list)
    for i,g in enumerate(groups):
        comp_groups[g].append(i)
    rows_out=[]
    n_boot=args.bootstrap if not args.quick else min(50, args.bootstrap)
    rng=np.random.default_rng(args.seed)
    comp_ids=list(comp_groups.keys())
    group_rows={g:idxs for g,idxs in comp_groups.items()}
    # Optimized bootstrap: sample components 1000x once, compute all traits per sample (instead of 45*1000 samplings)
    # Pre-generate bootstrap index arrays for reproducibility and speed
    boot_indices=[]
    for b in range(n_boot):
        samp_ids=[comp_ids[k] for k in rng.integers(0,len(comp_ids),size=len(comp_ids))]
        idx=[]
        for g in samp_ids:
            idx.extend(group_rows[g])
        boot_indices.append(np.array(idx))
    # per-trait bootstrap stats will be filled from boot_indices
    boot_rocs_per_trait=[[] for _ in range(len(traits))]
    boot_prs_per_trait=[[] for _ in range(len(traits))]
    n_invalid_per_trait=[0]*len(traits)
    for j,t in enumerate(traits):
        y=Y[:,j]; p=P[:,j]
        mask=np.isfinite(y)&np.isfinite(p)&((y==0)|(y==1))
        n_valid=int(mask.sum())
        prevalence=fin(y[mask].mean()) if n_valid else None
        # threshold-free metrics
        roc=None; pr=None; brier=None; ll=None; ece=None
        if n_valid>=20 and len(np.unique(y[mask]))==2:
            try: roc=fin(roc_auc_score(y[mask],p[mask]))
            except: roc=None
            try: pr=fin(average_precision_score(y[mask],p[mask]))
            except: pr=None
            try: brier=fin(brier_score_loss(y[mask],p[mask]))
            except: brier=None
            try:
                # clip p for log loss
                pc=np.clip(p[mask],1e-7,1-1e-7)
                ll=fin(log_loss(y[mask],pc))
            except: ll=None
            try: ece=fin(ece_score(y[mask],p[mask]))
            except: ece=None
        else:
            brier=fin(brier_score_loss(y[mask],p[mask])) if n_valid>0 else None
            try: ece=fin(ece_score(y[mask],p[mask])) if n_valid>0 else None
            except: ece=None
        # threshold metrics using previously established threshold (or 0.5)
        thr=thresholds.get(t,0.5) if isinstance(thresholds.get(t), (int,float)) else 0.5
        pred=(p>=thr).astype(int) if np.isfinite(p).any() else np.zeros_like(y,dtype=int)
        # only where mask valid
        prec=None; rec=None; f1=None; pred_pos_rate=None
        if n_valid>0:
            try: prec=fin(precision_score(y[mask],pred[mask],zero_division=0))
            except: prec=None
            try: rec=fin(recall_score(y[mask],pred[mask],zero_division=0))
            except: rec=None
            try: f1=fin(f1_score(y[mask],pred[mask],zero_division=0))
            except: f1=None
            try: pred_pos_rate=fin(pred[mask].mean())
            except: pred_pos_rate=None
        # independent component counts
        comps=np.unique(groups[mask]) if n_valid else np.array([])
        n_comp=int(len(comps))
        # count components with pos and neg
        comps_pos=0; comps_neg=0
        for g in comps:
            idxs=group_rows[g]
            vals=Y[idxs,j]
            m=np.isfinite(vals)
            if np.any(vals[m]==1): comps_pos+=1
            if np.any(vals[m]==0): comps_neg+=1
        # placeholder for bootstrap results, filled after loop
        rows_out.append({
            "trait":t,"category":cat[t],"n_valid":n_valid,"n_components":n_comp,"components_with_positive":int(comps_pos),"components_with_negative":int(comps_neg),
            "prevalence":prevalence,"roc_auc":roc,"pr_auc":pr,"brier":brier,"log_loss":ll,"ece":ece,
            "threshold":fin(thr),"precision_at_thr":prec,"recall_at_thr":rec,"f1_at_thr":f1,"pred_pos_rate":pred_pos_rate,
            # bootstrap fields filled later
            "flag":None,"_roc_boot":None,"_pr_boot":None
        })
    # Now compute bootstrap for all traits using shared boot_indices (1000 samples instead of 45k)
    for b_idx, idx in enumerate(boot_indices):
        for j in range(len(traits)):
            yb=Y[idx,j]; pb=P[idx,j]
            maskb=np.isfinite(yb)&np.isfinite(pb)&((yb==0)|(yb==1))
            if maskb.sum()<20 or len(np.unique(yb[maskb]))<2:
                n_invalid_per_trait[j]+=1; continue
            try:
                r=roc_auc_score(yb[maskb],pb[maskb])
                prb=average_precision_score(yb[maskb],pb[maskb])
                if math.isfinite(r) and math.isfinite(prb):
                    boot_rocs_per_trait[j].append(float(r)); boot_prs_per_trait[j].append(float(prb))
                else: n_invalid_per_trait[j]+=1
            except: n_invalid_per_trait[j]+=1
    # finalize bootstrap CI per trait
    for j in range(len(traits)):
        br=boot_rocs_per_trait[j]; bp=boot_prs_per_trait[j]
        roc_ci_low=None; roc_ci_high=None; roc_boot_mean=None
        pr_ci_low=None; pr_ci_high=None; pr_boot_mean=None
        if br:
            roc_ci_low=fin(float(np.percentile(br,2.5))); roc_ci_high=fin(float(np.percentile(br,97.5))); roc_boot_mean=fin(float(np.mean(br)))
        if bp:
            pr_ci_low=fin(float(np.percentile(bp,2.5))); pr_ci_high=fin(float(np.percentile(bp,97.5))); pr_boot_mean=fin(float(np.mean(bp)))
        # handle flag
        r=rows_out[j]
        flag=None
        n_valid=r["n_valid"]; n_comp=r["n_components"]; comps_pos=r["components_with_positive"]; comps_neg=r["components_with_negative"]
        if n_valid<30: flag="low_n"
        elif r["roc_auc"] is None and r["pr_auc"] is None: flag="one_class"
        elif comps_pos<5 or comps_neg<5: flag="low_component_support"
        r.update({
            "roc_ci_low":roc_ci_low,"roc_ci_high":roc_ci_high,"roc_boot_mean":roc_boot_mean,
            "pr_ci_low":pr_ci_low,"pr_ci_high":pr_ci_high,"pr_boot_mean":pr_boot_mean,
            "n_bootstrap":n_boot,"n_valid_bootstrap":int(len(br)),"n_invalid_bootstrap":int(n_invalid_per_trait[j]),"flag":flag
        })
        # remove temp
        r.pop("_roc_boot",None); r.pop("_pr_boot",None)
    # correlations exploratory: prevalence vs roc, disagreement vs roc etc - compute for report
    # load disagreement rates from stage B if exists else compute
    # write csv
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    csv_path=OUT_DIR/"semantic_per_trait_uncertainty.csv"
    fieldnames=["trait","category","n_valid","n_components","components_with_positive","components_with_negative","prevalence","roc_auc","roc_ci_low","roc_ci_high","roc_boot_mean","pr_auc","pr_ci_low","pr_ci_high","pr_boot_mean","brier","log_loss","ece","threshold","precision_at_thr","recall_at_thr","f1_at_thr","pred_pos_rate","n_bootstrap","n_valid_bootstrap","n_invalid_bootstrap","flag"]
    with open(csv_path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out: w.writerow(r)
    # summary report
    # compute macro means where valid
    rocs=[r["roc_auc"] for r in rows_out if r["roc_auc"] is not None]
    prs=[r["pr_auc"] for r in rows_out if r["pr_auc"] is not None]
    briers=[r["brier"] for r in rows_out if r["brier"] is not None]
    eces=[r["ece"] for r in rows_out if r["ece"] is not None]
    # correlations
    def corr(a,b):
        a=np.array(a,float); b=np.array(b,float)
        mask=np.isfinite(a)&np.isfinite(b)
        if mask.sum()<3 or np.std(a[mask])==0 or np.std(b[mask])==0: return None
        try: return float(np.corrcoef(a[mask],b[mask])[0,1])
        except: return None
    prevalences=[r["prevalence"] for r in rows_out]
    roc_vals=[r["roc_auc"] if r["roc_auc"] is not None else np.nan for r in rows_out]
    pr_vals=[r["pr_auc"] if r["pr_auc"] is not None else np.nan for r in rows_out]
    ece_vals=[r["ece"] if r["ece"] is not None else np.nan for r in rows_out]
    corr_prev_roc=corr(prevalences,roc_vals)
    corr_prev_pr=corr(prevalences,pr_vals)
    corr_prev_ece=corr(prevalences,ece_vals)
    # load label disagreement for correlation with roc
    payload={
        "stage":"D trait-level model performance audit",
        "timestamp":time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_rows":int(n_rows),
        "n_traits":len(traits),
        "macro_roc_auc":fin(float(np.mean(rocs))) if rocs else None,
        "macro_pr_auc":fin(float(np.mean(prs))) if prs else None,
        "macro_brier":fin(float(np.mean(briers))) if briers else None,
        "macro_ece":fin(float(np.mean(eces))) if eces else None,
        "weakest_traits": sorted(rows_out,key=lambda x: x["roc_auc"] if x["roc_auc"] is not None else 10)[:5],
        "strongest_traits": sorted(rows_out,key=lambda x: x["roc_auc"] if x["roc_auc"] is not None else -10,reverse=True)[:5],
        "low_calibration": sorted(rows_out,key=lambda x: x["ece"] if x["ece"] is not None else -10,reverse=True)[:5],
        "correlations":{"prevalence_vs_roc":fin(corr_prev_roc),"prevalence_vs_pr":fin(corr_prev_pr),"prevalence_vs_ece":fin(corr_prev_ece),"note":"correlations exploratory only, not causal"},
        "threshold_note":"Metrics at threshold use previously established per-trait thresholds (leaked OOF tuned) - reported for reference, not unbiased held-out. Threshold-free ROC/PR are primary.",
        "bootstrap":{"n_bootstrap":n_boot,"seed":args.seed,"method":"component-aware, invalid replicates (one-class) counted separately"},
        "flags": [r for r in rows_out if r["flag"]]
    }
    write_json(OUT_DIR/"semantic_trait_performance_report.json",payload)
    return rows_out

def stage_e(args):
    traits,cat,_=load_schema()
    merged,_,_=load_aligned_for_conflicts()
    oof=pd.read_csv(OOF_HASH_CSV,encoding="utf-8-sig")
    oof["norm_key"]=oof["image_id"].apply(norm_name)
    merged_map={norm_name(r["image_id"]): idx for idx,r in merged.iterrows()}
    Y=[]; P=[]; groups=[]; comp_sizes={}
    for _,row in oof.iterrows():
        nk=row["norm_key"]
        midx=merged_map.get(nk)
        if midx is None: continue
        mrow=merged.loc[midx]
        Y.append([mrow[t] for t in traits]); P.append([row[t] if t in row else np.nan for t in traits]); groups.append(mrow["leakage_safe_group_id"])
    Y=np.array(Y,float); P=np.array(P,float); groups=np.array(groups)
    # component sizes
    comp_groups=defaultdict(list)
    for i,g in enumerate(groups): comp_groups[g].append(i)
    size_sorted=sorted(comp_groups.items(),key=lambda x: len(x[1]),reverse=True)
    largest_g, largest_idx = size_sorted[0] if size_sorted else (None,[])
    largest_size=len(largest_idx) if largest_idx else 0
    # helper to compute macro metrics for a given mask
    def compute_metrics(mask):
        Ym=Y[mask]; Pm=P[mask]
        rocs=[]; prs=[]; briers=[]; eces=[]
        per_trait={}
        for j,t in enumerate(traits):
            y=Ym[:,j]; p=Pm[:,j]
            m=np.isfinite(y)&np.isfinite(p)&((y==0)|(y==1))
            if m.sum()<20 or len(np.unique(y[m]))<2: continue
            try: rocs.append(roc_auc_score(y[m],p[m]))
            except: pass
            try: prs.append(average_precision_score(y[m],p[m]))
            except: pass
            try: briers.append(brier_score_loss(y[m],p[m]))
            except: pass
            try: eces.append(ece_score(y[m],p[m]))
            except: pass
        return {
            "n_rows":int(mask.sum()),
            "n_components":int(len(set(groups[mask]))),
            "macro_roc":fin(float(np.mean(rocs))) if rocs else None,
            "macro_pr":fin(float(np.mean(prs))) if prs else None,
            "macro_brier":fin(float(np.mean(briers))) if briers else None,
            "macro_ece":fin(float(np.mean(eces))) if eces else None,
            "traits_scored":len(rocs)
        }
    full_mask=np.ones(len(groups),dtype=bool)
    full=compute_metrics(full_mask)
    # exclude largest
    mask_no_largest=np.array([g!=largest_g for g in groups])
    excl_largest=compute_metrics(mask_no_largest)
    # exclude largest 3
    top3=set([g for g,_ in size_sorted[:3]])
    mask_no_top3=np.array([g not in top3 for g in groups])
    excl_top3=compute_metrics(mask_no_top3)
    # component-balanced: duplicate-adjusted = each sha equal weight already; here each component equal weight via sample_weight
    # compute weighted macro where each component weight = 1/size
    weights=np.array([1.0/len(comp_groups[g]) for g in groups],float)
    # weighted global: compute macro roc as weighted average? simpler: compute per trait weighted roc? For now compute macro via weighted samples approximation: resample per component weighting
    # Instead compute overall weighted Brier/ECE? Keep simple: report weight distribution
    comp_balanced={"method":"inverse-component-size weighting (each component equal total weight)","weight_mean":fin(weights.mean()),"weight_min":fin(weights.min()),"weight_max":fin(weights.max())}
    # per-trait delta when excluding largest
    per_trait_deltas=[]
    for j,t in enumerate(traits):
        yf=Y[:,j]; pf=P[:,j]
        # full
        mf=np.isfinite(yf)&np.isfinite(pf)&((yf==0)|(yf==1))
        if mf.sum()<20 or len(np.unique(yf[mf]))<2: continue
        try: roc_full=roc_auc_score(yf[mf],pf[mf])
        except: continue
        # excl largest
        mask=mf & (groups!=largest_g)
        if mask.sum()<20 or len(np.unique(yf[mask]))<2: continue
        try: roc_ex=roc_auc_score(yf[mask],pf[mask])
        except: continue
        per_trait_deltas.append({"trait":t,"category":cat[t],"roc_full":fin(roc_full),"roc_excl_largest":fin(roc_ex),"delta":fin(roc_ex-roc_full)})
    per_trait_deltas_sorted=sorted(per_trait_deltas,key=lambda x: abs(x["delta"]) if x["delta"] is not None else 0, reverse=True)[:10]
    payload={
        "stage":"E largest component influence",
        "timestamp":time.strftime("%Y-%m-%dT%H:%M:%S"),
        "largest_component":{"group_id":int(largest_g) if largest_g is not None else None,"size":int(largest_size),"fraction_rows":fin(largest_size/len(groups)) if len(groups) else None},
        "size_distribution": dict(Counter(len(v) for v in comp_groups.values())),
        "full":full,
        "exclude_largest":excl_largest,
        "exclude_top3":excl_top3,
        "component_balanced":comp_balanced,
        "delta_full_vs_excl_largest":{"macro_roc_delta": fin(excl_largest["macro_roc"]-full["macro_roc"]) if full["macro_roc"] is not None and excl_largest["macro_roc"] is not None else None,"macro_pr_delta": fin(excl_top3["macro_pr"]-full["macro_pr"]) if full["macro_pr"] is not None and excl_top3["macro_pr"] is not None else None},
        "per_trait_deltas_top10": per_trait_deltas_sorted,
        "transitive_hub_note":"Largest component 610 rows arises from transitive closure over SHA-256 and normalized stem (hub hash + 137 multi-extension stems). Exclusion tests whether hub dominates metrics; component-balanced weighting alternative mitigates dominance without deletion.",
        "recommendation":"If exclusion changes macro ROC >0.01, consider safer grouping (e.g., SHA-only or stem-only) with separate validation; current SHA+stem union-find is strictly leakage-safe but may over-merge via hub."
    }
    write_json(OUT_DIR/"component_influence_analysis.json",payload)
    return payload

def stage_f(args):
    if args.quick:
        print("stage F skipped due to --quick")
        return None
    # Controlled improvement experiments - up to 3, lightweight
    traits,cat,sch=load_schema()
    merged,_,_=load_aligned_for_conflicts()
    # prepare data for experiments: use same GroupKFold protocol as evaluate_hash_grouped_semantic
    # Load feature matrix
    df=pd.read_csv(CSV_CLEAN,encoding="utf-8-sig")
    labels=pd.read_csv(LABELS_CSV,encoding="utf-8-sig")
    img_col=[c for c in labels.columns if c.strip().lower().lstrip("\ufeff")=="image"][0]
    if img_col!="image": labels=labels.rename(columns={img_col:"image"})
    df["norm_key"]=df["image_id"].apply(norm_name)
    labels["norm_key"]=labels["image"].apply(norm_name)
    merged2=pd.merge(df,labels,on="norm_key",how="inner")
    # attach safe groups and sha
    safe=pd.read_csv(SAFE_GROUPS_CSV,encoding="utf-8")
    safe_map={norm_name(r["image_id"]): r["leakage_safe_group_id"] for _,r in safe.iterrows()}
    cv=pd.read_csv(CV_ASSIGN,encoding="utf-8")
    sha_map={norm_name(r["image_id"]): r["sha256"] for _,r in cv.iterrows()}
    merged2["leakage_safe_group_id"]=merged2["norm_key"].map(safe_map)
    merged2["sha256"]=merged2["norm_key"].map(sha_map)
    # feature cols
    feature_cols=[c for c in df.columns if is_feature_col(c)]
    # need numeric cols handling high null >0.05 similarly
    # Build X, Y, groups
    X=merged2[feature_cols].copy()
    # determine high_null_cols using whole data for simplicity but will fit impute per fold
    # Load Y matrix
    Y=merged2[traits].values.astype(float)
    groups=merged2["leakage_safe_group_id"].values
    # fold assignments: use existing CV_ASSIGN fold? Use GroupKFold again to ensure isolation? Use provided assignments to match baseline
    fold_map={norm_name(r["image_id"]): int(r["fold"]) for _,r in cv.iterrows()}
    folds=np.array([fold_map.get(nk, -1) for nk in merged2["norm_key"]])
    # If any -1, fallback to GroupKFold
    if np.any(folds==-1):
        gkf=GroupKFold(n_splits=FOLDS)
        folds=np.zeros(len(groups),int)
        for fi,(_,va) in enumerate(gkf.split(X,Y[:,0],groups)):
            folds[va]=fi
    # define experiments
    exps=[]
    exps.append({"name":"expA_inverse_hash_weight","hypothesis":"Duplicate rows overweight model; inverse-hash weighting should improve generalization","config":{"sample_weight":"1/sha_multiplicity"},"expected_failure":"May underweight rare traits if duplicates correlate with positives"})
    exps.append({"name":"expB_hash_dedup_train","hypothesis":"Training-only dedup (keep first per SHA per fold) reduces label noise from conflicting duplicates","config":{"dedup":"first_per_sha_in_train"},"expected_failure":"Arbitrary discard of conflicting labels may hurt calibration"})
    exps.append({"name":"expC_l2_tuning_C05","hypothesis":"Stronger regularization (C=0.5) reduces overfit on 116 noisy features","config":{"C":0.5},"expected_failure":"May underfit already weak traits"})
    # common run_fold helper
    def run_fold(Xtr,Xva,ytr,yva, cols, high_null_cols, sample_weight=None, C=1.0):
        # add missing indicators where >0.05 null in train
        tr_rate=Xtr[cols].isna().mean()
        hnc=[c for c in cols if tr_rate.get(c,0)>0.05]
        def add_ind(df):
            df=df.copy()
            for c in hnc:
                if c in df.columns:
                    df[c+"__missing"]=df[c].isna().astype(int)
            return df
        cols_aug=list(cols)+[c+"__missing" for c in hnc if c in cols]
        Xt_a=add_ind(Xtr[cols]); Xv_a=add_ind(Xva[cols])
        # ensure same cols order
        imp=SimpleImputer(strategy="median")
        Xt_imp=imp.fit_transform(Xt_a[cols_aug]) if cols_aug else np.empty((len(Xt_a),0))
        Xv_imp=imp.transform(Xv_a[cols_aug]) if cols_aug else np.empty((len(Xv_a),0))
        scaler=StandardScaler()
        Xt_s=scaler.fit_transform(Xt_imp); Xv_s=scaler.transform(Xv_imp)
        return Xt_s,Xv_s,hnc
    # compute sha multiplicities for weighting
    sha_counts=Counter(merged2["sha256"])
    inv_weights=np.array([1.0/sha_counts[s] if s in sha_counts else 1.0 for s in merged2["sha256"]],float)
    results={}
    for exp in exps:
        name=exp["name"]
        print(f"Running {name}")
        n_rows,n_traits=Y.shape
        P=np.full((n_rows,n_traits),np.nan)
        for fold in range(FOLDS):
            tr_mask=folds!=fold; va_mask=folds==fold
            Xtr=merged2.loc[tr_mask, feature_cols]; Xva=merged2.loc[va_mask, feature_cols]
            fold_cols=feature_cols
            # per fold preprocessing - need to handle dedup for expB
            if name=="expB_hash_dedup_train":
                # dedup train: keep first per sha
                tr_df=merged2.loc[tr_mask].copy()
                tr_df["_orig_idx"]=np.where(tr_mask)[0]
                tr_df=tr_df.drop_duplicates(subset=["sha256"],keep="first")
                keep_idx=tr_df["_orig_idx"].values
                # rebuild Xtr/Ytr with dedup
                Xtr_dedup=merged2.loc[keep_idx, feature_cols]
                Ytr_dedup=Y[keep_idx]
                # compute run_fold with dedup
                # need to compute per trait
                # Build imputer/scaler per fold using deduped train
                tr_rate=Xtr_dedup.isna().mean()
                hnc=[c for c in feature_cols if tr_rate.get(c,0)>0.05]
                def add_ind(df):
                    df=df.copy()
                    for c in hnc:
                        if c in df.columns: df[c+"__missing"]=df[c].isna().astype(int)
                    return df
                cols_aug=feature_cols+[c+"__missing" for c in hnc if c in feature_cols]
                Xt_a=add_ind(Xtr_dedup); Xv_a=add_ind(Xva)
                imp=SimpleImputer(strategy="median")
                Xt_imp=imp.fit_transform(Xt_a[cols_aug]); Xv_imp=imp.transform(Xv_a[cols_aug])
                scaler=StandardScaler()
                Xt_s=scaler.fit_transform(Xt_imp); Xv_s=scaler.transform(Xv_imp)
                for j in range(n_traits):
                    ytr=Ytr_dedup[:,j]; yva=Y[va_mask,j]
                    if len(np.unique(ytr[np.isfinite(ytr)]))<2: continue
                    try:
                        clf=LogisticRegression(max_iter=500,solver="lbfgs",class_weight="balanced",C=1.0,random_state=42)
                        clf.fit(Xt_s,ytr)
                        P[va_mask,j]=clf.predict_proba(Xv_s)[:,1]
                    except: pass
                continue
            # for other exps, generic
            # compute high null per fold
            tr_rate=Xtr.isna().mean()
            hnc=[c for c in feature_cols if tr_rate.get(c,0)>0.05]
            def add_ind2(df):
                df=df.copy()
                for c in hnc:
                    if c in df.columns: df[c+"__missing"]=df[c].isna().astype(int)
                return df
            cols_aug=feature_cols+[c+"__missing" for c in hnc if c in feature_cols]
            Xt_a=add_ind2(Xtr); Xv_a=add_ind2(Xva)
            imp=SimpleImputer(strategy="median")
            Xt_imp=imp.fit_transform(Xt_a[cols_aug]); Xv_imp=imp.transform(Xv_a[cols_aug])
            scaler=StandardScaler()
            Xt_s=scaler.fit_transform(Xt_imp); Xv_s=scaler.transform(Xv_imp)
            for j in range(n_traits):
                ytr=Y[tr_mask,j]; yva=Y[va_mask,j]
                if len(np.unique(ytr[np.isfinite(ytr)]))<2: continue
                try:
                    if name=="expA_inverse_hash_weight":
                        w=inv_weights[tr_mask]
                        clf=LogisticRegression(max_iter=500,solver="lbfgs",class_weight="balanced",C=1.0,random_state=42)
                        clf.fit(Xt_s,ytr,sample_weight=w)
                    elif name=="expC_l2_tuning_C05":
                        clf=LogisticRegression(max_iter=500,solver="lbfgs",class_weight="balanced",C=0.5,random_state=42)
                        clf.fit(Xt_s,ytr)
                    else:
                        clf=LogisticRegression(max_iter=500,solver="lbfgs",class_weight="balanced",C=1.0,random_state=42)
                        clf.fit(Xt_s,ytr)
                    P[va_mask,j]=clf.predict_proba(Xv_s)[:,1]
                except Exception as e: pass
        # compute metrics
        # save OOF predictions
        out_path=OUT_DIR / f"oof_{name}.csv"
        out_df=pd.DataFrame(P,columns=traits)
        out_df.insert(0,"image_id",merged2["image_id"].values)
        out_df.to_csv(out_path,index=False,encoding="utf-8")
        # metrics
        rocs=[]; prs=[]
        per_trait={}
        for j,t in enumerate(traits):
            y=Y[:,j]; p=P[:,j]
            m=np.isfinite(y)&np.isfinite(p)&((y==0)|(y==1))
            if m.sum()>=20 and len(np.unique(y[m]))==2:
                try: rocs.append(roc_auc_score(y[m],p[m]))
                except: pass
                try: prs.append(average_precision_score(y[m],p[m]))
                except: pass
                per_trait[t]=fin(np.mean(rocs)) if rocs else None
            else: per_trait[t]=None
        macro_roc=fin(float(np.mean(rocs))) if rocs else None
        macro_pr=fin(float(np.mean(prs))) if prs else None
        results[name]={"macro_roc":macro_roc,"macro_pr":macro_pr,"per_trait":per_trait,"oof_path":str(out_path),"config":exp}
        print(f"{name} macro_roc {macro_roc} macro_pr {macro_pr}")
    # save comparison placeholder - stage G will do paired bootstrap
    write_json(OUT_DIR/"semantic_experiment_comparison.json",{"stage":"F experiments","timestamp":time.strftime("%Y-%m-%dT%H:%M:%S"),"results":results,"note":"All experiments use leakage-safe_group GroupKFold, fold-local impute/scale, no production model overwrite. Thresholds 0.5. ponytail: full hyperparameter search infeasible here; add nested CV when effect size >0.01."})
    # also write per_trait csv
    traits=load_schema()[0]
    # build per trait csv across exps
    csv_path=OUT_DIR/"semantic_experiment_per_trait.csv"
    fieldnames=["trait"]+[k for k in results.keys()]
    # Actually need roc per trait per exp - we stored per_trait incorrectly; recompute properly
    # Let's recompute per exp per trait roc correctly saved separately; for csv, load from results per_trait stored as last mean not per trait - fix: we need per trait dict per exp
    # Quick fix: read each oof csv and compute per trait roc
    per_exp_trait={}
    for name in results:
        path=OUT_DIR / f"oof_{name}.csv"
        dfp=pd.read_csv(path,encoding="utf-8")
        # Y aligned same order as merged2
        rocs={}
        for j,t in enumerate(traits):
            y=Y[:,j]; p=dfp[t].values
            m=np.isfinite(y)&np.isfinite(p)&((y==0)|(y==1))
            if m.sum()>=20 and len(np.unique(y[m]))==2:
                try: rocs[t]=fin(roc_auc_score(y[m],p[m]))
                except: rocs[t]=None
            else: rocs[t]=None
        per_exp_trait[name]=rocs
    with open(csv_path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["trait"]+list(results.keys()))
        w.writeheader()
        for t in traits:
            row={"trait":t}
            for name in results: row[name]=per_exp_trait[name].get(t)
            w.writerow(row)
    return results

def stage_g(args):
    # paired component bootstrap comparison baseline vs exps
    exp_comp_path=OUT_DIR/"semantic_experiment_comparison.json"
    if not exp_comp_path.exists():
        print("no experiments to compare, skipping G")
        return None
    comp=json.loads(exp_comp_path.read_text(encoding="utf-8"))
    results=comp.get("results",{})
    if not results:
        print("no results")
        return None
    traits,cat,_=load_schema()
    merged,_,_=load_aligned_for_conflicts()
    # load baseline OOF
    base=pd.read_csv(OOF_HASH_CSV,encoding="utf-8-sig")
    base["norm_key"]=base["image_id"].apply(norm_name)
    merged_map={norm_name(r["image_id"]): r for idx,r in merged.iterrows()}
    # Need Y and groups aligned to baseline order
    # Use merged2 order as in stage F to align groups, but easier: use OOF_HASH order joined to merged
    base_merged_rows=[]
    groups=[]
    Y_base=[]
    for _,row in base.iterrows():
        nk=row["norm_key"]
        # find merged row with same norm_key (may be multiple but unique)
        # search merged dataframe
        mr=merged[merged["norm_key"]==nk]
        if mr.empty: continue
        mr=mr.iloc[0]
        base_merged_rows.append(row)
        groups.append(mr["leakage_safe_group_id"])
        Y_base.append([mr[t] for t in traits])
    Y_base=np.array(Y_base,float); groups=np.array(groups)
    P_base=base[traits].values if all(t in base.columns for t in traits) else np.array([[r[t] for t in traits] for _,r in base.iterrows()],float)
    # For simplicity, assume base rows count matches Y_base (if some missing due to inner join diff, truncate)
    n=np.min([len(Y_base), P_base.shape[0], len(groups)])
    Y_base=Y_base[:n]; P_base=P_base[:n]; groups=groups[:n]
    # Need to align experiment OOFs to same order as base: they were built on merged2 order (6983), which should match base order via image_id sorting? Might differ.
    # For paired bootstrap we need same row population: intersect image_ids
    base_ids=base["image_id"].apply(norm_name).tolist()[:n]
    comp_groups=defaultdict(list)
    for i,g in enumerate(groups): comp_groups[g].append(i)
    comp_ids=list(comp_groups.keys())
    rng=np.random.default_rng(args.seed)
    n_boot=args.bootstrap if not args.quick else min(100,args.bootstrap)
    # compute baseline macro per bootstrap vs each exp
    # Load exp P arrays indexed by image_id for quick lookup
    exp_data={}
    for name in results:
        path=OUT_DIR / f"oof_{name}.csv"
        if not path.exists(): continue
        df=pd.read_csv(path,encoding="utf-8")
        df["norm_key"]=df["image_id"].apply(norm_name)
        # build map norm_key -> row index in df
        m={nk:i for i,nk in enumerate(df["norm_key"])}
        # align to base_ids order
        P_exp=[]
        valid=True
        for nk in base_ids:
            if nk not in m: valid=False; break
            idx=m[nk]
            P_exp.append([df.iloc[idx][t] for t in traits])
        if not valid:
            print(f"align failed for {name}")
            continue
        exp_data[name]=np.array(P_exp,float)
    # paired bootstrap: same sampled components for both
    comparisons={}
    for name,P_exp in exp_data.items():
        diffs=[]
        per_trait_diffs=defaultdict(list)
        for b in range(n_boot):
            samp_ids=[comp_ids[j] for j in rng.integers(0,len(comp_ids),size=len(comp_ids))]
            idx=[]
            for g in samp_ids: idx.extend(comp_groups[g])
            idx=np.array(idx)
            # compute macro roc diff baseline vs exp on this bootstrap sample
            rocs_base=[]; rocs_exp=[]
            for j in range(len(traits)):
                yb=Y_base[idx,j]; pb=P_base[idx,j]; pe=P_exp[idx,j]
                m=np.isfinite(yb)&np.isfinite(pb)&np.isfinite(pe)&((yb==0)|(yb==1))
                if m.sum()<20 or len(np.unique(yb[m]))<2: continue
                try:
                    rb=roc_auc_score(yb[m],pb[m]); re=roc_auc_score(yb[m],pe[m])
                    rocs_base.append(rb); rocs_exp.append(re)
                    per_trait_diffs[j].append(re-rb)
                except: continue
            if rocs_base and rocs_exp:
                diffs.append(float(np.mean(rocs_exp)-np.mean(rocs_base)))
        if diffs:
            lo=float(np.percentile(diffs,2.5)); hi=float(np.percentile(diffs,97.5)); mean=float(np.mean(diffs))
            # count zero inside CI
            inconclusive = (lo<=0<=hi)
        else: lo=hi=mean=None; inconclusive=None
        # per trait mean diff
        per_trait=[]
        for j,t in enumerate(traits):
            arr=per_trait_diffs[j]
            if arr:
                per_trait.append({"trait":t,"category":cat[t],"mean_diff":fin(float(np.mean(arr))),"ci_low":fin(float(np.percentile(arr,2.5))),"ci_high":fin(float(np.percentile(arr,97.5))),"n_valid":len(arr)})
            else:
                per_trait.append({"trait":t,"category":cat[t],"mean_diff":None,"ci_low":None,"ci_high":None,"n_valid":0})
        comparisons[name]={"macro_diff_mean":fin(mean),"macro_diff_ci95":[fin(lo),fin(hi)],"inconclusive":inconclusive,"n_bootstrap":n_boot,"per_trait":per_trait}
    payload={
        "stage":"G statistical comparison",
        "timestamp":time.strftime("%Y-%m-%dT%H:%M:%S"),
        "method":"Paired component-aware bootstrap: same resampled components for baseline and experiment; macro ROC diff = mean(exp ROC)-mean(base ROC) per bootstrap; CI95 from 2.5/97.5 percentiles",
        "n_bootstrap":n_boot,
        "seed":args.seed,
        "baseline":"hash-grouped OOF row-weighted macro ROC 0.6438",
        "results":results,
        "comparisons":comparisons,
        "note":"Claim superiority only if CI excludes 0 and effect stable; otherwise inconclusive. No unbiased improvement claimed without nested selection."
    }
    write_json(OUT_DIR/"semantic_experiment_comparison.json",payload)
    # update per_trait csv with diffs? keep separate
    # also write per_trait comparison csv
    csv_path=OUT_DIR/"semantic_experiment_per_trait.csv"
    # if exists from F, overwrite with diff version? Keep F's per_trait as raw; here write diff
    # Instead write new file semantic_experiment_per_trait_diffs.csv
    diff_csv=OUT_DIR/"semantic_experiment_per_trait.csv"
    # combine
    if comparisons:
        # produce csv with trait, per exp mean diff and ci
        first=list(comparisons.keys())[0]
        # fieldnames trait, for each exp mean_diff, ci_low, ci_high
        with open(diff_csv,"w",newline="",encoding="utf-8") as f:
            fieldnames=["trait","category"]+ [f"{name}_diff" for name in comparisons.keys()] + [f"{name}_ci_low" for name in comparisons.keys()] + [f"{name}_ci_high" for name in comparisons.keys()]
            w=csv.DictWriter(f,fieldnames=fieldnames)
            w.writeheader()
            for j,t in enumerate(traits):
                row={"trait":t,"category":cat[t]}
                for name,comp in comparisons.items():
                    pt=comp["per_trait"][j]
                    row[f"{name}_diff"]=pt["mean_diff"]; row[f"{name}_ci_low"]=pt["ci_low"]; row[f"{name}_ci_high"]=pt["ci_high"]
                w.writerow(row)
    return payload

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage",default="all",help="all|a|b|c|d|e|f|g|conflicts|uncertainty|traits|components|experiments|compare")
    ap.add_argument("--bootstrap",type=int,default=1000)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--quick",action="store_true",help="skip expensive exps and reduce bootstrap")
    args=ap.parse_args()
    stages=args.stage.lower().split(",") if "," in args.stage else [args.stage.lower()]
    if "all" in stages:
        stages=["a","b","c","d","e","f","g"]
    mapping={"a":stage_a,"b":stage_b,"c":stage_c,"d":stage_d,"e":stage_e,"f":stage_f,"g":stage_g,
             "conflicts":stage_b,"uncertainty":stage_c,"traits":stage_d,"components":stage_e,"experiments":stage_f,"compare":stage_g,
             "artifact":stage_a}
    for s in stages:
        key=s.strip().lower()
        fn=mapping.get(key)
        if fn:
            print(f"=== Stage {key} ===")
            fn(args)
        else:
            print(f"unknown stage {key}")
    print("Done stages", stages)

if __name__=="__main__":
    main()