import pathlib
out = pathlib.Path(r"c:\Users\PC\OneDrive\Desktop\My Project\Physiognomy\Source\validate_nested_semantic_calibration.py")
code = r'''
# -*- coding: utf-8 -*-
"""Phase 3.14 Nested pilot."""
import argparse, json, math, os, re, time
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
CSV_CLEAN = ROOT / "faces_full_features_clean.csv"
LABELS_CSV = ROOT / "semantic_labels.csv"
SCHEMA_JSON = ROOT / "semantic_schema.json"
FOLD_CSV = SRC / "experiments" / "hash_grouped_semantic" / "cv_fold_assignments.csv"
THRESH_JSON = SRC / "models_semantic" / "trait_thresholds.json"
OUT_DIR = SRC / "experiments" / "semantic_calibration_phase314"
SEED = 42
N_BINS = 10
CLIP = 1e-6
N_OUTER = 5
N_INNER = 4
BOOT_REPS = 300
PILOT_TRAITS = ["poised","learning_rich","intellectual_rel","favorable","adaptable","breakthroughs","warm","conservative","collaborative"]
PILOT_RATIONALE = {
    "poised": "personality low prev 0.10 strong ROC 0.738 medium-low support",
    "learning_rich": "career very low prev 0.05 strong ROC 0.734 low support 310",
    "intellectual_rel": "relationships very low prev 0.048 strong ROC 0.727 low support 296",
    "favorable": "future_destiny medium prev 0.37 weak ROC 0.520 high support - worst performer",
    "adaptable": "personality weak ROC 0.535 medium prev, high ECE",
    "breakthroughs": "future_destiny low prev 0.079 weak ROC 0.565",
    "warm": "personality high prev 0.63 strong ROC 0.696 high support - high prevalence anchor",
    "conservative": "wealth low prev 0.073 medium ROC 0.64 low support 408",
    "collaborative": "career high prev 0.44 medium ROC 0.66 high support",
}
GRID = [round(0.30+i*0.05,4) for i in range(9)]
def fin(x):
    try:
        v=float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None
def sanitize(o):
    if isinstance(o,dict): return {k:sanitize(v) for k,v in o.items()}
    if isinstance(o,(list,tuple)): return [sanitize(v) for v in o]
    if isinstance(o,(np.integer,)): return int(o)
    if isinstance(o,(np.floating,)): return fin(float(o))
    if isinstance(o,(np.bool_,)): return bool(o)
    if isinstance(o,float): return fin(o)
    return o
def write_json(p,obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(sanitize(obj), indent=2, ensure_ascii=False, allow_nan=False),encoding="utf-8")
def clip_prob(p): return np.clip(np.asarray(p,float), CLIP, 1-CLIP)
def logit(p):
    p=clip_prob(p)
    return np.log(p/(1-p))
def ece_score(y,p,n_bins=N_BINS):
    y=np.asarray(y,float); p=np.asarray(p,float)
    if len(y)==0: return None
    edges=np.linspace(0,1,n_bins+1); e=0.0
    for i in range(n_bins):
        lo,hi=edges[i],edges[i+1]
        m=(p>=lo)&(p<=hi) if i==n_bins-1 else (p>=lo)&(p<hi)
        if m.sum()==0: continue
        e+=abs(y[m].mean()-p[m].mean())*m.sum()/len(y)
    return float(e)
def safe_auc(y,p):
    if len(np.unique(y))<2: return None
    return fin(roc_auc_score(y,p))
def safe_ap(y,p):
    if len(np.unique(y))<2: return None
    return fin(average_precision_score(y,p))
def safe_brier(y,p):
    if len(y)==0: return None
    return fin(brier_score_loss(y, np.clip(p,0,1)))
def safe_logloss(y,p):
    if len(y)==0: return None
    return fin(log_loss(y, clip_prob(p), labels=[0,1]))
def calibration_slope_intercept(y,p):
    y=np.asarray(y,float)
    if len(y)<10 or len(np.unique(y))<2: return None,None
    x=logit(p).reshape(-1,1)
    try:
        clf=LogisticRegression(solver="lbfgs", max_iter=200)
        clf.fit(x,y)
    except Exception:
        return None,None
    return fin(float(clf.coef_[0,0])), fin(float(clf.intercept_[0]))
def prior_correct(p, pi):
    p=clip_prob(p); pi=float(np.clip(pi,1e-6,1-1e-6))
    lp=np.log(p/(1-p)); adj=np.log((1-pi)/pi)
    return 1/(1+np.exp(-(lp - adj)))
def fit_sigmoid(p_train,y_train):
    x=logit(p_train).reshape(-1,1)
    try:
        clf=LogisticRegression(solver="lbfgs", max_iter=500); clf.fit(x,y_train)
        return clf
    except Exception:
        return None
def apply_sigmoid(clf,p): return np.clip(clf.predict_proba(logit(p).reshape(-1,1))[:,1],0,1)
def group_support(y, groups):
    uniq=np.unique(groups)
    pos=0; neg=0
    for g in uniq:
        yy=y[groups==g]
        if yy.max()==1 and yy.min()==0: pos+=1; neg+=1
        elif yy.max()==1: pos+=1
        else: neg+=1
    return pos,neg
def is_feature_col(name): return name.endswith("_ratio") or "area_ratio" in name or "asymmetry" in name
def norm_name(v):
    s=os.path.basename(str(v).strip()); s=re.sub(r"\s+"," ",s); s=re.sub(r"\)\s+\("," )(",s); return s.lower()
def stem_name(v): return re.sub(r"\.[^.]+$","",norm_name(v))
def load_thresholds():
    if not THRESH_JSON.exists(): return {},0.5,"none"
    j=json.loads(THRESH_JSON.read_text(encoding="utf-8"))
    tmap=j.get("thresholds") or j.get("trait_thresholds") or {}
    strat=j.get("strategy","per_trait"); dflt=j.get("default_threshold",0.5)
    return tmap,dflt,strat
def threshold_for(trait, tmap, dflt): return float(tmap.get(trait, dflt))
pilot_traits_global=[]
def main():
    global pilot_traits_global
    ap=argparse.ArgumentParser()
    ap.add_argument("--outer-folds", type=int, default=N_OUTER)
    ap.add_argument("--inner-folds", type=int, default=N_INNER)
    ap.add_argument("--bootstrap", type=int, default=BOOT_REPS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--traits", nargs="*", default=PILOT_TRAITS)
    ap.add_argument("--quick", action="store_true")
    args=ap.parse_args()
    traits=args.traits
    pilot_traits_global=traits
    acceptance_config={"min_brier_drop":0.01,"max_roc_drop":0.005,"logloss_non_increasing":True,"ci_excludes_zero":True,"threshold_valid_required":True}
    t0=time.time()
    schema=json.loads(SCHEMA_JSON.read_text(encoding="utf-8"))
    cat_map={t["name"]:t["category"] for t in schema["traits"]}
    folds_df=pd.read_csv(FOLD_CSV, encoding="utf-8")
    df=pd.read_csv(CSV_CLEAN, encoding="utf-8-sig")
    feature_cols=[c for c in df.columns if c=="image_id" or is_feature_col(c)]
    df_slim=df[feature_cols].copy()
    df_slim["norm_key"]=df_slim["image_id"].apply(norm_name)
    df_slim["norm_stem"]=df_slim["image_id"].apply(stem_name)
    labels=pd.read_csv(LABELS_CSV, encoding="utf-8-sig")
    img_col=[c for c in labels.columns if c.strip().lower().lstrip("\ufeff")=="image"][0]
    if img_col!="image": labels=labels.rename(columns={img_col:"image"})
    labels["norm_key"]=labels["image"].apply(norm_name)
    labels["norm_stem"]=labels["image"].apply(stem_name)
    merged=pd.merge(df_slim, labels[["norm_key"]+traits], on="norm_key", how="inner")
    fmap=dict(zip(folds_df["norm_key"].astype(str), folds_df["fold"].astype(int)))
    gmap=dict(zip(folds_df["norm_key"].astype(str), folds_df["safe_group"].astype(int)))
    smap=dict(zip(folds_df["norm_key"].astype(str), folds_df["sha256"].astype(str)))
    merged["fold"]=merged["norm_key"].map(fmap)
    merged["safe_group"]=merged["norm_key"].map(gmap)
    merged["sha256"]=merged["norm_key"].map(smap)
    problems=[]
    if len(merged)!=6983: problems.append(f"row {len(merged)} !=6983")
    if merged["fold"].isna().any(): problems.append("missing fold")
    merged=merged.dropna(subset=["fold","safe_group"])
    merged["fold"]=merged["fold"].astype(int)
    merged["safe_group"]=merged["safe_group"].astype(int)
    n=len(merged)
    groups=merged["safe_group"].to_numpy()
    folds=merged["fold"].to_numpy()
    fold_integrity=[]
    leakage_ok=True
    for k in range(args.outer_folds):
        tr_groups=set(groups[folds!=k]); va_groups=set(groups[folds==k])
        tr_sha=set(merged.loc[folds!=k,"sha256"]); va_sha=set(merged.loc[folds==k,"sha256"])
        tr_stem=set(merged.loc[folds!=k,"norm_stem"]); va_stem=set(merged.loc[folds==k,"norm_stem"])
        overlap_g=len(tr_groups & va_groups); overlap_sha=len(tr_sha & va_sha); overlap_stem=len(tr_stem & va_stem)
        ok= overlap_g==0 and overlap_sha==0 and overlap_stem==0
        leakage_ok &= ok
        fold_integrity.append({"fold":int(k),"n_train":int((folds!=k).sum()),"n_val":int((folds==k).sum()),"group_overlap":overlap_g,"sha_overlap":overlap_sha,"stem_overlap":overlap_stem,"ok":ok})
    P_raw=np.full((n,len(traits)), np.nan)
    P_sig=np.full((n,len(traits)), np.nan)
    P_prior=np.full((n,len(traits)), np.nan)
    Y=np.column_stack([merged[t].to_numpy(float) for t in traits])
    old_tmap, old_dflt, old_strat = load_thresholds()
    thresholds_C_per_fold=[None]*args.outer_folds
    per_fold_metrics=[]
    for k in range(args.outer_folds):
        tr_mask=folds!=k; va_mask=folds==k
        X_tr_raw=merged.loc[tr_mask, feature_cols].copy()
        numeric_cols=[c for c in X_tr_raw.columns if c!="image_id" and is_feature_col(c)]
        tr_rate=X_tr_raw[numeric_cols].isna().mean()
        high_null_cols=[c for c in numeric_cols if tr_rate.get(c,0)>0.05]
        def add_ind(df_):
            df2=df_.copy()
            for c in high_null_cols:
                if c in df2.columns: df2[c+"__missing"]=df2[c].isna().astype(int)
            return df2
        Xt_add=add_ind(X_tr_raw[numeric_cols])
        cols=list(Xt_add.columns)
        imp=SimpleImputer(strategy="median")
        Xt_imp=imp.fit_transform(Xt_add[cols])
        scaler=StandardScaler()
        Xt_s=scaler.fit_transform(Xt_imp)
        X_va_raw=merged.loc[va_mask, numeric_cols].copy()
        Xv_add=add_ind(X_va_raw)
        for c in cols:
            if c not in Xv_add.columns: Xv_add[c]=0
        Xv_add=Xv_add[cols]
        Xv_imp=imp.transform(Xv_add)
        Xv_s=scaler.transform(Xv_imp)
        outer_train_idx=np.where(tr_mask)[0]
        inner_groups=groups[outer_train_idx]
        inner_oof_raw=np.full((len(outer_train_idx), len(traits)), np.nan)
        gkf_inner=GroupKFold(n_splits=args.inner_folds)
        for t_idx, trait in enumerate(traits):
            y_tr_outer=Y[outer_train_idx, t_idx]
            for inner_train_pos, inner_val_pos in gkf_inner.split(Xt_s, y_tr_outer, groups=inner_groups):
                X_inner_tr=Xt_s[inner_train_pos]
                y_inner_tr=y_tr_outer[inner_train_pos]
                if len(np.unique(y_inner_tr))<2:
                    p_val=np.full(len(inner_val_pos), float(np.clip(y_inner_tr.mean() if len(y_inner_tr) else 0.5,0,1)))
                else:
                    clf=LogisticRegression(max_iter=500, solver="lbfgs", class_weight="balanced")
                    clf.fit(X_inner_tr, y_inner_tr)
                    p_val=clf.predict_proba(Xt_s[inner_val_pos])[:,1]
                inner_oof_raw[inner_val_pos, t_idx]=p_val
        sigmoid_models={}
        prior_pis={}
        for t_idx, trait in enumerate(traits):
            y_tr_outer=Y[outer_train_idx, t_idx]
            p_inner=inner_oof_raw[:, t_idx]
            mask=np.isfinite(p_inner)
            pos_g,neg_g=group_support(y_tr_outer[mask], inner_groups[mask]) if mask.sum()>0 else (0,0)
            usable= pos_g>=2 and neg_g>=2 and mask.sum()>=10 and len(np.unique(y_tr_outer[mask]))==2
            if usable:
                m=fit_sigmoid(p_inner[mask], y_tr_outer[mask])
                sigmoid_models[trait]=m
            else:
                sigmoid_models[trait]=None
            pi=float(y_tr_outer.mean()) if len(y_tr_outer) else 0.5
            prior_pis[trait]=float(np.clip(pi,1e-6,1-1e-6))
        thr_C={}
        for t_idx, trait in enumerate(traits):
            y_tr_outer=Y[outer_train_idx, t_idx]
            p_inner=inner_oof_raw[:, t_idx]
            sig_m=sigmoid_models[trait]
            if sig_m is not None:
                p_cal_inner=apply_sigmoid(sig_m, p_inner)
            else:
                p_cal_inner=prior_correct(p_inner, prior_pis[trait])
            mask=np.isfinite(p_cal_inner)
            y_use=y_tr_outer[mask]; p_use=p_cal_inner[mask]
            groups_use=inner_groups[mask]
            pos_g,neg_g=group_support(y_use, groups_use) if len(y_use) else (0,0)
            if pos_g<5 or neg_g<5 or len(np.unique(y_use))<2 or len(y_use)<20:
                thr_C[trait]=threshold_for(trait, old_tmap, old_dflt)
                continue
            best_f1=-1; best_thr=0.5
            for thr in GRID:
                pred=(p_use>=thr).astype(int)
                f1=f1_score(y_use, pred, zero_division=0)
                if f1>best_f1:
                    best_f1=f1; best_thr=thr
            thr_C[trait]=float(best_thr)
        thresholds_C_per_fold[k]=thr_C
        for t_idx, trait in enumerate(traits):
            y_tr_outer=Y[outer_train_idx, t_idx]
            if len(np.unique(y_tr_outer))<2:
                p_va_raw=np.full(int(va_mask.sum()), float(y_tr_outer.mean()))
            else:
                clf_final=LogisticRegression(max_iter=500, solver="lbfgs", class_weight="balanced")
                clf_final.fit(Xt_s, y_tr_outer)
                p_va_raw=clf_final.predict_proba(Xv_s)[:,1]
            va_idx=np.where(va_mask)[0]
            P_raw[va_idx, t_idx]=p_va_raw
            sig_m=sigmoid_models[trait]
            if sig_m is not None:
                p_sig=apply_sigmoid(sig_m, p_va_raw)
            else:
                p_sig=p_va_raw
            P_sig[va_idx, t_idx]=p_sig
            P_prior[va_idx, t_idx]=prior_correct(p_va_raw, prior_pis[trait])
        per_fold_metrics.append({"fold":int(k),"n_train":int(tr_mask.sum()),"n_val":int(va_mask.sum()),"high_null_cols":int(len(high_null_cols))})
    P_raw=np.clip(P_raw,0,1); P_sig=np.clip(P_sig,0,1); P_prior=np.clip(P_prior,0,1)
    write_json(OUT_DIR / "nested_fold_integrity.json", {"n_rows":int(n),"n_traits":len(traits),"traits":traits,"folds":fold_integrity,"leakage_ok":leakage_ok,"per_fold":per_fold_metrics,"seed":args.seed,"outer_folds":args.outer_folds,"inner_folds":args.inner_folds})
    try:
        oof_df=pd.DataFrame(P_raw, columns=traits)
        oof_df.insert(0,"image_id", merged["image_id"].astype(str).values)
        oof_df["fold"]=folds; oof_df["component_group_id"]=groups
        oof_df.to_csv(OUT_DIR / "nested_oof_raw.csv", index=False, encoding="utf-8")
        pd.DataFrame(P_sig, columns=traits).to_csv(OUT_DIR / "nested_oof_sigmoid.csv", index=False)
        pd.DataFrame(P_prior, columns=traits).to_csv(OUT_DIR / "nested_oof_prior.csv", index=False)
    except Exception:
        pass
    def evaluate_block(P):
        per_trait=[]
        for t_idx, trait in enumerate(traits):
            y=Y[:,t_idx]; p=P[:,t_idx]
            slope,intercept=calibration_slope_intercept(y,p)
            per_trait.append({"trait":trait,"category":cat_map[trait],"prevalence":float(y.mean()),"roc_auc":safe_auc(y,p),"pr_auc":safe_ap(y,p),"brier":safe_brier(y,p),"ece":ece_score(y,p),"log_loss":safe_logloss(y,p),"slope":slope,"intercept":intercept,"n":int(len(y))})
        macro={"roc_auc":float(np.mean([r["roc_auc"] for r in per_trait if r["roc_auc"] is not None])) if any(r["roc_auc"] is not None for r in per_trait) else None,
               "pr_auc":float(np.mean([r["pr_auc"] for r in per_trait if r["pr_auc"] is not None])) if any(r["pr_auc"] is not None for r in per_trait) else None,
               "brier":float(np.mean([r["brier"] for r in per_trait if r["brier"] is not None])),
               "ece":float(np.mean([r["ece"] for r in per_trait if r["ece"] is not None])),
               "log_loss":float(np.mean([r["log_loss"] for r in per_trait if r["log_loss"] is not None])),
               }
        return per_trait, macro
    per_raw, macro_raw = evaluate_block(P_raw)
    per_sig, macro_sig = evaluate_block(P_sig)
    per_prior, macro_prior = evaluate_block(P_prior)
    def threshold_metrics(P, thr_map_or_scalar):
        P_pred=np.zeros_like(P, dtype=int)
        for t_idx, trait in enumerate(traits):
            pcol=P[:,t_idx]
            if isinstance(thr_map_or_scalar, dict):
                thr=thr_map_or_scalar[trait] if trait in thr_map_or_scalar else 0.5
                P_pred[:,t_idx]=(pcol>=thr).astype(int)
            elif isinstance(thr_map_or_scalar, (int,float)):
                P_pred[:,t_idx]=(pcol>=thr_map_or_scalar).astype(int)
            else:
                for k in range(args.outer_folds):
                    mask=folds==k
                    thr=thr_map_or_scalar[k][trait] if isinstance(thr_map_or_scalar, list) else 0.5
                    P_pred[mask, t_idx]=(pcol[mask]>=thr).astype(int)
        rows=[]
        for t_idx, trait in enumerate(traits):
            y=Y[:,t_idx]; pred=P_pred[:,t_idx]
            prec=precision_score(y,pred,zero_division=0); rec=recall_score(y,pred,zero_division=0); f1=f1_score(y,pred,zero_division=0)
            rows.append({"trait":trait,"category":cat_map[trait],"precision":float(prec),"recall":float(rec),"f1":float(f1),"pos_rate":float(pred.mean())})
        macro_f1=float(np.mean([r["f1"] for r in rows]))
        macro_prec=float(np.mean([r["precision"] for r in rows]))
        macro_rec=float(np.mean([r["recall"] for r in rows]))
        pos_rate=float(P_pred.mean())
        active_per_sample=float(P_pred.sum(axis=1).mean())
        cat_active={}
        for cat in set(cat_map.values()):
            idx=[i for i,t in enumerate(traits) if cat_map[t]==cat]
            cat_active[cat]=float(P_pred[:,idx].sum()) if idx else 0.0
        return {"rows":rows,"macro_f1":macro_f1,"macro_precision":macro_prec,"macro_recall":macro_rec,"pos_rate":pos_rate,"active_per_sample":active_per_sample,"cat_active":cat_active,"P_pred":P_pred}
    thr_old=dict(old_tmap) if old_tmap else {}
    thr_old_full={t: threshold_for(t, thr_old, old_dflt) for t in traits}
    metrics_A=threshold_metrics(P_raw, thr_old_full)
    metrics_B_raw_old=threshold_metrics(P_sig, thr_old_full)
    metrics_B_prior_old=threshold_metrics(P_prior, thr_old_full)
    metrics_C_sig=threshold_metrics(P_sig, thresholds_C_per_fold)
    metrics_C_prior=threshold_metrics(P_prior, thresholds_C_per_fold)
    thr_report={"policy_A_raw_old":{"thresholds":thr_old_full,"macro_f1":metrics_A["macro_f1"],"active_per_sample":metrics_A["active_per_sample"],"pos_rate":metrics_A["pos_rate"]},
                "policy_B_sig_old":{"thresholds":thr_old_full,"macro_f1":metrics_B_raw_old["macro_f1"],"active_per_sample":metrics_B_raw_old["active_per_sample"],"pos_rate":metrics_B_raw_old["pos_rate"]},
                "policy_B_prior_old":{"thresholds":thr_old_full,"macro_f1":metrics_B_prior_old["macro_f1"],"active_per_sample":metrics_B_prior_old["active_per_sample"],"pos_rate":metrics_B_prior_old["pos_rate"]},
                "policy_C_sig_recal":{"thresholds_per_fold":thresholds_C_per_fold,"macro_f1":metrics_C_sig["macro_f1"],"active_per_sample":metrics_C_sig["active_per_sample"],"pos_rate":metrics_C_sig["pos_rate"]},
                "policy_C_prior_recal":{"thresholds_per_fold":thresholds_C_per_fold,"macro_f1":metrics_C_prior["macro_f1"],"active_per_sample":metrics_C_prior["active_per_sample"],"pos_rate":metrics_C_prior["pos_rate"]},
                "grid":GRID, "threshold_stability": {t: [thresholds_C_per_fold[k][t] for k in range(args.outer_folds)] for t in traits},
                "effective_sample_note": "thresholds derived from inner OOF calibrated probs within outer-train only; insufficient support fell back to old threshold"
               }
    try:
        import generate_text_template as G
    except Exception:
        G=None
    def downstream_stats(P, thr_policy):
        n=len(P)
        active_counts=[]; cat_counts={c:[] for c in set(cat_map.values())}; lengths=[]
        empty_sections=0
        for i in range(n):
            probs={t: float(P[i,ti]) for ti,t in enumerate(traits)}
            if isinstance(thr_policy, list):
                k=int(folds[i]); thr_map=thr_policy[k]
            else:
                thr_map=thr_policy
            thrs={t: float(thr_map[t]) for t in traits}
            active=[t for t in traits if probs[t]>=thrs[t]]
            active_counts.append(len(active))
            for cat in set(cat_map.values()):
                idx=[t for t in traits if cat_map[t]==cat]
                cat_active_any=any(t in active for t in idx)
                cat_counts[cat].append(1 if cat_active_any else 0)
            if G is not None:
                try:
                    text, act = G.generate_text(probs, thresholds=thr_map if len(thr_map)==45 else None, threshold=0.5)
                    lengths.append(len(text))
                except Exception:
                    lengths.append(len(active)*20)
            else:
                lengths.append(len(active)*20)
            if len(active)==0:
                empty_sections+=1
        return {"mean_active_traits": float(np.mean(active_counts)) if active_counts else None,
                "cat_active_rows": {c:int(np.sum(cat_counts[c])) for c in cat_counts},
                "empty_proxy": int(empty_sections),
                "mean_length": float(np.mean(lengths)) if lengths else None}
    downstream={
        "policy_A": downstream_stats(P_raw, thr_old_full),
        "policy_B_sig": downstream_stats(P_sig, thr_old_full),
        "policy_C_sig": downstream_stats(P_sig, thresholds_C_per_fold),
        "policy_B_prior": downstream_stats(P_prior, thr_old_full),
        "policy_C_prior": downstream_stats(P_prior, thresholds_C_per_fold),
        "note": "pilot traits only; 45-trait downstream would differ."
    }
    reps=args.bootstrap if not args.quick else 30
    rng=np.random.RandomState(args.seed)
    uniq_groups=np.unique(groups)
    deltas={"sigmoid": {"brier":[],"ece":[],"roc_auc":[],"log_loss":[],"f1_C":[]},"prior":{"brier":[],"ece":[],"roc_auc":[],"log_loss":[],"f1_C":[]}}
    for r in range(reps):
        chosen=rng.choice(uniq_groups, size=len(uniq_groups), replace=True)
        idx=[]
        for g in chosen:
            idx.extend(np.where(groups==g)[0].tolist())
        idx=np.array(idx, dtype=int)
        if len(idx)==0:
            continue
        Yb=Y[idx]; Prb=P_raw[idx]; Psb=P_sig[idx]; Ppb=P_prior[idx]
        def macro_for(Pb):
            briers=[]; eces=[]; rocs=[]; lls=[]
            for ti in range(len(traits)):
                y=Yb[:,ti]; p=Pb[:,ti]
                b=safe_brier(y,p); e=ece_score(y,p); a=safe_auc(y,p); ll=safe_logloss(y,p)
                if b is not None: briers.append(b)
                if e is not None: eces.append(e)
                if a is not None: rocs.append(a)
                if ll is not None: lls.append(ll)
            return (float(np.mean(briers)) if briers else None, float(np.mean(eces)) if eces else None, float(np.mean(rocs)) if rocs else None, float(np.mean(lls)) if lls else None)
        br,er,rr,lr=macro_for(Prb); bs,es,rs,ls=macro_for(Psb); bp,ep,rp,lp=macro_for(Ppb)
        def f1_for_policy(Pb, thr_policy):
            Ppred=np.zeros_like(Pb, dtype=int)
            for ti,t in enumerate(traits):
                pcol=Pb[:,ti]
                if isinstance(thr_policy, list):
                    for k in range(args.outer_folds):
                        mask_local = folds[idx]==k
                        thr=thr_policy[k][t]
                        Ppred[mask_local, ti]=(pcol[mask_local]>=thr).astype(int)
                else:
                    thr=thr_policy[t]
                    Ppred[:,ti]=(pcol>=thr).astype(int)
            f1s=[]
            for ti in range(len(traits)):
                y=Yb[:,ti]; pred=Ppred[:,ti]
                f1s.append(f1_score(y,pred,zero_division=0))
            return float(np.mean(f1s)) if f1s else None
        fA=f1_for_policy(Prb, thr_old_full); fCs=f1_for_policy(Psb, thresholds_C_per_fold); fCp=f1_for_policy(Ppb, thresholds_C_per_fold)
        if bs is not None and br is not None: deltas["sigmoid"]["brier"].append(bs-br)
        if es is not None and er is not None: deltas["sigmoid"]["ece"].append(es-er)
        if rs is not None and rr is not None: deltas["sigmoid"]["roc_auc"].append(rs-rr)
        if ls is not None and lr is not None: deltas["sigmoid"]["log_loss"].append(ls-lr)
        if fCs is not None and fA is not None: deltas["sigmoid"]["f1_C"].append(fCs-fA)
        if bp is not None and br is not None: deltas["prior"]["brier"].append(bp-br)
        if ep is not None and er is not None: deltas["prior"]["ece"].append(ep-er)
        if rp is not None and rr is not None: deltas["prior"]["roc_auc"].append(rp-rr)
        if lp is not None and lr is not None: deltas["prior"]["log_loss"].append(lp-lr)
        if fCp is not None and fA is not None: deltas["prior"]["f1_C"].append(fCp-fA)
    def ci(arr):
        if len(arr)<10: return [None,None]
        lo=float(np.percentile(arr,2.5)); hi=float(np.percentile(arr,97.5))
        return [lo,hi]
    bootstrap_report={"reps":int(reps),"seed":int(args.seed),"unit":"connected component, paired, no refit, pilot traits only",
                      "deltas":{"sigmoid":{"brier":{"mean":float(np.mean(deltas['sigmoid']['brier'])) if deltas['sigmoid']['brier'] else None,"ci95":ci(deltas['sigmoid']['brier']),"frac_negative":float(np.mean([x<0 for x in deltas['sigmoid']['brier']])) if deltas['sigmoid']['brier'] else None},
                                          "ece":{"mean":float(np.mean(deltas['sigmoid']['ece'])) if deltas['sigmoid']['ece'] else None,"ci95":ci(deltas['sigmoid']['ece'])},
                                          "roc_auc":{"mean":float(np.mean(deltas['sigmoid']['roc_auc'])) if deltas['sigmoid']['roc_auc'] else None,"ci95":ci(deltas['sigmoid']['roc_auc'])},
                                          "log_loss":{"mean":float(np.mean(deltas['sigmoid']['log_loss'])) if deltas['sigmoid']['log_loss'] else None,"ci95":ci(deltas['sigmoid']['log_loss'])},
                                          "f1_C":{"mean":float(np.mean(deltas['sigmoid']['f1_C'])) if deltas['sigmoid']['f1_C'] else None,"ci95":ci(deltas['sigmoid']['f1_C'])}},
                                "prior":{"brier":{"mean":float(np.mean(deltas['prior']['brier'])) if deltas['prior']['brier'] else None,"ci95":ci(deltas['prior']['brier'])},
                                          "ece":{"mean":float(np.mean(deltas['prior']['ece'])) if deltas['prior']['ece'] else None,"ci95":ci(deltas['prior']['ece'])},
                                          "roc_auc":{"mean":float(np.mean(deltas['prior']['roc_auc'])) if deltas['prior']['roc_auc'] else None,"ci95":ci(deltas['prior']['roc_auc'])},
                                          "log_loss":{"mean":float(np.mean(deltas['prior']['log_loss'])) if deltas['prior']['log_loss'] else None,"ci95":ci(deltas['prior']['log_loss'])},
                                          "f1_C":{"mean":float(np.mean(deltas['prior']['f1_C'])) if deltas['prior']['f1_C'] else None,"ci95":ci(deltas['prior']['f1_C'])}}
                      }}
    brier_drop_sig = macro_raw["brier"] - macro_sig["brier"] if macro_raw["brier"] is not None and macro_sig["brier"] is not None else None
    roc_drop_sig = macro_raw["roc_auc"] - macro_sig["roc_auc"] if macro_raw["roc_auc"] is not None and macro_sig["roc_auc"] is not None else None
    logloss_improved = macro_sig["log_loss"] <= macro_raw["log_loss"] if macro_sig["log_loss"] is not None and macro_raw["log_loss"] is not None else None
    brier_ci=bootstrap_report["deltas"]["sigmoid"]["brier"]["ci95"]
    roc_ci=bootstrap_report["deltas"]["sigmoid"]["roc_auc"]["ci95"]
    decision={
        "sigmoid_independently_supported": bool(brier_drop_sig is not None and brier_drop_sig>=0.01 and (roc_drop_sig is not None and (macro_raw["roc_auc"]-macro_sig["roc_auc"])<0.005 and logloss_improved)),
        "threshold_compatibility_established": bool(metrics_C_sig["macro_f1"]>= metrics_A["macro_f1"]-0.02),
        "ready_for_wider_45_trait_eval": False,
        "ready_for_production_promotion": False,
        "reasons": ["pilot traits only, nested validation required for full 45","threshold stability limited, low support traits fallback","no promotion without full bootstrap CI and downstream check"],
        "acceptance": acceptance_config,
        "observed": {"brier_drop_sig": brier_drop_sig, "roc_drop_sig": roc_drop_sig, "logloss_improved": logloss_improved, "brier_ci": brier_ci, "roc_ci": roc_ci}
    }
    write_json(OUT_DIR / "nested_calibration_config.json", {"phase":"3.14","pilot_traits":traits,"rationale":PILOT_RATIONALE,"outer_folds":args.outer_folds,"inner_folds":args.inner_folds,"seed":args.seed,"bootstrap_reps":reps,"acceptance":acceptance_config,"label_semantics":"agreement with provider-derived weak labels only","grid":GRID})
    write_json(OUT_DIR / "nested_calibration_metrics.json", {"per_trait_raw": per_raw, "macro_raw": macro_raw, "per_trait_sigmoid": per_sig, "macro_sigmoid": macro_sig, "per_trait_prior": per_prior, "macro_prior": macro_prior, "threshold_comparison": thr_report, "downstream": downstream})
    rows=[]
    for a,b,c in zip(per_raw, per_sig, per_prior):
        rows.append({"trait":a["trait"],"category":a["category"],"prevalence":a["prevalence"],"roc_raw":a["roc_auc"],"roc_sig":b["roc_auc"],"roc_prior":c["roc_auc"],"brier_raw":a["brier"],"brier_sig":b["brier"],"brier_prior":c["brier"],"ece_raw":a["ece"],"ece_sig":b["ece"],"logloss_raw":a["log_loss"],"logloss_sig":b["log_loss"],"slope_raw":a["slope"],"intercept_raw":a["intercept"]})
    pd.DataFrame(rows).to_csv(OUT_DIR / "nested_calibration_per_trait.csv", index=False, encoding="utf-8")
    write_json(OUT_DIR / "nested_threshold_comparison.json", thr_report)
    write_json(OUT_DIR / "nested_downstream_impact.json", downstream)
    write_json(OUT_DIR / "nested_bootstrap_report.json", bootstrap_report)
    write_json(OUT_DIR / "nested_calibration_decision.json", decision)
    elapsed=time.time()-t0
    print(json.dumps(sanitize({"elapsed_sec":fin(elapsed),"leakage_ok":leakage_ok,"macro_raw":macro_raw,"macro_sig":macro_sig,"macro_prior":macro_prior,"thr_A_f1":metrics_A["macro_f1"],"thr_C_sig_f1":metrics_C_sig["macro_f1"],"decision":decision}), indent=2))
if __name__=="__main__":
    main()
'''
out.write_text(code, encoding="utf-8")
print(f"Wrote {out} {len(code)} chars")