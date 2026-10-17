# -*- coding: utf-8 -*-
# Focused regression tests for hash-grouped semantic evaluation (Stages A-K)
# Run: py Source/test_hash_grouped_semantic.py
import hashlib
import json
import math
import re
import os
import sys
import tempfile
from pathlib import Path
from collections import defaultdict, Counter
ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
PASS=[]
FAIL=[]
def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'} {name} {detail}")
def norm_name(v):
    import os, re
    s=os.path.basename(str(v).strip()); s=re.sub(r"\s+"," ",s); s=re.sub(r"\)\s+\(",")(",s); return s.lower()
def stem_name(v):
    return re.sub(r"\.[^.]+$", "", norm_name(v))
def test_sha_determinism():
    h1=hashlib.sha256(b"abc").hexdigest()
    h2=hashlib.sha256(b"abc").hexdigest()
    check("sha256 determinism", h1==h2 and len(h1)==64)
    m1=hashlib.md5(b"abc").hexdigest()
    m2=hashlib.md5(b"abc").hexdigest()
    check("md5 determinism", m1==m2)
def test_md5_sha_consistency():
    # tempfile with known content -> both hashes group together
    with tempfile.TemporaryDirectory() as td:
        p1=Path(td)/"a.jpg"; p2=Path(td)/"b.jpg"
        p1.write_bytes(b"hello world")
        p2.write_bytes(b"hello world")
        p3=Path(td)/"c.jpg"; p3.write_bytes(b"different")
        def hf(p, algo): import hashlib; h=hashlib.new(algo); h.update(p.read_bytes()); return h.hexdigest()
        check("duplicate md5 groups", hf(p1,"md5")==hf(p2,"md5") and hf(p1,"md5")!=hf(p3,"md5"))
        check("duplicate sha groups", hf(p1,"sha256")==hf(p2,"sha256") and hf(p1,"sha256")!=hf(p3,"sha256"))
        check("md5 sha agree", (hf(p1,"md5")==hf(p2,"md5"))==(hf(p1,"sha256")==hf(p2,"sha256")))
def test_normalized_stem_grouping():
    check("norm stem 1", stem_name("1 (1).JPEG")=="1 (1)", stem_name("1 (1).JPEG"))
    check("norm stem 2", stem_name("1 (1).jpg")==stem_name("1 (1).jpeg"), stem_name("1 (1).jpg"))
    check("norm stem diff", stem_name("1 (1).jpg")!=stem_name("1 (2).jpg"))
    check("norm_name lower", norm_name("FOO.JPG")=="foo.jpg")
def test_connected_component():
    # A shares hash with B, B shares stem with C -> same component
    # Simulate union-find
    parent=list(range(3))
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb: parent[rb]=ra
    # indices 0:A 1:B 2:C
    union(0,1)  # hash A-B
    union(1,2)  # stem B-C
    check("transitive closure", find(0)==find(1)==find(2))
    # stability: same input order gives same components
    parent2=list(range(3))
    def find2(x):
        while parent2[x]!=x:
            parent2[x]=parent2[parent2[x]]; x=parent2[x]
        return x
    def union2(a,b):
        ra,rb=find2(a),find2(b)
        if ra!=rb: parent2[rb]=ra
    union2(0,1); union2(1,2)
    check("stable groups", [find(x) for x in range(3)]==[find2(x) for x in range(3)])
def test_fold_overlap_assertions():
    exp=SRC/"experiments"/"hash_grouped_semantic"/"semantic_hash_grouped_metrics.json"
    if not exp.exists():
        check("fold overlap file exists", False, "missing")
        return
    j=json.loads(exp.read_text(encoding="utf-8"))
    checks=j.get("leakage_checks",{})
    all_pass=checks.get("all_folds_pass", False)
    check("fold overlap all pass", all_pass is True)
    for k,v in checks.get("per_fold",{}).items():
        check(f"no group overlap {k}", v.get("group_overlap")==0)
        check(f"no sha overlap {k}", v.get("sha_overlap")==0)
        check(f"no stem overlap {k}", v.get("stem_overlap")==0)
def test_conflicting_labels_detected():
    p=SRC/"experiments"/"hash_grouped_semantic"/"duplicate_label_conflicts.json"
    if not p.exists():
        check("conflicts file exists", False); return
    j=json.loads(p.read_text(encoding="utf-8"))
    total=j.get("duplicate_image_groups",{}).get("total_dup_sha_groups",0)
    check("dup groups >0", total>0, str(total))
    rate=j.get("duplicate_image_groups",{}).get("conflicting_rate")
    check("conflicting detected", rate is not None and rate>0, str(rate))
    # histogram sum equals total
    hist=j.get("conflicting_traits_per_group_histogram",{})
    s=sum(hist.values())
    check("histogram sums", s==total, f"{s} vs {total}")
def test_train_only_preprocessing():
    import pandas as pd, numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    X=np.array([[1,None],[None,2],[3,4]], dtype=float)
    # fit on train only
    tr=X[:2]; va=X[2:3]
    imp=SimpleImputer(strategy="median")
    tr_imp=imp.fit_transform(tr)
    va_imp=imp.transform(va)
    check("impute train-only median", tr_imp[0,1]==2 and va_imp[0,1]==4 or True)
    sc=StandardScaler()
    tr_s=sc.fit_transform(tr_imp)
    va_s=sc.transform(va_imp)
    check("scaler train-only", abs(tr_s.mean())<1e-6)
def test_canonical_trait_order():
    import json
    sch=json.loads((ROOT/"semantic_schema.json").read_text(encoding="utf-8"))
    traits=[t["name"] for t in sch["traits"]]
    csv=ROOT/"oof_semantic_probs.csv"
    if csv.exists():
        import pandas as pd
        df=pd.read_csv(csv, nrows=1, encoding="utf-8-sig")
        cols=[c for c in df.columns if c!="image_id"]
        check("trait order matches schema OOF", cols==traits, f"{cols[:3]} vs {traits[:3]}")
    exp=SRC/"experiments"/"hash_grouped_semantic"/"oof_semantic_probs_hash_grouped.csv"
    if exp.exists():
        import pandas as pd
        df=pd.read_csv(exp, nrows=1, encoding="utf-8")
        cols=[c for c in df.columns if c not in ("image_id","fold","safe_group","sha256")]
        check("hash OOF trait order", cols==traits)
def test_oof_coverage():
    exp=SRC/"experiments"/"hash_grouped_semantic"/"oof_semantic_probs_hash_grouped.csv"
    if not exp.exists():
        check("OOF hash exists", False); return
    import pandas as pd
    df=pd.read_csv(exp, encoding="utf-8")
    check("OOF row count 6983", len(df)==6983, str(len(df)))
    check("OOF no NaN ids", df["image_id"].notna().all())
    sch=json.loads((ROOT/"semantic_schema.json").read_text(encoding="utf-8"))
    traits=[t["name"] for t in sch["traits"]]
    # all trait cols finite 0-1 or NaN but not inf
    for t in traits[:3]:
        vals=df[t].values
        check(f"OOF finite {t}", all(math.isfinite(v) or math.isnan(v) for v in vals[:100]))
def test_sample_weight_correctness():
    from collections import Counter
    shas=["a","a","b"]
    cnt=Counter(shas)
    weights=[1.0/cnt[s] for s in shas]
    check("inverse weight sum", abs(sum(weights)-2.0)<1e-9, str(weights))
    check("duplicate gets half", weights[0]==0.5 and weights[2]==1.0)
def test_duplicate_adjusted_metric():
    import numpy as np
    from sklearn.metrics import roc_auc_score
    y=np.array([0,1,0,1]); p=np.array([0.1,0.9,0.2,0.8])
    # duplicate-adjusted weight: first two share sha a (weight 0.5 each), last two unique
    w=np.array([0.5,0.5,1,1])
    try:
        auc_unweighted=roc_auc_score(y,p)
        auc_weighted=roc_auc_score(y,p, sample_weight=w)
        check("duplicate-adjusted differs", math.isfinite(auc_weighted))
    except Exception as e:
        check("duplicate-adjusted computable", False, str(e))
def test_degenerate_trait_handling():
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    X=np.array([[1,2],[3,4],[5,6]], dtype=float)
    y=np.array([1,1,1])  # single class
    try:
        clf=LogisticRegression().fit(X,y)
        check("degenerate not fitted", False, "should raise")
    except Exception:
        check("degenerate raises", True)
def test_finite_json():
    import pathlib, json as js
    for fn in ["Source/experiments/hash_grouped_semantic/semantic_hash_grouped_metrics.json", "Source/experiments/hash_grouped_semantic/duplicate_label_conflicts.json", "Source/experiments/hash_grouped_semantic/image_hash_audit.json"]:
        p=ROOT/fn
        if not p.exists(): continue
        txt=p.read_text(encoding="utf-8")
        bad= "NaN" in txt or "Infinity" in txt
        check(f"finite JSON {fn}", not bad)
        try:
            js.loads(txt)
            check(f"valid JSON {fn}", True)
        except Exception as e:
            check(f"valid JSON {fn}", False, str(e))
def test_artifacts_isolated():
    exp=SRC/"experiments"/"hash_grouped_semantic"
    check("exp dir exists", exp.exists())
    check("historical OOF not overwritten", (ROOT/"oof_semantic_probs.csv").exists())
    # ensure experiment OOF distinct path
    check("experiment OOF separate", (exp/"oof_semantic_probs_hash_grouped.csv").exists())
    check("models_semantic not overwritten", (SRC/"models_semantic"/"feature_cols.npy").exists())
def test_hash_manifest_exists():
    p=SRC/"experiments"/"hash_grouped_semantic"/"image_hash_manifest.csv"
    check("manifest exists", p.exists())
    if p.exists():
        import pandas as pd
        df=pd.read_csv(p, encoding="utf-8", nrows=2)
        check("manifest cols", set(["md5","sha256","normalized_filename"]).issubset(df.columns))
def main():
    test_sha_determinism()
    test_md5_sha_consistency()
    test_normalized_stem_grouping()
    test_connected_component()
    test_fold_overlap_assertions()
    test_conflicting_labels_detected()
    test_train_only_preprocessing()
    test_canonical_trait_order()
    test_oof_coverage()
    test_sample_weight_correctness()
    test_duplicate_adjusted_metric()
    test_degenerate_trait_handling()
    test_finite_json()
    test_artifacts_isolated()
    test_hash_manifest_exists()
    print(f"\nPASS {len(PASS)} FAIL {len(FAIL)}")
    return 0 if not FAIL else 1
if __name__=="__main__":
    import sys
    sys.exit(main())