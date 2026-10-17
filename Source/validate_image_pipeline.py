# -*- coding: utf-8 -*-
"""
Validation for image→CSV→semantic→text pipeline.
Covers: feature parity, semantic parity, image→semantic consistency, E2E smoke, error handling.
"""
import warnings
warnings.filterwarnings("ignore")
import json, time, sys, traceback, os, re, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(__file__).parent / "models_semantic"
CSV_CLEAN = ROOT / "faces_full_features_clean.csv"

def norm_name(s):
    s = os.path.basename(str(s).strip())
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\)\s+\(", ")(", s)
    return s.lower()

def test_feature_parity():
    print("== Feature schema parity ==")
    feat_cols = np.load(MODEL_DIR / "feature_cols.npy", allow_pickle=True).tolist()
    base_cols = [c for c in feat_cols if not c.endswith("__missing")]
    miss_cols = [c for c in feat_cols if c.endswith("__missing")]
    clean_cols = pd.read_csv(CSV_CLEAN, nrows=0, encoding="utf-8-sig").columns.tolist()
    clean_set = set(clean_cols)
    base_set = set(base_cols)
    common = base_set & clean_set
    missing = base_set - clean_set
    unexpected = clean_set - base_set  # many, clean is 452 vs 95 base
    print(f" base required: {len(base_cols)}  missing-indicator: {len(miss_cols)}  clean total: {len(clean_cols)}")
    print(f" common base in clean: {len(common)}/{len(base_cols)}")
    print(f" missing required columns: {missing if missing else 'none'}")
    print(f" clean not in semantic base (expected ~357): {len(unexpected)}")
    # check exact missing bases should be 0
    ok = len(missing)==0 and len(common)==len(base_cols)
    print(f" schema match: {'PASS' if ok else 'FAIL'}")
    return {"base_count": len(base_cols), "missing_count": len(miss_cols), "clean_count": len(clean_cols), "common": len(common), "missing_cols": list(missing), "ok": ok, "feat_cols": feat_cols}

def test_numeric_parity(sample_image="1 (1).jpg"):
    print("\n== Numeric parity (cached lookup vs stored row) ==")
    from predict_image_to_text import extract_features_from_image, prepare_features, load_feature_cols, predict_semantic_from_dataframe
    import pathlib
    feat_cols = load_feature_cols()
    # stored row
    full = pd.read_csv(CSV_CLEAN, encoding="utf-8-sig")
    # find row
    m = full[full["image_id"]==sample_image]
    if m.empty:
        # norm match
        for _, r in full.iterrows():
            if norm_name(r["image_id"])==norm_name(sample_image):
                m = r.to_frame().T
                break
    if m.empty:
        print(f" sample {sample_image} not found in clean CSV")
        return {"ok": False, "reason": "sample not found"}
    stored = m.iloc[0]
    # extracted via pipeline
    df_raw, meta, warns = extract_features_from_image(ROOT / "Humans" / sample_image)
    # compare common columns
    common_cols = [c for c in df_raw.columns if c in full.columns]
    # numeric diff
    max_diff = 0
    mean_diff = 0
    n = 0
    diffs = []
    for c in common_cols:
        a = pd.to_numeric(df_raw.iloc[0][c], errors="coerce")
        b = pd.to_numeric(stored[c], errors="coerce")
        if pd.isna(a) and pd.isna(b):
            continue
        if pd.isna(a) or pd.isna(b):
            diffs.append((c, "null mismatch", a, b))
            continue
        d = abs(float(a)-float(b))
        max_diff = max(max_diff, d)
        mean_diff += d
        n+=1
    mean_diff = mean_diff / n if n else 0
    print(f" sample: {sample_image}")
    print(f" common cols compared: {n}")
    print(f" max abs diff: {max_diff:.6g}  mean: {mean_diff:.6g}")
    print(f" null mismatches: {len([x for x in diffs if 'null' in str(x)])}")
    print(f" warnings: {warns}")
    ok = max_diff < 1e-6
    print(f" numeric parity: {'PASS' if ok else 'FAIL'}")
    return {"sample": sample_image, "common": n, "max_diff": float(max_diff), "mean_diff": float(mean_diff), "ok": ok, "meta": meta}

def test_semantic_parity(sample_image="1 (1).jpg"):
    print("\n== Semantic parity (stored CSV row -> semantic) vs in-memory helper ==")
    from predict_image_to_text import predict_semantic_from_dataframe, prepare_features, load_semantic_artifacts
    import pickle
    feat_cols, trait_names, imp, scaler, clfs = load_semantic_artifacts()
    full = pd.read_csv(CSV_CLEAN, encoding="utf-8-sig")
    m = full[full["image_id"]==sample_image]
    if m.empty:
        for _, r in full.iterrows():
            if norm_name(r["image_id"])==norm_name(sample_image):
                m = r.to_frame().T
                break
    df_raw = m.iloc[0:1].copy()
    # path A: via helper
    probs_a, _ = predict_semantic_from_dataframe(df_raw)
    # path B: manual same as predict_semantic.py
    X_raw = prepare_features(df_raw.copy(), feat_cols)
    X_imp = imp.transform(X_raw[feat_cols])
    X_s = scaler.transform(X_imp)
    probs_b = {}
    for t in trait_names:
        probs_b[t] = float(clfs[t].predict_proba(X_s[0:1])[0,1])
    maxd = max(abs(probs_a[k]-probs_b[k]) for k in trait_names)
    print(f" max abs prob diff: {maxd:.2e}")
    print(f" PASS" if maxd <= 1e-10 else " FAIL")
    # also compare vs running predict_semantic helper directly
    return {"max_diff": float(maxd), "ok": maxd <= 1e-10, "probs_a": probs_a}

def test_image_semantic_consistency(sample_image="1 (1).jpg"):
    print("\n== Image -> semantic consistency ==")
    from predict_image_to_text import extract_features_from_image, predict_semantic_from_dataframe
    full = pd.read_csv(CSV_CLEAN, encoding="utf-8-sig")
    m = full[full["image_id"]==sample_image]
    if m.empty:
        for _, r in full.iterrows():
            if norm_name(r["image_id"])==norm_name(sample_image):
                m = r.to_frame().T
                break
    df_stored = m.iloc[0:1].copy()
    probs_stored, _ = predict_semantic_from_dataframe(df_stored)
    df_extracted, meta, _ = extract_features_from_image(ROOT / "Humans" / sample_image)
    probs_extracted, _ = predict_semantic_from_dataframe(df_extracted)
    # feature diff
    common_cols = [c for c in df_stored.columns if c in df_extracted.columns]
    max_fd = 0
    for c in common_cols:
        a = pd.to_numeric(df_stored.iloc[0][c], errors="coerce")
        b = pd.to_numeric(df_extracted.iloc[0][c], errors="coerce")
        if pd.isna(a) and pd.isna(b):
            continue
        if pd.isna(a) or pd.isna(b):
            continue
        max_fd = max(max_fd, abs(float(a)-float(b)))
    max_pd = max(abs(probs_stored[k]-probs_extracted[k]) for k in probs_stored)
    print(f" max feature diff: {max_fd:.6g}")
    print(f" max prob diff: {max_pd:.6g}")
    print(f" PASS" if max_pd < 1e-9 else " notable drift" if max_pd < 1e-3 else " FAIL drift")
    return {"max_feature_diff": float(max_fd), "max_prob_diff": float(max_pd), "ok": max_pd < 1e-6}

def test_e2e_smoke(sample_image="1 (1).jpg"):
    print("\n== E2E smoke (image->CSV->semantic->text) ==")
    from predict_image_to_text import predict_image_to_text
    from text_gen_common import validate_structure
    t0 = time.time()
    out_json = ROOT / "outputs" / "validate_smoke_result.json"
    res = predict_image_to_text(ROOT / "Humans" / sample_image, output=str(out_json))
    dt = time.time()-t0
    csv_path = Path(res["features"]["csv_path"])
    csv_ok = csv_path.exists()
    trait_ok = res["semantic"]["trait_count"]==45
    valid = res["validation"]["valid"]
    print(f" image: {sample_image}")
    print(f" success: {res['success']}  valid 5-section: {valid}  trait_count: {res['semantic']['trait_count']}")
    print(f" CSV exists: {csv_ok}  path: {csv_path}")
    print(f" timing: {res['timing']}  wall: {dt:.1f}s")
    # check probs in [0,1]
    bad = [k for k,v in res["semantic"]["probabilities"].items() if not (0<=v<=1 and np.isfinite(v))]
    print(f" probs out of range: {bad if bad else 'none'}")
    # check all 5 sections present
    secs = res["predictions"]
    empty = [k for k,v in secs.items() if not v.strip()]
    print(f" empty sections: {empty if empty else 'none'}")
    ok = csv_ok and trait_ok and valid and not bad and not empty
    print(f" E2E SMOKE: {'PASS' if ok else 'FAIL'}")
    return {"ok": ok, "result": res, "wall_s": dt}

def test_errors():
    print("\n== Error handling ==")
    from predict_image_to_text import predict_image_to_text
    cases = []
    # missing image
    try:
        predict_image_to_text(ROOT / "no_such_file.jpg")
        cases.append(("missing image", False, "should have raised"))
    except FileNotFoundError as e:
        print(f" missing image: PASS ({e})")
        cases.append(("missing image", True, str(e)))
    except Exception as e:
        print(f" missing image: FAIL wrong exc {e}")
        cases.append(("missing image", False, str(e)))
    # corrupt image
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
        tf.write(b"not an image")
        tf_path = tf.name
    try:
        predict_image_to_text(tf_path)
        cases.append(("corrupt image", False, "should have raised"))
        print(" corrupt image: FAIL should have raised")
    except ValueError as e:
        ok = "corrupt" in str(e).lower() or "Unsupported" in str(e)
        print(f" corrupt image: {'PASS' if ok else 'FAIL'} ({e})")
        cases.append(("corrupt image", ok, str(e)))
    except Exception as e:
        print(f" corrupt image: FAIL {e}")
        cases.append(("corrupt image", False, str(e)))
    finally:
        try: os.remove(tf_path)
        except: pass
    # no face / unknown image (heavy detector missing -> unknown not in CSV)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
        # create a valid 10x10 png but unknown id
        from PIL import Image
        im = Image.new("RGB", (32,32), color=(200,100,50))
        im.save(tf.name, "JPEG")
        tf_path2 = tf.name
    try:
        predict_image_to_text(tf_path2)
        print(" no-face unknown: FAIL should have raised")
        cases.append(("no-face unknown", False, "should have raised"))
    except ValueError as e:
        ok = "No face" in str(e)
        print(f" no-face unknown: {'PASS' if ok else 'FAIL'} ({e})")
        cases.append(("no-face unknown", ok, str(e)))
    except Exception as e:
        print(f" no-face unknown: FAIL {e}")
        cases.append(("no-face unknown", False, str(e)))
    finally:
        try: os.remove(tf_path2)
        except: pass
    # schema missing already handled via fill NaN - not error, but ensure trait count still 45
    try:
        df = pd.read_csv(CSV_CLEAN, nrows=1, encoding="utf-8-sig")
        df2 = df.drop(columns=[c for c in df.columns if c.endswith("_ratio")][:5])
        from predict_image_to_text import predict_semantic_from_dataframe
        probs, _ = predict_semantic_from_dataframe(df2)
        ok = len(probs)==45
        print(f" schema missing cols (filled NaN via imputer): {'PASS' if ok else 'FAIL'}")
        cases.append(("schema missing cols", ok, "filled via saved imputer"))
    except Exception as e:
        print(f" schema missing: FAIL {e}")
        cases.append(("schema missing", False, str(e)))
    # missing model
    try:
        from predict_image_to_text import predict_image_to_text as pit
        pit(ROOT / "Humans" / "1 (1).jpg", model_path="no_such_model_dir")
        print(" missing text model: FAIL should have raised")
        cases.append(("missing text model", False, "should have raised"))
    except Exception as e:
        print(f" missing text model: PASS ({type(e).__name__}: {str(e)[:120]})")
        cases.append(("missing text model", True, str(e)[:200]))
    return cases

def main():
    print("Validate image pipeline")
    print(f" ROOT: {ROOT}")
    print(f" Python: {sys.version.split()[0]}")
    r1 = test_feature_parity()
    r2 = test_numeric_parity("1 (1).jpg")
    # also test a second image
    r2b = test_numeric_parity("1 (10).jpg")
    r3 = test_semantic_parity("1 (1).jpg")
    r4 = test_image_semantic_consistency("1 (1).jpg")
    r5 = test_e2e_smoke("1 (1).jpg")
    r6 = test_errors()
    summary = {
        "feature_parity": r1,
        "numeric_parity_1": r2,
        "numeric_parity_2": r2b,
        "semantic_parity": r3,
        "image_semantic_consistency": r4,
        "e2e_smoke": {"ok": r5["ok"], "wall_s": r5["wall_s"], "validation": r5["result"]["validation"], "timing": r5["result"]["timing"]},
        "errors": r6,
    }
    out = ROOT / "outputs" / "validate_image_pipeline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {out}")
    # overall
    all_ok = r1["ok"] and r2["ok"] and r3["ok"] and r4["ok"] and r5["ok"] and all(x[1] for x in r6)
    print(f"\nOVERALL: {'PASS' if all_ok else 'PARTIAL/FAIL'}")
    if not all_ok:
        print(" Check outputs/validate_image_pipeline.json for details")
    return 0 if all_ok else 1

if __name__=="__main__":
    sys.exit(main())