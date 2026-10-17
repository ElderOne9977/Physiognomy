# -*- coding: utf-8 -*-
"""Phase 3.16 - Real-world robustness diagnostics for the frozen Physiognomy pipeline.

This harness MEASURES robustness. It does not retrain, retune, or change any
frozen artifact. It is read-only with respect to models: every stage only calls
existing public entry points and writes JSON evidence into
`Source/experiments/real_world_robustness_phase316/`.

Stages
  A  environment & artifact fingerprint (hashes of frozen files)
  B  corpus provenance & genuine-unseen audit
  C  genuine unseen-image validation (BLOCKED when no confirmed-unseen faces exist)
  D  decode / IO robustness matrix (formats, corrupt inputs)
  E  detector robustness across the corpus
  F  feature stability under image perturbation
  G  semantic stability under image perturbation
  H  text stability across template generators under image perturbation
  I  text-level robustness (determinism, structure, no hallucinated traits)
  J  shadow-mode quality gate evaluation
  K  CPU latency benchmark (cold/warm, percentiles)
  L  summary

Usage:
  python Source/validate_real_world_robustness.py --stage all
  python Source/validate_real_world_robustness.py --stage quick --max-images 6
  python Source/validate_real_world_robustness.py --stage D
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import warnings  # noqa: E402
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

EXP_DIR = SRC / "experiments" / "real_world_robustness_phase316"
SEED = 42
FROZEN_ARTIFACTS = [
    SRC / "models_semantic" / "feature_cols.npy",
    SRC / "models_semantic" / "trait_names.npy",
    SRC / "models_semantic" / "imputer.pkl",
    SRC / "models_semantic" / "scaler.pkl",
    SRC / "models_semantic" / "semantic_schema.json",
    SRC / "models_semantic" / "missing_indicator_cols.json",
    SRC / "models_semantic" / "trait_thresholds.json",
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _finite(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _round(x, n=6):
    v = _finite(x)
    return None if v is None else round(v, n)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def write_json(name, payload):
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    p = EXP_DIR / name
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"  wrote {p.relative_to(ROOT)}")
    return p


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def load_image_bgr(path):
    from physiognomy import load_image_robust
    return load_image_robust(path)


def extract(path):
    """Returns (df, meta). Raises on failure."""
    from physiognomy import extract_features
    return extract_features(path)


def semantic_probs(df):
    from predict_image_to_text import predict_semantic_from_dataframe
    probs, names = predict_semantic_from_dataframe(df)
    return probs, list(names)


def load_feature_cols():
    fc = np.load(SRC / "models_semantic" / "feature_cols.npy", allow_pickle=True).tolist()
    return [str(c) for c in fc]


def base_missing_count(df):
    """Count of the 95 required base features missing/non-numeric in df row 0."""
    fc = load_feature_cols()
    base = [c for c in fc if not c.endswith("__missing")]
    row = df.iloc[0]
    miss = 0
    for c in base:
        if c not in df.columns:
            miss += 1
            continue
        if _finite(pd.to_numeric(pd.Series([row[c]]), errors="coerce").iloc[0]) is None:
            miss += 1
    return miss, len(base)


def _candidate_corpus_images(limit=None):
    humans = ROOT / "Humans"
    if not humans.is_dir():
        return []
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = sorted(p for p in humans.iterdir() if p.suffix.lower() in exts)
    if limit and limit < len(files):
        rng = np.random.RandomState(SEED)
        idx = rng.choice(len(files), size=limit, replace=False)
        files = [files[i] for i in sorted(idx)]
    return files


def _demo_tmp_unique():
    d = ROOT / "outputs" / "demo_tmp"
    if not d.is_dir():
        return []
    seen = {}
    for p in sorted(d.glob("*.jpg")):
        h = sha256_file(p)
        seen.setdefault(h, p)
    return list(seen.values())


# --------------------------------------------------------------------------
# Stage A - environment & artifact fingerprint
# --------------------------------------------------------------------------
def stage_A():
    print("[A] environment & artifact fingerprint")
    fp = {}
    for a in FROZEN_ARTIFACTS:
        if a.exists():
            fp[a.name] = {"sha256": sha256_file(a), "bytes": a.stat().st_size}
        else:
            fp[a.name] = {"sha256": None, "bytes": None, "missing": True}

    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ["numpy", "pandas", "sklearn", "cv2", "PIL", "onnxruntime", "psutil"]:
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "unknown")
        except Exception as e:
            versions[mod] = f"unavailable: {type(e).__name__}"

    try:
        import psutil
        cpu = {
            "logical_cores": psutil.cpu_count(logical=True),
            "physical_cores": psutil.cpu_count(logical=False),
            "total_ram_gb": round(psutil.virtual_memory().total / 1e9, 2),
        }
    except Exception:
        cpu = {"logical_cores": os.cpu_count()}

    try:
        import torch
        cpu["torch_cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        cpu["torch_cuda_available"] = None

    fc = load_feature_cols()
    base = [c for c in fc if not c.endswith("__missing")]
    payload = {
        "stage": "A",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": SEED,
        "frozen_artifacts": fp,
        "versions": versions,
        "hardware": cpu,
        "feature_cols_total": len(fc),
        "feature_cols_base": len(base),
        "feature_cols_missing_indicators": len(fc) - len(base),
    }
    return payload


# --------------------------------------------------------------------------
# Stage B - corpus provenance & genuine-unseen audit
# --------------------------------------------------------------------------
def _training_hashes():
    man = SRC / "experiments" / "hash_grouped_semantic" / "image_hash_manifest.csv"
    if not man.exists():
        return None
    df = pd.read_csv(man)
    return set(df["sha256"].astype(str))


def stage_B():
    print("[B] corpus provenance & genuine-unseen audit")
    train = _training_hashes()
    humans = _candidate_corpus_images()
    demo = _demo_tmp_unique()

    def classify(p):
        h = sha256_file(p)
        return {"path": str(p.relative_to(ROOT)), "sha256": h,
                "in_training_manifest": (h in train) if train else None}

    demo_recs = [classify(p) for p in demo]
    extra = []
    for cand in ["outputs/unseen_face_test.jpg"]:
        cp = ROOT / cand
        if cp.exists():
            extra.append(classify(cp))

    payload = {
        "stage": "B",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "training_manifest_available": bool(train),
        "training_manifest_unique_hashes": len(train) if train else None,
        "humans_corpus_count": len(humans),
        "humans_all_in_training": None,
        "demo_tmp_unique_images": len(demo_recs),
        "demo_tmp_any_in_training": (any(r["in_training_manifest"] for r in demo_recs) if train else None),
        "demo_tmp": demo_recs,
        "extra_candidates": extra,
        "genuine_unseen_candidates": [r for r in demo_recs + extra if r["in_training_manifest"] is False],
        "provenance_note": (
            "Humans/ is the source corpus used to build the training manifest. "
            "outputs/demo_tmp/ images are NOT byte-identical to any training image, "
            "but their provenance is unverified: they may be re-encoded/derived "
            "copies of training faces. They are therefore reported as 'candidate "
            "unseen' only, never as confirmed unseen."
        ),
    }
    if train and humans:
        in_train = 0
        checked = 0
        for p in humans[:200]:
            checked += 1
            if sha256_file(p) in train:
                in_train += 1
        payload["humans_all_in_training"] = (in_train == checked)
        payload["humans_sampled"] = checked
        payload["humans_sampled_in_training"] = in_train
    return payload


# --------------------------------------------------------------------------
# Stage C - genuine unseen-image validation
# --------------------------------------------------------------------------
def stage_C():
    print("[C] genuine unseen-image validation")
    b = stage_B()
    confirmed = list(b["genuine_unseen_candidates"])
    payload = {
        "stage": "C",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "BLOCKED",
        "blocked_reason": (
            "No confirmed-unseen face images exist locally. The only non-training "
            "bytes available (outputs/demo_tmp/) have unverified provenance and may "
            "be derived from training faces, so accuracy on them would not be a "
            "valid unseen-image generalization estimate."
        ),
        "candidate_non_training_images": len(confirmed),
        "candidate_paths": [r["path"] for r in confirmed][:20],
        "generalization_claim": None,
        "what_would_unblock": (
            "A labelled face image whose bytes are provably outside the training "
            "manifest AND whose subject is confirmed absent from the corpus "
            "(e.g. a freshly captured photo with signed provenance)."
        ),
    }
    return payload


# --------------------------------------------------------------------------
# Stage D - decode / IO robustness
# --------------------------------------------------------------------------
def stage_D():
    print("[D] decode / IO robustness matrix")
    import cv2
    from PIL import Image

    tmp = Path(tempfile.mkdtemp(prefix="p316_io_"))
    cases = []
    try:
        Image.new("RGB", (120, 120), (180, 120, 90)).save(tmp / "rgb.jpg", "JPEG")
        Image.new("RGB", (120, 120), (180, 120, 90)).save(tmp / "rgb.png", "PNG")
        Image.new("L", (120, 120), 128).save(tmp / "gray.png", "PNG")
        Image.new("RGBA", (120, 120), (180, 120, 90, 128)).save(tmp / "rgba.png", "PNG")
        Image.new("P", (120, 120)).save(tmp / "palette.png", "PNG")
        arr16 = (np.random.RandomState(0).rand(120, 120) * 65535).astype(np.uint16)
        cv2.imwrite(str(tmp / "u16.png"), arr16)
        Image.new("RGB", (3, 3), (0, 0, 0)).save(tmp / "tiny3.png", "PNG")
        Image.new("RGB", (120, 120), (10, 10, 10)).save(tmp / "webp.webp", "WEBP")
        (tmp / "zero.jpg").write_bytes(b"")
        data = (tmp / "rgb.jpg").read_bytes()
        (tmp / "trunc.jpg").write_bytes(data[: max(1, len(data) // 3)])
        (tmp / "text.txt").write_text("not an image")
        (tmp / "random.jpg").write_bytes(os.urandom(2000))

        expectations = {
            "rgb.jpg": "ok", "rgb.png": "ok", "gray.png": "ok", "rgba.png": "ok",
            "palette.png": "ok", "u16.png": "ok", "tiny3.png": "ok", "webp.webp": "ok",
            "zero.jpg": "none", "trunc.jpg": "none", "text.txt": "none",
            "random.jpg": "none", "missing.jpg": "none",
        }
        targets = sorted(tmp.iterdir()) + [tmp / "missing.jpg"]
        for f in targets:
            name = f.name
            rec = {"case": name, "expected": expectations.get(name, "ok")}
            t = time.time()
            try:
                img = load_image_bgr(str(f))
                rec["outcome"] = "none" if img is None else "ok"
                if img is not None:
                    rec["shape"] = list(img.shape)
                    rec["dtype"] = str(img.dtype)
                    rec["finite"] = bool(np.isfinite(img).all())
            except FileNotFoundError:
                rec["outcome"] = "error"
                rec["error_type"] = "FileNotFoundError"
            except Exception as e:
                rec["outcome"] = "error"
                rec["error_type"] = type(e).__name__
                rec["error_msg"] = str(e)[:120]
            rec["seconds"] = round(time.time() - t, 4)
            rec["matches_expectation"] = (rec["outcome"] == rec["expected"])
            cases.append(rec)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    passed = sum(1 for c in cases if c.get("matches_expectation"))
    return {
        "stage": "D",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cases": cases,
        "cases_total": len(cases),
        "cases_matching_expectation": passed,
        "note": (
            "load_image_robust returns None for undecodable input and raises "
            "FileNotFoundError for a missing path. Downstream code must treat "
            "None as a hard decode failure."
        ),
    }


# --------------------------------------------------------------------------
# Stage E - detector robustness
# --------------------------------------------------------------------------
def stage_E(max_images):
    print(f"[E] detector robustness (n<={max_images})")
    import physiognomy as ph
    from physiognomy import robust_detect_single_face, ensure_heavy_models
    ensure_heavy_models()

    imgs = _candidate_corpus_images(max_images)
    recs = []
    for p in imgs:
        rec = {"path": str(p.relative_to(ROOT))}
        try:
            img = load_image_bgr(str(p))
            if img is None:
                rec["status"] = "decode_none"
                recs.append(rec)
                continue
            t = time.time()
            det = robust_detect_single_face(ph.app, img)
            rec["seconds"] = round(time.time() - t, 3)
            if det is None:
                rec["status"] = "no_face"
            else:
                rec["status"] = det.get("status", "detected")
                rec["n_faces_history"] = [h.get("num_faces") for h in det.get("history", [])]
        except Exception as e:
            rec["status"] = "error"
            rec["error_type"] = type(e).__name__
            rec["error_msg"] = str(e)[:120]
        recs.append(rec)

    ok_states = {"normal_success", "recovered", "multi_face_selected", "detected"}
    det = sum(1 for r in recs if r.get("status") in ok_states)
    noface = sum(1 for r in recs if r.get("status") == "robust_miss")
    return {
        "stage": "E",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_images": len(recs),
        "n_detected": det,
        "n_no_face": noface,
        "detection_rate": _round(det / len(recs)) if recs else None,
        "records": recs,
    }


# --------------------------------------------------------------------------
# perturbation operators (deterministic)
# --------------------------------------------------------------------------
def perturbations(img):
    import cv2
    ops = {}
    ops["brightness_+20"] = cv2.convertScaleAbs(img, alpha=1.0, beta=20)
    ops["brightness_-20"] = cv2.convertScaleAbs(img, alpha=1.0, beta=-20)
    ops["contrast_1.2"] = cv2.convertScaleAbs(img, alpha=1.2, beta=0)
    ops["contrast_0.8"] = cv2.convertScaleAbs(img, alpha=0.8, beta=0)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 55])
    ops["jpeg_q55"] = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 30])
    ops["jpeg_q30"] = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * 0.7)), max(1, int(h * 0.7))), interpolation=cv2.INTER_AREA)
    ops["downup_0.7"] = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    small = cv2.resize(img, (max(1, int(w * 0.5)), max(1, int(h * 0.5))), interpolation=cv2.INTER_AREA)
    ops["downup_0.5"] = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 3.0, 1.0)
    ops["rotate_3deg"] = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 7.0, 1.0)
    ops["rotate_7deg"] = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    rng = np.random.RandomState(SEED)
    n = rng.normal(0, 6, img.shape).astype(np.float32)
    ops["gauss_noise_s6"] = np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)
    return ops


def _feature_delta(base_df, pert_df):
    base = base_df.iloc[0]
    pert = pert_df.iloc[0]
    common = [c for c in base_df.columns if c in pert_df.columns and c != "image_id"]
    diffs = []
    for c in common:
        a = _finite(pd.to_numeric(pd.Series([pert[c]]), errors="coerce").iloc[0])
        b = _finite(pd.to_numeric(pd.Series([base[c]]), errors="coerce").iloc[0])
        if a is not None and b is not None:
            diffs.append(abs(a - b))
    if not diffs:
        return {"n_compared": 0, "max_abs_delta": None, "mean_abs_delta": None}
    return {"n_compared": len(diffs), "max_abs_delta": _round(max(diffs)),
            "mean_abs_delta": _round(float(np.mean(diffs)))}


def _prob_delta(base_probs, pert_probs):
    keys = sorted(set(base_probs) & set(pert_probs))
    d = [abs(float(base_probs[k]) - float(pert_probs[k])) for k in keys]
    if not d:
        return {"n_compared": 0, "max_abs_delta": None, "mean_abs_delta": None}
    return {"n_compared": len(d), "max_abs_delta": _round(max(d)),
            "mean_abs_delta": _round(float(np.mean(d)))}


# --------------------------------------------------------------------------
# Stage F/G - feature & semantic stability under perturbation
# --------------------------------------------------------------------------
def stage_FG(max_images):
    print(f"[F/G] feature & semantic stability under perturbation (n<={max_images})")
    import cv2
    imgs = _candidate_corpus_images(max_images)
    tmp = Path(tempfile.mkdtemp(prefix="p316_pert_"))
    per_image = []
    try:
        for p in imgs:
            img = load_image_bgr(str(p))
            if img is None:
                continue
            try:
                base_df, base_meta = extract(str(p))
            except Exception as e:
                per_image.append({"path": str(p.relative_to(ROOT)),
                                  "error": f"{type(e).__name__}: {str(e)[:80]}"})
                continue
            try:
                base_probs, _ = semantic_probs(base_df)
            except Exception:
                base_probs = None
            base_active = {k for k, v in (base_probs or {}).items() if v >= 0.5}

            rec = {
                "path": str(p.relative_to(ROOT)),
                "base_qc_pass": bool(base_meta.get("qc_pass")),
                "base_detection_score": _round(base_meta.get("detection_score")),
                "perturbations": {},
            }
            for name, pimg in perturbations(img).items():
                pf = tmp / f"{p.stem}__{name}.png".replace("/", "_")
                cv2.imwrite(str(pf), pimg)
                entry = {}
                try:
                    p_df, p_meta = extract(str(pf))
                    entry["qc_pass"] = bool(p_meta.get("qc_pass"))
                    entry["qc_reasons"] = p_meta.get("qc_reasons")
                    entry["feature_delta"] = _feature_delta(base_df, p_df)
                    if base_probs is not None:
                        try:
                            p_probs, _ = semantic_probs(p_df)
                            entry["prob_delta"] = _prob_delta(base_probs, p_probs)
                            p_active = {k for k, v in p_probs.items() if v >= 0.5}
                            entry["active_trait_jaccard"] = _round(jaccard(base_active, p_active))
                            entry["n_active_base"] = len(base_active)
                            entry["n_active_pert"] = len(p_active)
                        except Exception as e:
                            entry["prob_error"] = f"{type(e).__name__}: {str(e)[:80]}"
                except Exception as e:
                    entry["error"] = f"{type(e).__name__}: {str(e)[:80]}"
                rec["perturbations"][name] = entry
            per_image.append(rec)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    agg = {}
    for rec in per_image:
        for name, e in rec.get("perturbations", {}).items():
            a = agg.setdefault(name, {"feature_max": [], "prob_max": [], "jaccard": [],
                                      "qc_fail": 0, "n": 0, "errors": 0})
            a["n"] += 1
            if "error" in e:
                a["errors"] += 1
                continue
            if e.get("qc_pass") is False:
                a["qc_fail"] += 1
            fd = e.get("feature_delta") or {}
            if fd.get("max_abs_delta") is not None:
                a["feature_max"].append(fd["max_abs_delta"])
            pdl = e.get("prob_delta") or {}
            if pdl.get("max_abs_delta") is not None:
                a["prob_max"].append(pdl["max_abs_delta"])
            if e.get("active_trait_jaccard") is not None:
                a["jaccard"].append(e["active_trait_jaccard"])

    summary = {}
    for name, a in agg.items():
        summary[name] = {
            "n": a["n"],
            "errors": a["errors"],
            "qc_fail": a["qc_fail"],
            "feature_max_abs_delta_mean": _round(np.mean(a["feature_max"])) if a["feature_max"] else None,
            "feature_max_abs_delta_worst": _round(max(a["feature_max"])) if a["feature_max"] else None,
            "prob_max_abs_delta_mean": _round(np.mean(a["prob_max"])) if a["prob_max"] else None,
            "prob_max_abs_delta_worst": _round(max(a["prob_max"])) if a["prob_max"] else None,
            "active_trait_jaccard_mean": _round(np.mean(a["jaccard"])) if a["jaccard"] else None,
            "active_trait_jaccard_min": _round(min(a["jaccard"])) if a["jaccard"] else None,
        }
    return per_image, summary


# --------------------------------------------------------------------------
# Stage H - text stability across generators
# --------------------------------------------------------------------------
def stage_H(max_images):
    print(f"[H] text stability across template generators (n<={max_images})")
    import cv2
    from predict_image_to_text import TEMPLATE_GENERATOR_CHOICES
    template_gens = list(TEMPLATE_GENERATOR_CHOICES)
    imgs = _candidate_corpus_images(max_images)
    tmp = Path(tempfile.mkdtemp(prefix="p316_txt_"))

    def gen_text(df, generator):
        from predict_image_to_text import _generate_template, _generate_template_concise
        probs, _ = semantic_probs(df)
        if generator == "template_0.5":
            text, sections, v, _, active, policy, _t = _generate_template(probs, "threshold_0.5")
        elif generator == "template_calibrated":
            text, sections, v, _, active, policy, _t = _generate_template(probs, "calibrated")
        else:
            text, sections, v, _, active, policy, _t, _trace = _generate_template_concise(probs)
        return text, active, bool(v.get("valid"))

    per_image = []
    try:
        for p in imgs:
            img = load_image_bgr(str(p))
            if img is None:
                continue
            try:
                base_df, _ = extract(str(p))
            except Exception as e:
                per_image.append({"path": str(p.relative_to(ROOT)),
                                  "error": f"{type(e).__name__}: {str(e)[:80]}"})
                continue
            rec = {"path": str(p.relative_to(ROOT)), "generators": {}}
            base_texts = {}
            for g in template_gens:
                try:
                    text, active, valid = gen_text(base_df, g)
                    base_texts[g] = (text, active)
                    rec["generators"][g] = {"valid": valid, "n_active": len(active), "chars": len(text)}
                except Exception as e:
                    rec["generators"][g] = {"error": f"{type(e).__name__}: {str(e)[:80]}"}
            rec["cross_generator_agreement"] = {}
            for i in range(len(template_gens)):
                for j in range(i + 1, len(template_gens)):
                    gi, gj = template_gens[i], template_gens[j]
                    if gi in base_texts and gj in base_texts:
                        rec["cross_generator_agreement"][f"{gi}|{gj}"] = {
                            "jaccard": _round(jaccard(base_texts[gi][1], base_texts[gj][1]))}
            rec["determinism"] = {}
            for g in template_gens:
                if g in base_texts:
                    t2, _, _ = gen_text(base_df, g)
                    rec["determinism"][g] = (t2 == base_texts[g][0])

            pert = {}
            for name, pimg in perturbations(img).items():
                pf = tmp / f"{p.stem}__{name}.png".replace("/", "_")
                cv2.imwrite(str(pf), pimg)
                try:
                    p_df, _ = extract(str(pf))
                except Exception:
                    continue
                pe = {}
                for g in template_gens:
                    if g not in base_texts:
                        continue
                    try:
                        text, active, valid = gen_text(p_df, g)
                        pe[g] = {"valid": valid, "text_identical": (text == base_texts[g][0]),
                                 "active_jaccard": _round(jaccard(base_texts[g][1], active))}
                    except Exception as e:
                        pe[g] = {"error": f"{type(e).__name__}: {str(e)[:60]}"}
                pert[name] = pe
            rec["perturbation_stability"] = pert
            per_image.append(rec)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    agg = {}
    for rec in per_image:
        for name, pe in rec.get("perturbation_stability", {}).items():
            for g, e in pe.items():
                a = agg.setdefault(f"{name}::{g}", {"identical": 0, "n": 0, "jac": [], "invalid": 0})
                if "error" in e:
                    continue
                a["n"] += 1
                if e.get("text_identical"):
                    a["identical"] += 1
                if e.get("active_jaccard") is not None:
                    a["jac"].append(e["active_jaccard"])
                if e.get("valid") is False:
                    a["invalid"] += 1
    summary = {}
    for k, a in agg.items():
        summary[k] = {
            "n": a["n"],
            "identical_rate": _round(a["identical"] / a["n"]) if a["n"] else None,
            "active_jaccard_mean": _round(np.mean(a["jac"])) if a["jac"] else None,
            "active_jaccard_min": _round(min(a["jac"])) if a["jac"] else None,
            "invalid_outputs": a["invalid"],
        }
    return per_image, summary


# --------------------------------------------------------------------------
# Stage I - text-level robustness (structure / hallucination)
# --------------------------------------------------------------------------
def stage_I():
    print("[I] text-level robustness (structure, determinism, trait grounding)")
    from predict_image_to_text import TEMPLATE_GENERATOR_CHOICES, _generate_template, _generate_template_concise
    from generate_text_template import TRAIT_PHRASES
    from text_gen_common import load_schema

    schema_names, _cat = load_schema()
    schema_set = set(schema_names)
    template_gens = list(TEMPLATE_GENERATOR_CHOICES)

    rng = np.random.RandomState(SEED)
    vectors = {
        "all_zero": {t: 0.0 for t in schema_names},
        "all_one": {t: 1.0 for t in schema_names},
        "all_0.5": {t: 0.5 for t in schema_names},
        "random_1": {t: float(rng.rand()) for t in schema_names},
        "random_2": {t: float(rng.rand()) for t in schema_names},
    }
    t0 = schema_names[0]
    vectors["boundary_below"] = {x: (0.499 if x == t0 else 0.5) for x in schema_names}
    vectors["boundary_above"] = {x: (0.501 if x == t0 else 0.5) for x in schema_names}

    def gen(probs, generator):
        if generator == "template_0.5":
            text, sections, v, _, active, policy, _t = _generate_template(probs, "threshold_0.5")
        elif generator == "template_calibrated":
            text, sections, v, _, active, policy, _t = _generate_template(probs, "calibrated")
        else:
            text, sections, v, _, active, policy, _t, _trace = _generate_template_concise(probs)
        return text, active, v

    results = {}
    issues = []
    for vname, probs in vectors.items():
        for g in template_gens:
            rec = {"valid": None, "deterministic": None, "active_subset_of_schema": None, "chars": None}
            try:
                t1, a1, v1 = gen(probs, g)
                t2, a2, v2 = gen(probs, g)
                rec["valid"] = bool(v1.get("valid"))
                rec["deterministic"] = (t1 == t2)
                rec["active_subset_of_schema"] = set(a1).issubset(schema_set)
                rec["chars"] = len(t1)
                rec["n_active"] = len(a1)
                if not rec["valid"]:
                    issues.append(f"{vname}/{g}: invalid structure")
                if not rec["deterministic"]:
                    issues.append(f"{vname}/{g}: non-deterministic")
                if not rec["active_subset_of_schema"]:
                    issues.append(f"{vname}/{g}: active trait not in schema")
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {str(e)[:100]}"
                issues.append(f"{vname}/{g}: raised {type(e).__name__}")
            results[f"{vname}::{g}"] = rec

    boundary = {}
    for g in template_gens:
        try:
            _, a_below, _ = gen(vectors["boundary_below"], g)
            _, a_above, _ = gen(vectors["boundary_above"], g)
            boundary[g] = {"above_includes_below": set(a_below).issubset(set(a_above))}
        except Exception as e:
            boundary[g] = {"error": str(e)[:80]}

    phrase_traits = set(TRAIT_PHRASES.keys())
    return {
        "stage": "I",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "template_phrase_traits_subset_of_schema": phrase_traits.issubset(schema_set),
        "n_schema_traits": len(schema_set),
        "n_template_traits": len(phrase_traits),
        "vectors_tested": list(vectors.keys()),
        "generators": template_gens,
        "results": results,
        "boundary_monotonicity": boundary,
        "issues": issues,
        "issues_count": len(issues),
    }


# --------------------------------------------------------------------------
# Stage J - shadow-mode quality gate evaluation
# --------------------------------------------------------------------------
def stage_J(max_images):
    print(f"[J] shadow-mode quality gate evaluation (n<={max_images})")
    from image_quality_gate import compute_image_stats, evaluate_quality_gate

    imgs = _candidate_corpus_images(max_images)
    demo = _demo_tmp_unique()[:3]
    all_imgs = imgs + demo

    recs = []
    for p in all_imgs:
        rec = {"path": str(p.relative_to(ROOT))}
        img = None
        try:
            img = load_image_bgr(str(p))
        except Exception as e:
            rec["status"] = evaluate_quality_gate(decode_ok=False, inference_error=str(e)[:120])["status"]
            rec["reasons"] = ["image_decode_failed"]
            rec["error"] = f"{type(e).__name__}: {str(e)[:80]}"
            recs.append(rec)
            continue
        if img is None:
            gate = evaluate_quality_gate(decode_ok=False)
            rec["status"] = gate["status"]
            rec["reasons"] = gate["reasons"]
            rec["decode_ok"] = False
            recs.append(rec)
            continue
        try:
            stats = compute_image_stats(img)
            df, meta = extract(str(p))
            bm, br = base_missing_count(df)
            gate = evaluate_quality_gate(
                decode_ok=True,
                detection_status=meta.get("detection_status"),
                detected=bool(meta.get("detected")),
                multiple_faces_detected=bool(meta.get("multiple_faces_detected")),
                multiple_faces_selected=(meta.get("selected_face") == "largest_of_multiple"),
                detection_score=meta.get("detection_score"),
                qc_pass=meta.get("qc_pass"),
                qc_reasons=meta.get("qc_reasons"),
                raw_feature_count=int(df.shape[1]),
                base_required=br,
                base_missing=bm,
                image_stats=stats,
            )
            rec["qc_pass"] = bool(meta.get("qc_pass"))
            rec["raw_feature_count"] = int(df.shape[1])
            rec["base_missing"] = bm
            rec["status"] = gate["status"]
            rec["reasons"] = gate["reasons"]
        except Exception as e:
            msg = str(e)
            if "FACE_NOT_DETECTED" in msg:
                gate = evaluate_quality_gate(decode_ok=True, detected=False)
            else:
                gate = evaluate_quality_gate(decode_ok=True, inference_error=msg[:120])
            rec["status"] = gate["status"]
            rec["reasons"] = gate["reasons"]
            rec["error"] = f"{type(e).__name__}: {msg[:80]}"
        recs.append(rec)

    synthetic = {
        "decode_fail": evaluate_quality_gate(decode_ok=False)["status"],
        "no_face": evaluate_quality_gate(detected=False)["status"],
        "multi_ambiguous": evaluate_quality_gate(detection_status="multi_face", detected=True)["status"],
        "extraction_fail": evaluate_quality_gate(detected=True, raw_feature_count=120)["status"],
        "severe_missing": evaluate_quality_gate(detected=True, raw_feature_count=486,
                                                base_required=95, base_missing=60)["status"],
        "clean_pass": evaluate_quality_gate(detected=True, raw_feature_count=486, qc_pass=True,
                                            detection_score=0.9, base_required=95, base_missing=0)["status"],
        "qc_warn": evaluate_quality_gate(detected=True, raw_feature_count=486, qc_pass=False,
                                         qc_reasons="small_face", detection_score=0.9,
                                         base_required=95, base_missing=3)["status"],
        "missing_warn": evaluate_quality_gate(detected=True, raw_feature_count=486, qc_pass=True,
                                              detection_score=0.9, base_required=95, base_missing=30)["status"],
        "low_conf_warn": evaluate_quality_gate(detected=True, raw_feature_count=486, qc_pass=True,
                                               detection_score=0.2, base_required=95, base_missing=0)["status"],
    }

    from collections import Counter
    status_counts = dict(Counter(r["status"] for r in recs))
    expected = {"decode_fail": "ERROR", "no_face": "ABSTAIN", "multi_ambiguous": "ABSTAIN",
                "extraction_fail": "ABSTAIN", "severe_missing": "ABSTAIN", "clean_pass": "PASS",
                "qc_warn": "WARN", "missing_warn": "WARN", "low_conf_warn": "WARN"}
    synthetic_ok = synthetic == expected
    print(f"  synthetic gate contract: {'OK' if synthetic_ok else 'MISMATCH ' + str(synthetic)}")
    return {
        "stage": "J",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "policy": "shadow (non-blocking)",
        "n_images": len(recs),
        "status_counts": status_counts,
        "records": recs,
        "synthetic_expectations": synthetic,
        "synthetic_contract_ok": synthetic_ok,
        "note": (
            "In shadow mode the gate never blocks or alters predictions; it is "
            "recorded alongside results for later enforcement decisions."
        ),
    }


# --------------------------------------------------------------------------
# Stage K - CPU latency benchmark
# --------------------------------------------------------------------------
def stage_K(n_warm=8):
    print(f"[K] CPU latency benchmark (warm n={n_warm})")
    from physiognomy import ensure_heavy_models
    imgs = _candidate_corpus_images(max(1, n_warm))
    ensure_heavy_models()

    cold = None
    try:
        t = time.time()
        extract(str(imgs[0]))
        cold = round(time.time() - t, 3)
    except Exception as e:
        cold = f"error: {type(e).__name__}"

    warm_extract = []
    warm_semantic = []
    warm_end_to_end = []
    for p in imgs:
        try:
            t = time.time()
            df, _ = extract(str(p))
            warm_extract.append(time.time() - t)
            t = time.time()
            semantic_probs(df)
            warm_semantic.append(time.time() - t)
        except Exception:
            continue
    try:
        from predict_image_to_text import predict_image_to_text
        e2e_tmp = Path(tempfile.mkdtemp(prefix="p316_e2e_"))
        for p in imgs[: min(3, len(imgs))]:
            t = time.time()
            predict_image_to_text(str(p), feature_source="extract",
                                  generator="template_calibrated",
                                  features_output=str(e2e_tmp / f"{p.stem}_features.csv"))
            warm_end_to_end.append(time.time() - t)
        shutil.rmtree(e2e_tmp, ignore_errors=True)
    except Exception as e:
        warm_end_to_end = [f"error: {type(e).__name__}: {str(e)[:60]}"]

    def stats(xs):
        xs = [x for x in xs if isinstance(x, (int, float))]
        if not xs:
            return {"n": 0}
        a = np.asarray(xs, float)
        return {"n": len(xs), "mean_s": _round(a.mean()), "median_s": _round(np.median(a)),
                "p90_s": _round(np.percentile(a, 90)), "p99_s": _round(np.percentile(a, 99)),
                "min_s": _round(a.min()), "max_s": _round(a.max())}

    return {
        "stage": "K",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cold_start_extract_s": cold,
        "warm_extract": stats(warm_extract),
        "warm_semantic": stats(warm_semantic),
        "end_to_end_template": stats(warm_end_to_end),
        "device": "cpu",
        "note": (
            "Text generation measured with the deterministic template path only; "
            "FLAN-T5 inference is intentionally excluded from this CPU budget "
            "because it is far heavier and covered by earlier phases."
        ),
    }


# --------------------------------------------------------------------------
# Stage L - summary
# --------------------------------------------------------------------------
def _load_persisted():
    """Best-effort load of previously written stage artifacts (for `--stage L`)."""
    out = {}
    mapping = {"stage_a.json": "A", "stage_b.json": "B", "stage_c.json": "C",
               "stage_d.json": "D", "stage_e.json": "E", "stage_i.json": "I",
               "stage_j.json": "J", "stage_k.json": "K"}
    for fn, tag in mapping.items():
        p = EXP_DIR / fn
        if p.exists():
            try:
                out[tag] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
    p = EXP_DIR / "stage_fg.json"
    if p.exists():
        try:
            out["FG_summary"] = json.loads(p.read_text(encoding="utf-8")).get("summary", {})
        except Exception:
            pass
    p = EXP_DIR / "stage_h.json"
    if p.exists():
        try:
            out["H_summary"] = json.loads(p.read_text(encoding="utf-8")).get("summary", {})
        except Exception:
            pass
    return out


def stage_L(stages):
    print("[L] summary")
    if not stages:
        stages = _load_persisted()
    a = stages.get("A", {})
    c = stages.get("C", {})
    d = stages.get("D", {})
    e = stages.get("E", {})
    fg_summary = stages.get("FG_summary", {})
    i = stages.get("I", {})
    j = stages.get("J", {})
    k = stages.get("K", {})

    worst_feat = None
    worst_prob = None
    worst_jac = None
    for name, s in fg_summary.items():
        if s.get("feature_max_abs_delta_worst") is not None:
            if worst_feat is None or s["feature_max_abs_delta_worst"] > worst_feat[1]:
                worst_feat = (name, s["feature_max_abs_delta_worst"])
        if s.get("prob_max_abs_delta_worst") is not None:
            if worst_prob is None or s["prob_max_abs_delta_worst"] > worst_prob[1]:
                worst_prob = (name, s["prob_max_abs_delta_worst"])
        if s.get("active_trait_jaccard_min") is not None:
            if worst_jac is None or s["active_trait_jaccard_min"] < worst_jac[1]:
                worst_jac = (name, s["active_trait_jaccard_min"])

    findings = []
    if c.get("status") == "BLOCKED":
        findings.append("Genuine unseen-image validation is BLOCKED (no confirmed-unseen faces).")
    if d.get("cases_total"):
        findings.append(f"IO robustness: {d.get('cases_matching_expectation')}/{d.get('cases_total')} inputs handled as expected.")
    if e.get("detection_rate") is not None:
        findings.append(f"Detector recovered faces on {e.get('n_detected')}/{e.get('n_images')} corpus images.")
    if worst_feat:
        findings.append(f"Largest feature drift under perturbation: {worst_feat[0]} (max |delta|={worst_feat[1]}).")
    if worst_prob:
        findings.append(f"Largest semantic prob drift: {worst_prob[0]} (max |delta|={worst_prob[1]}).")
    if worst_jac:
        findings.append(f"Lowest active-trait agreement under perturbation: {worst_jac[0]} (Jaccard={worst_jac[1]}).")
    if i.get("issues_count") is not None:
        findings.append(f"Text structural issues across synthetic vectors: {i.get('issues_count')}.")
    if j.get("status_counts"):
        findings.append(f"Shadow quality gate statuses: {j.get('status_counts')}.")
    if isinstance(k.get("warm_extract"), dict) and k["warm_extract"].get("median_s") is not None:
        findings.append(f"Warm extract median {k['warm_extract']['median_s']}s, semantic median "
                        f"{k.get('warm_semantic', {}).get('median_s')}s on CPU.")

    return {
        "stage": "L",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "phase": "3.16 real-world robustness diagnostics",
        "frozen_artifacts_hashed": list(a.get("frozen_artifacts", {}).keys()),
        "findings": findings,
        "key_metrics": {
            "detection_rate": e.get("detection_rate"),
            "io_cases_ok": f"{d.get('cases_matching_expectation')}/{d.get('cases_total')}",
            "worst_feature_drift": worst_feat,
            "worst_prob_drift": worst_prob,
            "worst_active_jaccard": worst_jac,
            "text_issues": i.get("issues_count"),
            "gate_status_counts": j.get("status_counts"),
            "warm_extract_median_s": k.get("warm_extract", {}).get("median_s"),
            "warm_semantic_median_s": k.get("warm_semantic", {}).get("median_s"),
        },
        "unseen_validation_status": c.get("status"),
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    global EXP_DIR
    ap = argparse.ArgumentParser(description="Phase 3.16 real-world robustness diagnostics")
    ap.add_argument("--stage", default="all",
                    help="all | quick | A | B | C | D | E | F | G | FG | H | I | J | K | L")
    ap.add_argument("--max-images", type=int, default=8, help="images per heavy stage")
    ap.add_argument("--out-dir", default=str(EXP_DIR))
    args = ap.parse_args()

    EXP_DIR = Path(args.out_dir)
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    which = args.stage.lower()
    do_all = which in ("all", "quick")
    max_images = args.max_images

    stages = {}
    t0 = time.time()

    def run(tag, fn, *a, **kw):
        p = fn(*a, **kw)
        stages[tag] = p
        write_json(f"stage_{tag.lower()}.json", p)
        return p

    try:
        if do_all or which == "a":
            run("A", stage_A)
        if do_all or which == "b":
            run("B", stage_B)
        if do_all or which == "c":
            run("C", stage_C)
        if do_all or which == "d":
            run("D", stage_D)
        if do_all or which == "e":
            run("E", stage_E, max_images)
        if do_all or which in ("f", "g", "fg"):
            per_image, summary = stage_FG(max_images)
            stages["FG"] = {"stage": "F/G", "per_image": per_image, "summary": summary,
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
            write_json("stage_fg.json", stages["FG"])
            stages["FG_summary"] = summary
        if do_all or which == "h":
            per_image, summary = stage_H(max_images)
            stages["H"] = {"stage": "H", "per_image": per_image, "summary": summary,
                           "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
            write_json("stage_h.json", stages["H"])
            stages["H_summary"] = summary
        if do_all or which == "i":
            run("I", stage_I)
        if do_all or which == "j":
            run("J", stage_J, max_images)
        if do_all or which == "k":
            run("K", stage_K, min(max_images, 8))
        if do_all or which == "l":
            run("L", stage_L, stages)

        print(f"\nDone in {time.time() - t0:.1f}s. Artifacts in {EXP_DIR}")
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()