# -*- coding: utf-8 -*-
"""
End-to-end demo pipeline:
  face image -> facial geometry (452-col CSV row) -> saved semantic preprocessing
  -> 45 probs -> text generator (FLAN-T5 or deterministic template) -> 5-section text
CLI: python Source/predict_image_to_text.py --image path/to/face.jpg
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
import argparse, json, time, sys, traceback, os, re
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(__file__).parent / "models_semantic"
TEXT_MODEL_DEFAULT = Path(__file__).parent / "models_text" / "flan_t5_small"
CSV_CLEAN = ROOT / "faces_full_features_clean.csv"
GENERATOR_CHOICES = ("flan_t5", "template_0.5", "template_calibrated", "template_concise",
                     "controlled_deterministic", "controlled_flan_guarded")
# Legacy deterministic template modes (phrase-bank only, no planner/validator).
# Kept explicit so older robustness harnesses do not silently pick up new modes.
TEMPLATE_GENERATOR_CHOICES = ("template_0.5", "template_calibrated", "template_concise")
TEMPLATE_THRESHOLDS_PATH = MODEL_DIR / "trait_thresholds.json"

def get_default_generator() -> str:
    """Return default generator from PHYSIO_DEFAULT_GENERATOR env var, falling back safely to flan_t5."""
    val = os.environ.get("PHYSIO_DEFAULT_GENERATOR", "flan_t5").strip()
    return val if val in GENERATOR_CHOICES else "flan_t5"

DEFAULT_GENERATOR = get_default_generator()
_semantic_cache = {}
_text_cache = {}
sys.path.insert(0, str(Path(__file__).parent))
from predict_semantic import prepare_features, load_models as _load_models  # single source of truth

def load_feature_cols():
    return np.load(MODEL_DIR / "feature_cols.npy", allow_pickle=True).tolist()

def load_semantic_artifacts():
    if _semantic_cache:
        return _semantic_cache["feat_cols"], _semantic_cache["trait_names"], _semantic_cache["imp"], _semantic_cache["scaler"], _semantic_cache["clfs"]
    feat_cols, trait_names, imp, scaler, clfs = _load_models()
    _semantic_cache.update(feat_cols=feat_cols, trait_names=trait_names, imp=imp, scaler=scaler, clfs=clfs)
    return feat_cols, trait_names, imp, scaler, clfs

def predict_semantic_from_dataframe(df_raw: pd.DataFrame):
    feat_cols, trait_names, imp, scaler, clfs = load_semantic_artifacts()
    X_raw = prepare_features(df_raw.copy(), feat_cols)
    X_imp = imp.transform(X_raw[feat_cols])
    X_s = scaler.transform(X_imp)
    probs = {}
    for t in trait_names:
        clf = clfs[t]
        p = float(clf.predict_proba(X_s[0:1])[0, 1])
        probs[t] = p
    assert len(probs) == 45, f"expected 45 got {len(probs)}"
    for v in probs.values():
        assert np.isfinite(v) and 0 <= v <= 1, f"prob out of range {v}"
    return probs, trait_names

def load_text_generator(model_path=None):
    key = str(model_path or TEXT_MODEL_DEFAULT)
    if key in _text_cache:
        return _text_cache[key]
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    mp = Path(key)
    if not mp.exists():
        raise PipelineError("MODEL_MISSING", f"text model not found: {key}")
    try:
        tok = AutoTokenizer.from_pretrained(key)
        model = AutoModelForSeq2SeqLM.from_pretrained(key)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    except PipelineError:
        raise
    except OSError as e:
        raise PipelineError("MODEL_MISSING", str(e)) from e
    except Exception as e:
        raise PipelineError("MODEL_LOAD_FAILED", str(e)) from e
    _text_cache[key] = (tok, model, device)
    return tok, model, device

def generate_text_from_semantics(probs: dict, trait_names, model_path=None):
    sys.path.insert(0, str(Path(__file__).parent))
    from text_gen_common import load_schema, format_input, parse_sections, validate_structure
    schema_names, cat_of = load_schema()
    tok, model, device = load_text_generator(model_path)
    import torch
    prompt = format_input(probs, schema_names, cat_of, representation="continuous")
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=320)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=256, do_sample=False, num_beams=1)
    text = tok.decode(gen[0], skip_special_tokens=True)
    sections, order = parse_sections(text)
    v = validate_structure(text)
    return text, sections, v, prompt

def _generate_template(probs: dict, policy: str):
    """Deterministic template path. policy: threshold_0.5 or calibrated."""
    sys.path.insert(0, str(Path(__file__).parent))
    from generate_text_template import generate_text as template_generate, load_thresholds, DEFAULT_THRESHOLD
    from text_gen_common import parse_sections, validate_structure
    if policy == "threshold_0.5":
        text, active = template_generate(probs, threshold=DEFAULT_THRESHOLD, thresholds=None)
        strategy = "threshold_0.5"
        tmap = None
    elif policy == "calibrated":
        tmap, strategy = load_thresholds(str(TEMPLATE_THRESHOLDS_PATH))
        if tmap is None:
            raise PipelineError("MODEL_MISSING", f"calibrated thresholds not found: {TEMPLATE_THRESHOLDS_PATH} (no silent fallback to 0.5)")
        text, active = template_generate(probs, threshold=DEFAULT_THRESHOLD, thresholds=tmap)
        strategy = strategy or "calibrated"
    else:
        raise PipelineError("TEXT_GENERATION_FAILED", f"unknown template policy {policy!r}")
    sections, order = parse_sections(text)
    v = validate_structure(text)
    return text, sections, v, None, active, strategy, tmap

def _generate_template_concise(probs: dict):
    """Deterministic concise template using calibrated per-trait thresholds."""
    sys.path.insert(0, str(Path(__file__).parent))
    from generate_text_template import generate_text_concise, load_thresholds, DEFAULT_THRESHOLD
    from text_gen_common import parse_sections, validate_structure
    tmap, strategy = load_thresholds(str(TEMPLATE_THRESHOLDS_PATH))
    if tmap is None:
        raise PipelineError("MODEL_MISSING", f"calibrated thresholds not found: {TEMPLATE_THRESHOLDS_PATH} (no silent fallback to 0.5)")
    text, active, trace = generate_text_concise(probs, threshold=DEFAULT_THRESHOLD, thresholds=tmap)
    # attach threshold_policy per section
    for cat in trace:
        trace[cat]["threshold_policy"] = strategy or "per_trait"
        trace[cat]["thresholds"] = {k: float(tmap[k]) for k in trace[cat]["probabilities"]}
    sections, order = parse_sections(text)
    v = validate_structure(text)
    return text, sections, v, None, active, strategy or "per_trait", tmap, trace

def _norm_name(s):
    s = os.path.basename(str(s).strip())
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\)\s+\(", ")(", s)
    return s.lower()

def _neural_plan_realizer(model_path=None):
    """Return a callable(prompt)->text for the guarded neural realizer (plan prompt)."""
    tok, model, device = load_text_generator(model_path)
    import torch

    def _gen(prompt):
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=320)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=256, do_sample=False, num_beams=1)
        return tok.decode(out[0], skip_special_tokens=True)

    return _gen

def _generate_controlled(probs: dict, guarded: bool, model_path=None):
    """Two-stage controlled generator: planner -> realization -> validator.

    guarded=False -> deterministic realization only.
    guarded=True  -> attempt FLAN-T5 plan realization; fall back deterministically
                     (and flag it) when the semantic consistency validator rejects.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from generate_text_controlled import generate_text_controlled
    from generate_text_template import load_thresholds, DEFAULT_THRESHOLD
    from text_gen_common import parse_sections, validate_structure
    tmap, strategy = load_thresholds(str(TEMPLATE_THRESHOLDS_PATH))
    if tmap is None:
        raise PipelineError("MODEL_MISSING", f"calibrated thresholds not found: {TEMPLATE_THRESHOLDS_PATH} (no silent fallback to 0.5)")
    if guarded and model_path is None:
        ctrl_path = Path(__file__).parent / "models_text" / "flan_t5_small_controlled"
        if ctrl_path.exists():
            model_path = ctrl_path
    neural = _neural_plan_realizer(model_path) if guarded else None
    text, plan, prov = generate_text_controlled(
        probs, thresholds=tmap, threshold=DEFAULT_THRESHOLD,
        realizer="flan_guarded" if guarded else "deterministic",
        neural_generate=neural,
    )
    sections, order = parse_sections(text)
    v = validate_structure(text)
    return text, sections, v, plan, prov, strategy or "per_trait"

def _lookup_cached_row(image_path: Path):
    try:
        name = image_path.name
        full = pd.read_csv(CSV_CLEAN, encoding="utf-8-sig")
        m = full[full["image_id"] == name]
        if not m.empty:
            return m.iloc[0:1].copy()
        nk = _norm_name(name)
        for _, row in full.iterrows():
            if _norm_name(row["image_id"]) == nk:
                return row.to_frame().T.copy()
        return None
    except Exception:
        return None

def extract_features_from_image(image_path: str | Path, feature_source: str = "extract"):
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"FILE_NOT_FOUND: {p}")
    return _extract_or_cache(p, feature_source)

ERROR_CODES = ("FILE_NOT_FOUND", "INVALID_IMAGE", "DEPENDENCY_MISSING", "MODEL_MISSING", "MODEL_LOAD_FAILED", "FACE_NOT_DETECTED", "MULTIPLE_FACES_WARNING", "FEATURE_EXTRACTION_FAILED", "FEATURE_SCHEMA_MISMATCH", "SEMANTIC_INFERENCE_FAILED", "TEXT_GENERATION_FAILED", "MALFORMED_TEXT_OUTPUT", "CACHE_MISS")

FRIENDLY = {
    "FILE_NOT_FOUND": "Image file not found.",
    "INVALID_IMAGE": "The image could not be processed.",
    "FACE_NOT_DETECTED": "No face was detected. Please upload a clearer image.",
    "MULTIPLE_FACES_WARNING": "Multiple faces detected. One face was selected for analysis.",
    "FEATURE_EXTRACTION_FAILED": "The image could not be processed.",
    "CACHE_MISS": "No cached features for this image. Use real extraction.",
    "MODEL_MISSING": "The required model is unavailable.",
    "MODEL_LOAD_FAILED": "The required model is unavailable.",
    "DEPENDENCY_MISSING": "The required model is unavailable.",
    "SEMANTIC_INFERENCE_FAILED": "The image could not be processed.",
    "TEXT_GENERATION_FAILED": "The image could not be processed.",
}


def friendly_message(code: str, detail: str = "") -> str:
    base = FRIENDLY.get(code, "The image could not be processed.")
    if code == "MULTIPLE_FACES_WARNING":
        return base
    return base


class PipelineError(Exception):
    """Typed pipeline error. code in ERROR_CODES. Never downgrade DEPENDENCY_MISSING/MODEL_* to FACE_NOT_DETECTED."""
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message

def _extract_or_cache(p: Path, feature_source: str = "extract"):
    """extract (default) or cache/auto. Never silent cache fallback on extract failure."""
    if feature_source not in ("extract", "cache", "auto"):
        raise PipelineError("FEATURE_EXTRACTION_FAILED", f"unknown feature_source={feature_source}")
    if feature_source == "cache":
        cached = _lookup_cached_row(p)
        if cached is None:
            raise PipelineError("CACHE_MISS", f"no cached row for {p.name} (explicit cache mode; not falling back to extraction)")
        face_meta = {"detected": True, "detection_score": None, "face_detected": True, "multiple_faces_detected": False, "selected_face": "cached_lookup", "feature_source": "cache", "warnings": ["used cached feature row (--feature-source cache)"]}
        return cached, face_meta, face_meta["warnings"]
    # try real extractor first
    try:
        try:
            from physiognomy import extract_features as _extract
        except ImportError:
            from Source.physiognomy import extract_features as _extract
    except ImportError as e:
        if feature_source == "auto":
            cached = _lookup_cached_row(p)
            if cached is not None:
                face_meta = {"detected": True, "detection_score": None, "face_detected": True, "multiple_faces_detected": False, "selected_face": "cached_lookup", "feature_source": "cache", "warnings": [f"DEPENDENCY_MISSING, fell back to cache: {e}"]}
                return cached, face_meta, face_meta["warnings"]
        raise PipelineError("DEPENDENCY_MISSING", f"cannot import physiognomy extractor: {e}")
    try:
        df, face_meta = _extract(p)
    except FileNotFoundError:
        raise
    except ValueError as e:
        msg = str(e)
        for code in ("FACE_NOT_DETECTED", "MULTIPLE_FACES_WARNING", "INVALID_IMAGE", "FEATURE_SCHEMA_MISMATCH", "FEATURE_EXTRACTION_FAILED", "MODEL_MISSING", "MODEL_LOAD_FAILED", "CACHE_MISS"):
            if msg.startswith(code):
                raise PipelineError(code, msg)
        raise PipelineError("FEATURE_EXTRACTION_FAILED", msg)
    except RuntimeError as e:
        msg = str(e)
        if "MODEL_LOAD_FAILED" in msg or "MODEL_MISSING" in msg:
            raise PipelineError("MODEL_LOAD_FAILED", msg)
        if "heavy detector" in msg or "insightface" in msg.lower() or "facexlib" in msg.lower():
            raise PipelineError("DEPENDENCY_MISSING", msg)
        raise PipelineError("FEATURE_EXTRACTION_FAILED", msg)
    except ImportError as e:
        raise PipelineError("DEPENDENCY_MISSING", str(e))
    except PipelineError:
        raise
    except Exception as e:
        raise PipelineError("FEATURE_EXTRACTION_FAILED", str(e)) from e
    face_meta = dict(face_meta)
    face_meta["feature_source"] = "extracted"
    face_meta.setdefault("warnings", [])
    return df, face_meta, face_meta["warnings"]

def predict_image_to_text(image_path, model_path=None, output=None, features_output=None, semantic_output=None, verbose=False, feature_source="extract", generator=None):
    if generator is None:
        generator = get_default_generator()
    if generator not in GENERATOR_CHOICES:
        raise PipelineError("TEXT_GENERATION_FAILED", f"unknown generator {generator!r} choices={GENERATOR_CHOICES}")
    t0 = time.time()
    p = Path(image_path)
    t_ex0 = time.time()
    df_raw, face_meta, warnings = extract_features_from_image(p, feature_source=feature_source)
    t_ex = time.time() - t_ex0
    if "image_id" not in df_raw.columns:
        df_raw = df_raw.copy()
        df_raw["image_id"] = p.name
    csv_path = Path(features_output) if features_output else ROOT / "outputs" / f"{p.stem}_features.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_raw.to_csv(csv_path, index=False, encoding="utf-8-sig")
    t_sem0 = time.time()
    try:
        probs, trait_names = predict_semantic_from_dataframe(df_raw)
    except AssertionError as e:
        raise PipelineError("SEMANTIC_INFERENCE_FAILED", str(e)) from e
    except PipelineError:
        raise
    except Exception as e:
        raise PipelineError("SEMANTIC_INFERENCE_FAILED", str(e)) from e
    t_sem = time.time() - t_sem0
    if semantic_output:
        sp = Path(semantic_output)
        sp.parent.mkdir(parents=True, exist_ok=True)
        with open(sp, "w", encoding="utf-8") as f:
            f.write(json.dumps({"image": p.name, "probs": probs}, ensure_ascii=False) + "\n")
    t_txt0 = time.time()
    prompt = None
    active_traits = None
    threshold_policy = None
    concise_trace = None
    controlled_plan = None
    controlled_prov = None
    representation = "continuous" if generator == "flan_t5" else "template"
    if generator in ("controlled_deterministic", "controlled_flan_guarded"):
        representation = "content_plan"
    if generator == "flan_t5":
        try:
            text, sections, v, prompt = generate_text_from_semantics(probs, trait_names, model_path=str(model_path) if model_path else None)
        except PipelineError:
            raise
        except (FileNotFoundError, OSError) as e:
            raise PipelineError("MODEL_MISSING", str(e)) from e
        except Exception as e:
            raise PipelineError("TEXT_GENERATION_FAILED", str(e)) from e
        threshold_policy = "n/a"
    elif generator in ("controlled_deterministic", "controlled_flan_guarded"):
        try:
            text, sections, v, controlled_plan, controlled_prov, threshold_policy = _generate_controlled(
                probs, guarded=(generator == "controlled_flan_guarded"),
                model_path=str(model_path) if model_path else None)
        except PipelineError:
            raise
        except (FileNotFoundError, OSError) as e:
            raise PipelineError("MODEL_MISSING", str(e)) from e
        except Exception as e:
            raise PipelineError("TEXT_GENERATION_FAILED", str(e)) from e
    elif generator == "template_0.5":
        try:
            text, sections, v, prompt, active_traits, threshold_policy, _tmap = _generate_template(probs, "threshold_0.5")
        except PipelineError:
            raise
        except Exception as e:
            raise PipelineError("TEXT_GENERATION_FAILED", str(e)) from e
    elif generator == "template_calibrated":
        try:
            text, sections, v, prompt, active_traits, threshold_policy, _tmap = _generate_template(probs, "calibrated")
        except PipelineError:
            raise
        except Exception as e:
            raise PipelineError("TEXT_GENERATION_FAILED", str(e)) from e
    else:  # template_concise
        try:
            text, sections, v, prompt, active_traits, threshold_policy, _tmap, concise_trace = _generate_template_concise(probs)
        except PipelineError:
            raise
        except Exception as e:
            raise PipelineError("TEXT_GENERATION_FAILED", str(e)) from e
    if not v.get("valid"):
        raise PipelineError("MALFORMED_TEXT_OUTPUT", f"invalid sections: missing={v.get('missing_sections')} empty={v.get('empty_sections')}")
    t_txt = time.time() - t_txt0
    total = time.time() - t0
    result = {
        "success": True,
        "image": str(p),
        "feature_source": face_meta.get("feature_source", feature_source),
        "features": {"csv_path": str(csv_path), "raw_feature_count": int(df_raw.shape[1]), "extraction_time_s": round(t_ex, 3)},
        "face": {"detected": bool(face_meta.get("detected", True)), "face_detected": True, "multiple_faces_detected": bool(face_meta.get("multiple_faces_detected", False)), "detection_score": face_meta.get("detection_score"), "warnings": warnings, "selected_face": face_meta.get("selected_face"), "feature_source": face_meta.get("feature_source")},
        "semantic": {"trait_count": 45, "probabilities": probs, "inference_time_s": round(t_sem, 3)},
        "text": text,
        "validation": v,
        "timing": {"extraction_s": round(t_ex, 3), "semantic_s": round(t_sem, 3), "text_s": round(t_txt, 3), "total_s": round(total, 3)},
        "model": ("template" if generator in ("template_0.5", "template_calibrated", "template_concise", "controlled_deterministic")
                  else str(model_path or TEXT_MODEL_DEFAULT)),
        "generator": {"method": generator, "model": ("template" if generator in ("template_0.5", "template_calibrated", "template_concise", "controlled_deterministic") else str(model_path or TEXT_MODEL_DEFAULT)), "representation": representation, "threshold_policy": threshold_policy},
    }
    if active_traits is not None:
        result["generator"]["active_traits"] = active_traits
    if concise_trace is not None:
        result["generator"]["concise_trace"] = concise_trace
    if controlled_prov is not None:
        result["generator"]["content_plan"] = controlled_plan
        result["generator"]["provenance"] = controlled_prov
        result["generator"]["neural_used"] = controlled_prov.get("realizer") == "flan_t5"
        result["generator"]["fallback_used"] = bool(controlled_prov.get("fallback_used"))
    if prompt is not None:
        result["generator"]["prompt_preview"] = prompt[:800]
    result["predictions"] = {
        "personality": sections.get("Personality", ""),
        "career": sections.get("Career", ""),
        "relationships": sections.get("Relationships", ""),
        "future_destiny": sections.get("Future & Destiny", ""),
        "wealth": sections.get("Wealth", ""),
    }
    result["face"].update(face_meta)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result

def main():
    ap = argparse.ArgumentParser(description="Face image -> CSV -> text (continuous FLAN-T5-small)")
    ap.add_argument("--image", required=True, help="path to face image")
    ap.add_argument("--output", default=None, help="result JSON path")
    ap.add_argument("--features-output", default=None, help="extracted features CSV path")
    ap.add_argument("--semantic-output", default=None, help="semantic JSONL path")
    ap.add_argument("--model", default=str(TEXT_MODEL_DEFAULT), help="FLAN-T5 model dir")
    ap.add_argument("--verbose", action="store_true", help="show semantic probs and timing")
    ap.add_argument("--show-semantic", action="store_true", help="alias for --verbose")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--feature-source", choices=["extract", "cache", "auto"], default="extract", help="extract=real SCRFD (default); cache=CSV lookup only; auto=extract, cache only if deps missing")
    ap.add_argument("--generator", choices=list(GENERATOR_CHOICES), default=None, help=f"text generation method (default: {get_default_generator()})")
    args = ap.parse_args()
    gen = args.generator if args.generator is not None else get_default_generator()
    try:
        res = predict_image_to_text(args.image, model_path=args.model, output=args.output, features_output=args.features_output, semantic_output=args.semantic_output, verbose=args.verbose or args.show_semantic, feature_source=args.feature_source, generator=gen)
        print(res["text"])
        if args.verbose or args.show_semantic:
            print("\n--- semantic (45) ---")
            for k, v in sorted(res["semantic"]["probabilities"].items()):
                print(f"{k}: {v:.4f}")
            print(f"\nCSV: {res['features']['csv_path']}")
            print(f"Face: {res['face']}")
            print(f"Timing: {res['timing']} Generator: {res.get('generator')}")
            print(f"Valid: {res['validation']}")
        if args.output:
            print(f"\nWrote {args.output}")
    except PipelineError as e:
        print(f"Error: {e}", file=sys.stderr)
        print(friendly_message(e.code, e.message), file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        else:
            print("(use --debug for traceback)", file=sys.stderr)
        if e.code == "FACE_NOT_DETECTED":
            sys.exit(3)
        if e.code == "FILE_NOT_FOUND":
            sys.exit(2)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)
    except ValueError as e:
        msg = str(e)
        if "No face" in msg:
            print("No face detected.", file=sys.stderr)
            sys.exit(3)
        print(f"Error: {e}", file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        else:
            print("(use --debug for traceback)", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
