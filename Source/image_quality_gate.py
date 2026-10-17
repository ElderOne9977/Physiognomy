# -*- coding: utf-8 -*-
"""Shadow-mode image quality gate for the Physiognomy pipeline.

Pure, deterministic, dependency-light. Turns signals already produced by the
pipeline (face QC, detection status, feature missingness) plus optional
lightweight image statistics into a quality status:

    PASS     image passed required technical quality checks
    WARN     usable image, one or more reliability warnings should be shown
    ABSTAIN  input quality or ambiguity prevents a responsible interpretation
    ERROR    technical failure prevents inference

SHADOW MODE CONTRACT
--------------------
This module NEVER blocks inference, NEVER alters semantic probabilities and
NEVER alters generated text. Callers may attach the returned dict to their
result under ``quality_gate``. Production enforcement is out of scope until a
separate approval backed by measured evidence.

Threshold provenance
--------------------
Every threshold below is *experimental* and was chosen from the Phase 3.16
diagnostics (`Source/experiments/real_world_robustness_phase316/`), not from a
validated ground-truth failure set. They are documented so they can be revised
once annotated technical-failure labels exist. ``MIN_FACE_SIDE_PX`` and
``LOW_DET_CONF`` intentionally mirror ``physiognomy.MIN_FACE_SIZE_PX`` and
``physiognomy.LOW_CONF_WARN`` so the gate cannot disagree with existing QC.

A blur score (variance of Laplacian) is NOT a universally calibrated measure of
image quality, and detection confidence is NOT a validated probability that
semantic predictions are correct. Blur is therefore recorded as a diagnostic
metric but is deliberately NOT used to gate.
"""
from __future__ import annotations

import math

STATUS_PASS = "PASS"
STATUS_WARN = "WARN"
STATUS_ABSTAIN = "ABSTAIN"
STATUS_ERROR = "ERROR"

POLICY_VERSION = "phase316-shadow-v1"

# --- QC mirrors (must match Source/physiognomy.py) ---------------------------
MIN_FACE_SIDE_PX = 60      # physiognomy.MIN_FACE_SIZE_PX
LOW_DET_CONF = 0.45        # physiognomy.LOW_CONF_WARN

# --- Experimental gate thresholds (Phase 3.16 diagnostics) ------------------
# Fraction of the 95 selected base features that are missing/non-numeric.
MISSING_BASE_WARN_FRACTION = 0.20     # observed: qc_pass=True yet 19/95 missing
MISSING_BASE_ABSTAIN_FRACTION = 0.50  # severe; extractor itself fails <400 raw cols
RAW_COLS_MIN = 400                    # mirrors physiognomy FEATURE_SCHEMA_MISMATCH

# Extreme global brightness (0..255) is a weak, experimental signal only.
EXTREME_DARK_MEAN = 25.0
EXTREME_BRIGHT_MEAN = 235.0


def _is_finite_number(x):
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def compute_image_stats(bgr):
    """Lightweight, unit-documented image statistics from a BGR uint8 array.

    Returns a dict with:
      mean_brightness  mean grayscale intensity (0..255)
      contrast_std     grayscale standard deviation (0..~127)
      laplacian_var    variance of Laplacian (blur proxy; NOT calibrated)
      width/height     pixel dimensions
    Values that cannot be computed are returned as None (never NaN/inf).
    """
    if bgr is None:
        return {
            "mean_brightness": None,
            "contrast_std": None,
            "laplacian_var": None,
            "width": None,
            "height": None,
        }
    try:
        import cv2
        import numpy as np

        arr = np.asarray(bgr)
        if arr.ndim == 2:
            gray = arr
        else:
            gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        h, w = gray.shape[:2]
        return {
            "mean_brightness": round(float(gray.mean()), 4),
            "contrast_std": round(float(gray.std()), 4),
            "laplacian_var": round(float(lap.var()), 4),
            "width": int(w),
            "height": int(h),
        }
    except Exception:
        return {
            "mean_brightness": None,
            "contrast_std": None,
            "laplacian_var": None,
            "width": None,
            "height": None,
        }


def _normalize_qc_reasons(qc_reasons):
    if qc_reasons is None:
        return []
    if isinstance(qc_reasons, (list, tuple, set)):
        return [str(r) for r in qc_reasons if str(r) and str(r) != "ok"]
    text = str(qc_reasons)
    if text.strip() in ("", "ok"):
        return []
    return [p for p in text.split("|") if p]


def evaluate_quality_gate(
    *,
    decode_ok=True,
    detection_status=None,
    detected=None,
    multiple_faces_detected=False,
    multiple_faces_selected=False,
    detection_score=None,
    qc_pass=None,
    qc_reasons=None,
    raw_feature_count=None,
    base_required=95,
    base_missing=None,
    image_stats=None,
    inference_error=None,
):
    """Compute the shadow quality status. Deterministic; no side effects.

    Only signals explicitly provided are used; missing metrics never trigger a
    rule (they are recorded in ``metrics`` as null). Returns a JSON-serializable
    dict with finite values only.
    """
    reasons = []
    qc_list = _normalize_qc_reasons(qc_reasons)

    metrics = {
        "decode_ok": bool(decode_ok),
        "detection_status": detection_status,
        "detected": detected,
        "multiple_faces_detected": bool(multiple_faces_detected),
        "multiple_faces_selected": bool(multiple_faces_selected),
        "detection_score": float(detection_score) if _is_finite_number(detection_score) else None,
        "qc_pass": qc_pass,
        "qc_reasons": qc_list,
        "raw_feature_count": int(raw_feature_count) if isinstance(raw_feature_count, (int, float)) and _is_finite_number(raw_feature_count) else None,
        "base_required": int(base_required) if _is_finite_number(base_required) else None,
        "base_missing": int(base_missing) if isinstance(base_missing, (int, float)) and _is_finite_number(base_missing) else None,
        "missing_base_fraction": None,
        "inference_error": (str(inference_error)[:200] if inference_error else None),
        "image_stats": image_stats if isinstance(image_stats, dict) else None,
    }

    bm = metrics["base_missing"]
    br = metrics["base_required"]
    if bm is not None and br:
        metrics["missing_base_fraction"] = round(float(bm) / float(br), 4)

    # ---- ERROR: hard technical failures ------------------------------------
    if decode_ok is False:
        reasons.append("image_decode_failed")
        return _result(STATUS_ERROR, reasons, metrics, "error")
    if inference_error:
        reasons.append("inference_error")
        return _result(STATUS_ERROR, reasons, metrics, "error")

    # ---- ABSTAIN: no responsible interpretation possible --------------------
    if detected is False:
        reasons.append("no_face_detected")
        return _result(STATUS_ABSTAIN, reasons, metrics, "abstain")
    if detection_status in ("robust_miss",):
        reasons.append("no_face_detected")
        return _result(STATUS_ABSTAIN, reasons, metrics, "abstain")
    if detection_status == "multi_face" and not multiple_faces_selected:
        reasons.append("multiple_faces_ambiguous")
        return _result(STATUS_ABSTAIN, reasons, metrics, "abstain")
    if metrics["raw_feature_count"] is not None and metrics["raw_feature_count"] < RAW_COLS_MIN:
        reasons.append("feature_extraction_failure")
        return _result(STATUS_ABSTAIN, reasons, metrics, "abstain")
    frac = metrics["missing_base_fraction"]
    if frac is not None and frac >= MISSING_BASE_ABSTAIN_FRACTION:
        reasons.append("severe_feature_missingness")
        return _result(STATUS_ABSTAIN, reasons, metrics, "abstain")

    # ---- WARN: usable but with reliability warnings -------------------------
    warn = False
    if multiple_faces_detected or multiple_faces_selected:
        reasons.append("multiple_faces_selected")
        warn = True
    if qc_pass is False or qc_list:
        for r in qc_list or ["qc_failed"]:
            reasons.append(f"qc_{r}")
        warn = True
    if _is_finite_number(detection_score) and float(detection_score) < LOW_DET_CONF:
        reasons.append("low_detection_confidence")
        warn = True
    if frac is not None and MISSING_BASE_WARN_FRACTION <= frac < MISSING_BASE_ABSTAIN_FRACTION:
        reasons.append("feature_missingness")
        warn = True
    if isinstance(image_stats, dict):
        mb = image_stats.get("mean_brightness")
        if _is_finite_number(mb):
            if float(mb) < EXTREME_DARK_MEAN:
                reasons.append("extreme_darkness")
                warn = True
            elif float(mb) > EXTREME_BRIGHT_MEAN:
                reasons.append("extreme_brightness")
                warn = True

    if warn:
        return _result(STATUS_WARN, reasons, metrics, "proceed_with_warning")
    return _result(STATUS_PASS, reasons, metrics, "proceed")


def _result(status, reasons, metrics, action):
    return {
        "policy": "shadow",
        "policy_version": POLICY_VERSION,
        "status": status,
        "reasons": list(reasons),
        "metrics": metrics,
        "recommended_action": action,
    }


def evaluate_from_face_meta(face_meta, *, raw_feature_count=None, base_required=95,
                            base_missing=None, image_stats=None):
    """Convenience wrapper around ``evaluate_quality_gate`` for pipeline metadata."""
    face_meta = face_meta or {}
    return evaluate_quality_gate(
        decode_ok=True,
        detection_status=face_meta.get("detection_status"),
        detected=face_meta.get("detected"),
        multiple_faces_detected=bool(face_meta.get("multiple_faces_detected")),
        multiple_faces_selected=(face_meta.get("selected_face") == "largest_of_multiple"),
        detection_score=face_meta.get("detection_score"),
        qc_pass=face_meta.get("qc_pass"),
        qc_reasons=face_meta.get("qc_reasons"),
        raw_feature_count=raw_feature_count,
        base_required=base_required,
        base_missing=base_missing,
        image_stats=image_stats,
    )