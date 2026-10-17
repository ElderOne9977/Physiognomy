# -*- coding: utf-8 -*-
"""Regression checks for Phase 3.16 shadow quality gate + robustness harness.

No model fitting, no heavy model loading. Verifies the pure quality-gate
contract and that the harness exposes its CLI stages. Run:

    python Source/test_real_world_robustness_phase316.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import image_quality_gate as Q

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
    line = ("PASS " if cond else "FAIL ") + str(name)
    try:
        print(line)
    except UnicodeEncodeError:
        print(line.encode("ascii", "backslashreplace").decode("ascii"))


def test_status_constants():
    check("statuses_distinct", len({Q.STATUS_PASS, Q.STATUS_WARN, Q.STATUS_ABSTAIN, Q.STATUS_ERROR}) == 4)
    check("policy_version_present", bool(Q.POLICY_VERSION))


def test_decode_and_inference_errors():
    r = Q.evaluate_quality_gate(decode_ok=False)
    check("decode_fail_is_error", r["status"] == Q.STATUS_ERROR)
    check("decode_fail_reason", "image_decode_failed" in r["reasons"])
    r = Q.evaluate_quality_gate(inference_error="boom")
    check("inference_error_is_error", r["status"] == Q.STATUS_ERROR)
    check("inference_error_reason", "inference_error" in r["reasons"])


def test_abstain_paths():
    check("no_face_abstain", Q.evaluate_quality_gate(detected=False)["status"] == Q.STATUS_ABSTAIN)
    check("robust_miss_abstain",
          Q.evaluate_quality_gate(detection_status="robust_miss", detected=True)["status"] == Q.STATUS_ABSTAIN)
    check("multi_ambiguous_abstain",
          Q.evaluate_quality_gate(detection_status="multi_face", detected=True)["status"] == Q.STATUS_ABSTAIN)
    check("multi_selected_not_abstain",
          Q.evaluate_quality_gate(detection_status="multi_face", detected=True,
                                  multiple_faces_selected=True)["status"] == Q.STATUS_WARN)
    check("extraction_fail_abstain",
          Q.evaluate_quality_gate(detected=True, raw_feature_count=120)["status"] == Q.STATUS_ABSTAIN)
    check("severe_missing_abstain",
          Q.evaluate_quality_gate(detected=True, raw_feature_count=486,
                                  base_required=95, base_missing=60)["status"] == Q.STATUS_ABSTAIN)


def test_warn_paths():
    check("qc_fail_warn",
          Q.evaluate_quality_gate(detected=True, raw_feature_count=486, qc_pass=False,
                                  qc_reasons="small_face", detection_score=0.9,
                                  base_required=95, base_missing=0)["status"] == Q.STATUS_WARN)
    check("missing_warn",
          Q.evaluate_quality_gate(detected=True, raw_feature_count=486, qc_pass=True,
                                  detection_score=0.9, base_required=95,
                                  base_missing=30)["status"] == Q.STATUS_WARN)
    check("low_conf_warn",
          Q.evaluate_quality_gate(detected=True, raw_feature_count=486, qc_pass=True,
                                  detection_score=0.2, base_required=95,
                                  base_missing=0)["status"] == Q.STATUS_WARN)
    check("dark_warn",
          Q.evaluate_quality_gate(detected=True, raw_feature_count=486, qc_pass=True,
                                  detection_score=0.9, base_required=95, base_missing=0,
                                  image_stats={"mean_brightness": 5.0})["status"] == Q.STATUS_WARN)
    check("bright_warn",
          Q.evaluate_quality_gate(detected=True, raw_feature_count=486, qc_pass=True,
                                  detection_score=0.9, base_required=95, base_missing=0,
                                  image_stats={"mean_brightness": 250.0})["status"] == Q.STATUS_WARN)


def test_clean_pass():
    r = Q.evaluate_quality_gate(detected=True, raw_feature_count=486, qc_pass=True,
                                detection_score=0.9, base_required=95, base_missing=0,
                                image_stats={"mean_brightness": 120.0})
    check("clean_pass", r["status"] == Q.STATUS_PASS)
    check("clean_reasons_empty", r["reasons"] == [])


def test_missing_metric_never_gates():
    # With no signals at all (detected=None), the gate must not invent a failure.
    r = Q.evaluate_quality_gate()
    check("no_signals_pass", r["status"] == Q.STATUS_PASS)
    check("no_signals_metrics_null_detection_score", r["metrics"]["detection_score"] is None)
    check("no_signals_missing_fraction_null", r["metrics"]["missing_base_fraction"] is None)


def test_threshold_mirrors():
    import physiognomy as ph
    check("min_face_side_mirrors", Q.MIN_FACE_SIDE_PX == ph.MIN_FACE_SIZE_PX)
    check("low_conf_mirrors", Q.LOW_DET_CONF == ph.LOW_CONF_WARN)


def test_determinism_and_json():
    import json
    a = Q.evaluate_quality_gate(detected=True, raw_feature_count=486, qc_pass=False,
                                qc_reasons="x", detection_score=0.3, base_required=95, base_missing=10)
    b = Q.evaluate_quality_gate(detected=True, raw_feature_count=486, qc_pass=False,
                                qc_reasons="x", detection_score=0.3, base_required=95, base_missing=10)
    check("deterministic", a == b)
    check("json_serializable", json.dumps(a) is not None)


def test_image_stats():
    gray = np.full((32, 32), 128, dtype=np.uint8)
    s = Q.compute_image_stats(gray)
    check("stats_brightness", abs(s["mean_brightness"] - 128.0) < 1e-6)
    check("stats_contrast_zero", abs(s["contrast_std"]) < 1e-6)
    check("stats_dims", s["width"] == 32 and s["height"] == 32)
    s0 = Q.compute_image_stats(None)
    check("stats_none_all_null", all(v is None for v in s0.values()))


def test_evaluate_from_face_meta():
    meta = {"detected": True, "qc_pass": True, "detection_score": 0.9,
            "multiple_faces_detected": False, "selected_face": "extracted",
            "qc_reasons": "ok", "detection_status": "normal_success"}
    r = Q.evaluate_from_face_meta(meta, raw_feature_count=486, base_required=95, base_missing=0)
    check("wrapper_pass", r["status"] == Q.STATUS_PASS)
    r2 = Q.evaluate_from_face_meta({"detected": False}, raw_feature_count=None)
    check("wrapper_no_face_abstain", r2["status"] == Q.STATUS_ABSTAIN)


def test_harness_cli():
    import validate_real_world_robustness as V
    check("seed_fixed", V.SEED == 42)
    check("frozen_artifacts_listed", len(V.FROZEN_ARTIFACTS) >= 5)
    check("exp_dir_isolated", str(V.EXP_DIR).endswith("real_world_robustness_phase316"))
    check("has_main", callable(getattr(V, "main", None)))
    for fn in ("stage_A", "stage_B", "stage_C", "stage_D", "stage_E",
               "stage_FG", "stage_H", "stage_I", "stage_J", "stage_K", "stage_L"):
        check(f"has_{fn}", callable(getattr(V, fn, None)))


def test_frozen_artifacts_untouched():
    import validate_real_world_robustness as V
    for a in V.FROZEN_ARTIFACTS:
        check(f"frozen_exists_{a.name}", a.exists())
    # the harness must never write into models_semantic/
    check("no_write_target_into_models",
          "models_semantic" not in str(V.EXP_DIR))


def main():
    test_status_constants()
    test_decode_and_inference_errors()
    test_abstain_paths()
    test_warn_paths()
    test_clean_pass()
    test_missing_metric_never_gates()
    test_threshold_mirrors()
    test_determinism_and_json()
    test_image_stats()
    test_evaluate_from_face_meta()
    test_harness_cli()
    test_frozen_artifacts_untouched()
    print(f"{PASS} PASS / {FAIL} FAIL")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()