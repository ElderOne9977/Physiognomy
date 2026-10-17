# -*- coding: utf-8 -*-
"""Lightweight UI tests using mocks (no real inference). Run:
python Source/test_demo_gradio.py
"""
from __future__ import annotations
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import ExitStack

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo_gradio as D
from predict_image_to_text import PipelineError

ROOT = Path(__file__).resolve().parents[1]
PASS = []
FAIL = []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    line = f"{'PASS' if cond else 'FAIL'} {name} {detail}"
    try:
        print(line)
    except UnicodeEncodeError:
        print(line.encode("ascii", "backslashreplace").decode("ascii"))


def fake_result(path):
    D._ensure_tmp()
    csv = D.DEMO_TMP / "demo_features.csv"
    csv.write_text("image_id,ratio\nx,1.0\n", encoding="utf-8")
    return {
        "success": True,
        "feature_source": "extracted",
        "features": {"csv_path": str(csv), "raw_feature_count": 486},
        "face": {"detected": True, "multiple_faces_detected": False,
                 "detection_score": 0.99, "selected_face": "extracted",
                 "qc_reasons": "ok", "warnings": []},
        "semantic": {"trait_count": 45,
                     "probabilities": {f"t{i}": 0.5 for i in range(45)}},
        "text": "Personality:\na\n\nCareer:\nb\n\nRelationships:\nc\n\nFuture & Destiny:\nd\n\nWealth:\ne",
        "validation": {"valid": True},
        "timing": {"extraction_s": 1, "semantic_s": 0.1, "text_s": 0.5, "total_s": 1.6},
        "predictions": {"personality": "a", "career": "b", "relationships": "c",
                        "future_destiny": "d", "wealth": "e"},
        "model": "flan_t5_small",
        "generator": {"method": "flan_t5", "model": "flan_t5_small",
                      "representation": "continuous", "threshold_policy": "n/a"},
    }


def test_1_import():
    check("ui module imports", True)


def test_2_interface():
    demo = D.build_interface()
    check("interface constructs", demo is not None)


def test_3_reuses_inference(monkeypatch=None):
    called = {}
    def spy(image_path, **kw):
        called.update(kw)
        if kw.get("output"):
            Path(kw["output"]).write_text(json.dumps(fake_result(image_path)), encoding="utf-8")
        return fake_result(image_path)
    orig = D.predict_image_to_text
    D.predict_image_to_text = spy
    D._models_ready = True
    try:
        from PIL import Image
        tmp = D.DEMO_TMP / "gradio_test_in.jpg"
        Image.new("RGB", (64, 64), (200, 100, 50)).save(tmp, "JPEG")
        out = D.analyze_image(str(tmp))
    finally:
        D.predict_image_to_text = orig
    check("inference function reused", called.get("feature_source") == "extract" and len(called) >= 1, str(called))
    check("csv exposed", out[5] is not None and str(out[5]).endswith("csv"), str(out[5]))
    check("json exposed", out[6] is not None and str(out[6]).endswith("json"), str(out[6]))


def test_4_five_sections():
    res = fake_result("x.jpg")
    sections = ["personality", "career", "relationships", "future_destiny", "wealth"]
    check("five sections mapped", all(res["predictions"][s] for s in sections))


def test_5_tech_details_separate():
    res = fake_result("x.jpg")
    tech = D._format_tech(res)
    check("tech details formatted", "feature source" in tech and "semantic probabilities" in tech)


def test_6_missing_image():
    orig = D._models_ready
    D._models_ready = True
    try:
        out = D.analyze_image(None)
    finally:
        D._models_ready = orig
    check("missing image handled", out[8] not in ("Idle.", "") and out[5] is None, out[8])


def test_7_error_mapping():
    import predict_image_to_text as P
    for code in ("FACE_NOT_DETECTED", "MODEL_MISSING", "INVALID_IMAGE", "CACHE_MISS"):
        check(f"friendly {code}", isinstance(P.friendly_message(code), str) and P.friendly_message(code))
    check("all error codes listed", set(P.ERROR_CODES) == {
        "FILE_NOT_FOUND", "INVALID_IMAGE", "DEPENDENCY_MISSING", "MODEL_MISSING",
        "MODEL_LOAD_FAILED", "FACE_NOT_DETECTED", "MULTIPLE_FACES_WARNING",
        "FEATURE_EXTRACTION_FAILED", "FEATURE_SCHEMA_MISMATCH",
        "SEMANTIC_INFERENCE_FAILED", "TEXT_GENERATION_FAILED",
        "MALFORMED_TEXT_OUTPUT", "CACHE_MISS"}, str(P.ERROR_CODES))


def test_8_no_reload():
    D._models_ready = True
    a = D.load_models_once()
    check("models not reloaded", a in (None, 0.0) or isinstance(a, float), str(a))
    D._models_ready = False


def test_9_selector_wiring():
    called = {}
    def spy(image_path, **kw):
        called.update(kw)
        if kw.get("output"):
            Path(kw["output"]).write_text(json.dumps(fake_result(image_path)), encoding="utf-8")
        return fake_result(image_path)
    orig = D.predict_image_to_text
    D.predict_image_to_text = spy
    D._models_ready = True
    try:
        from PIL import Image
        tmp = D.DEMO_TMP / "gradio_test_gen.jpg"
        Image.new("RGB", (64, 64), (200, 100, 50)).save(tmp, "JPEG")
        D.analyze_image(str(tmp), D.GENERATOR_LABELS["template_0.5"])
        check("label resolves to value", called.get("generator") == "template_0.5", str(called.get("generator")))
        D.analyze_image(str(tmp), "template_calibrated")
        check("calibrated passes through", called.get("generator") == "template_calibrated", str(called.get("generator")))
        called.clear()
        out = D.analyze_image(str(tmp), "garbage_value")
        check("unknown generator rejected before inference", not called and "unknown generator" in out[7], str(out[7]))
        check("unknown clears sections and downloads", out[:7] == ("", "", "", "", "", None, None))
    finally:
        D.predict_image_to_text = orig


def test_10_generator_metadata():
    res = fake_result("x.jpg")
    g = res["generator"]
    check("generator metadata has 4 keys",
          set(g) >= {"method", "model", "representation", "threshold_policy"}, str(g))
    tech = D._format_tech(res)
    check("tech shows threshold policy", "threshold policy" in tech, tech.splitlines()[10])


def test_11_template_modes_skip_flan():
    """Template generators must never call load_text_generator."""
    calls = {"flan": 0}
    orig_ltg = D.load_text_generator
    def spy(*a, **kw):
        calls["flan"] += 1
        raise AssertionError("FLAN must not load for template")
    D.load_text_generator = spy
    D._models_ready = False
    try:
        with patch.object(D, "load_semantic_artifacts"), patch.dict(sys.modules, {
            "physiognomy": SimpleNamespace(ensure_heavy_models=lambda: None)
        }):
            for gen in ("template_0.5", "template_calibrated", "template_concise", "controlled_deterministic"):
                D._models_ready = False
                D.load_models_once(generator=gen)
                check(f"{gen} skips FLAN", calls["flan"] == 0, str(calls))
    finally:
        D.load_text_generator = orig_ltg
        D._models_ready = False


def test_12_missing_calibrated_thresholds_errors():
    import predict_image_to_text as P
    orig = P.TEMPLATE_THRESHOLDS_PATH
    P.TEMPLATE_THRESHOLDS_PATH = Path("C:/tmp/definitely_missing_trait_thresholds.json")
    try:
        try:
            P._generate_template({"warm": 0.6}, "calibrated")
            check("missing calibrated raises", False, "no exception")
        except PipelineError as e:
            check("missing calibrated raises", e.code == "MODEL_MISSING", e.code)
            check("no silent 0.5 fallback", "no silent fallback" in e.message, e.message)
    finally:
        P.TEMPLATE_THRESHOLDS_PATH = orig


def test_12b_missing_concise_thresholds_errors():
    import predict_image_to_text as P
    orig = P.TEMPLATE_THRESHOLDS_PATH
    P.TEMPLATE_THRESHOLDS_PATH = Path("C:/tmp/definitely_missing_trait_thresholds.json")
    try:
        try:
            P._generate_template_concise({"warm": 0.6})
            check("missing concise raises", False, "no exception")
        except PipelineError as e:
            check("missing concise raises", e.code == "MODEL_MISSING", e.code)
            check("no silent fallback concise", "no silent fallback" in e.message, e.message)
    finally:
        P.TEMPLATE_THRESHOLDS_PATH = orig

def test_13_explicit_concept_fixture():
    """Caller-supplied concepts; no face-derived vectors, FLAN, or extraction."""
    import predict_image_to_text as P
    from text_gen_common import load_schema
    real = dict.fromkeys(load_schema()[0], 0.9)
    for policy, gen in (("threshold_0.5", "template_0.5"), ("calibrated", "template_calibrated")):
        text, sections, v, prompt, active, strategy, tmap = P._generate_template(real, policy)
        check(f"real template valid 5-section {gen}", v["valid"], str(v))
        check(f"real template active traits {gen}", isinstance(active, list) and len(active) >= 0, str(len(active)))


def test_14_generator_choices():
    import predict_image_to_text as P
    expected = {"flan_t5", "template_0.5", "template_calibrated", "template_concise",
                "controlled_deterministic", "controlled_flan_guarded"}
    check("6 generator choices", set(P.GENERATOR_CHOICES) == expected, str(P.GENERATOR_CHOICES))
    check("all choices labeled", set(P.GENERATOR_CHOICES) <= set(D.GENERATOR_LABELS), str(sorted(D.GENERATOR_LABELS)))
    check("thresholds artifact exists", P.TEMPLATE_THRESHOLDS_PATH.exists(), str(P.TEMPLATE_THRESHOLDS_PATH))


def test_15_dropdown_present():
    demo = D.build_interface()
    import gradio as gr
    blocks = getattr(demo, "blocks", {}) or {}
    dd = [c for c in blocks.values() if isinstance(c, gr.Dropdown)]
    check("dropdown exists", len(dd) >= 1, str(len(dd)))
    if dd:
        check("dropdown label", dd[0].label == "Text Generation Method", str(dd[0].label))
        check("dropdown matches generator choices", len(dd[0].choices) == len(D.GENERATOR_VALUES), str(len(dd[0].choices)))
        check("dropdown default flan", dd[0].value == D.GENERATOR_LABELS["flan_t5"], str(dd[0].value))
        check("flan labeled experimental", "experimental" in D.GENERATOR_LABELS["flan_t5"], D.GENERATOR_LABELS["flan_t5"])
        check("calibrated labeled recommended", "recommended" in D.GENERATOR_LABELS["template_calibrated"], D.GENERATOR_LABELS["template_calibrated"])


def test_16_pipeline_dispatch():
    """Exercise routing and real CSV/JSON serialization with all model work mocked."""
    import pandas as pd
    import predict_image_to_text as P
    from text_gen_common import load_schema, CATEGORY_TITLES
    probs = dict.fromkeys(load_schema()[0], 0.9)
    sections = dict(zip(CATEGORY_TITLES.values(), ("a", "b", "c", "d", "e")))
    text = "\n\n".join(f"{k}: {v}" for k, v in sections.items())
    valid = {"valid": True}
    base = (text, sections, valid, None, [], "per_trait", {})
    with tempfile.TemporaryDirectory() as td, ExitStack() as stack:
        extract = stack.enter_context(patch.object(P, "extract_features_from_image", return_value=(
            pd.DataFrame([{"image_id": "fixture", "ratio": 1.0}]), {"detected": True}, [])))
        sem = stack.enter_context(patch.object(P, "predict_semantic_from_dataframe", return_value=(probs, list(probs))))
        flan = stack.enter_context(patch.object(P, "generate_text_from_semantics", return_value=(text, sections, valid, "fixture prompt")))
        template = stack.enter_context(patch.object(P, "_generate_template", return_value=base))
        concise = stack.enter_context(patch.object(P, "_generate_template_concise", return_value=(*base, {})))
        controlled = stack.enter_context(patch.object(P, "_generate_controlled"))
        for gen in P.GENERATOR_CHOICES:
            for mock in (flan, template, concise, controlled):
                mock.reset_mock()
            controlled.return_value = (text, sections, valid, {"selected_all": []}, {
                "realizer": "flan_t5_rejected_fallback" if gen == "controlled_flan_guarded" else "deterministic",
                "fallback_used": True, "neural_attempted": gen == "controlled_flan_guarded",
            }, "per_trait")
            csv_path, json_path = Path(td) / f"{gen}.csv", Path(td) / f"{gen}.json"
            result = P.predict_image_to_text("fixture.jpg", generator=gen,
                features_output=csv_path, output=json_path)
            expected = (gen == "flan_t5", gen in ("template_0.5", "template_calibrated"),
                        gen == "template_concise", gen.startswith("controlled_"))
            check(f"{gen} dispatch", tuple(m.called for m in (flan, template, concise, controlled)) == expected)
            invoked = next(m for m in (flan, template, concise, controlled) if m.called)
            check(f"{gen} vector preserved", invoked.call_args.args[0] is probs)
            check(f"{gen} five sections and metadata", len(result["predictions"]) == 5 and result["generator"]["method"] == gen)
            check(f"{gen} CSV/JSON serialized", json.loads(json_path.read_text(encoding="utf-8")) == result
                  and pd.read_csv(csv_path).iloc[0]["ratio"] == 1.0)
            if gen.startswith("controlled_"):
                check(f"{gen} guarded parameter", controlled.call_args.kwargs["guarded"] == (gen == "controlled_flan_guarded"))
                check(f"{gen} fallback is not neural", result["generator"]["fallback_used"] and not result["generator"]["neural_used"])
        extract.reset_mock()
        sem.reset_mock()
        bad_csv = Path(td) / "invalid.csv"
        try:
            P.predict_image_to_text("fixture.jpg", generator="invalid", features_output=bad_csv)
            check("API invalid rejected before extraction", False)
        except PipelineError as e:
            check("API invalid rejected before extraction", e.code == "TEXT_GENERATION_FAILED" and not extract.called and not sem.called and not bad_csv.exists())


def test_17_ui_requests_and_provenance():
    """UI selector, request isolation, and error clearing use a mock backend."""
    from PIL import Image
    source = D.DEMO_TMP / "request_fixture.jpg"
    Image.new("RGB", (32, 32)).save(source)
    paths = []
    def backend(image_path, **kw):
        paths.append(str(image_path))
        result = fake_result(image_path)
        result["predictions"]["personality"] = str(len(paths))
        result["generator"]["method"] = kw["generator"]
        Path(kw["output"]).write_text(json.dumps(result), encoding="utf-8")
        return result
    with patch.object(D, "load_models_once"), patch.object(D, "predict_image_to_text", side_effect=backend) as model:
        for gen in D.GENERATOR_VALUES:
            out = D.analyze_image(source, D.GENERATOR_LABELS[gen])
            check(f"UI selects {gen}", model.call_args.kwargs["generator"] == gen and f"generator: {gen}" in out[7])
            check(f"UI fresh result {gen}", out[0] == str(len(paths)) and out[6] is not None)
        check("unique upload paths per request", len(set(paths)) == len(paths))
        model.side_effect = PipelineError("INVALID_IMAGE", "fixture error")
        out = D.analyze_image(source)
        check("error clears previous result", out[:7] == ("", "", "", "", "", None, None) and "INVALID_IMAGE" in out[7])
    tech = D._format_tech({"generator": {"method": "controlled_flan_guarded", "neural_used": False,
        "fallback_used": True, "provenance": {"realizer": "flan_t5_rejected_fallback", "neural_attempted": True}}})
    check("UI fallback provenance truthful", "neural generation used: False" in tech and "deterministic fallback used: True" in tech)


def test_18_resource_limits_and_readiness():
    import os
    import time
    import numpy as np

    # 1. Readiness status structure
    ready = D.check_readiness()
    check("readiness status key", "status" in ready and "models_ready" in ready)
    check("readiness upload limits", "upload_limits" in ready and ready["upload_limits"]["max_upload_size_bytes"] == D.MAX_UPLOAD_SIZE_BYTES)
    check("readiness ttl present", ready["upload_limits"]["demo_file_ttl_seconds"] == D.DEMO_FILE_TTL_SECONDS)

    # 2. TTL Cleanup of old files
    old_file = D.DEMO_TMP / "old_expired.jpg"
    old_file.write_text("old", encoding="utf-8")
    past_mtime = time.time() - (D.DEMO_FILE_TTL_SECONDS + 500)
    os.utime(old_file, (past_mtime, past_mtime))

    fresh_file = D.DEMO_TMP / "fresh_active.jpg"
    fresh_file.write_text("fresh", encoding="utf-8")

    removed = D.cleanup_old_temp_files(D.DEMO_FILE_TTL_SECONDS)
    check("ttl cleanup removed expired file", removed >= 1 and not old_file.exists())
    check("ttl cleanup kept fresh file", fresh_file.exists())
    if fresh_file.exists():
        fresh_file.unlink()

    # 3. Pixel limit on image upload
    oversized_arr = np.zeros((4500, 4500, 3), dtype=np.uint8)
    try:
        D._save_upload(oversized_arr)
        check("pixel limit enforced", False, "expected error")
    except PipelineError as e:
        check("pixel limit enforced", e.code == "INVALID_IMAGE" and "resolution" in str(e))

    # 4. Pretrained weight inventory
    import inventory_weights
    audit = inventory_weights.audit_weights()
    check("inventory weights audit PASS", audit["status"] == "PASS")


def test_19_default_generator_env_override():
    import os
    import predict_image_to_text as P
    import gradio as gr

    # 1. Default without env var is flan_t5
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("PHYSIO_DEFAULT_GENERATOR", None)
        check("default gen without env is flan_t5 (D)", D.get_default_generator() == "flan_t5")
        check("default gen without env is flan_t5 (P)", P.get_default_generator() == "flan_t5")
        ready = D.check_readiness()
        check("readiness default gen is flan_t5", ready["default_generator"] == "flan_t5")

    # 2. Valid override via PHYSIO_DEFAULT_GENERATOR
    for candidate in ("template_calibrated", "template_concise", "template_0.5", "controlled_deterministic"):
        with patch.dict(os.environ, {"PHYSIO_DEFAULT_GENERATOR": candidate}):
            check(f"override {candidate} (D)", D.get_default_generator() == candidate)
            check(f"override {candidate} (P)", P.get_default_generator() == candidate)
            ready = D.check_readiness()
            check(f"readiness reflects {candidate}", ready["default_generator"] == candidate)

    # 3. Invalid override falls back safely to flan_t5
    with patch.dict(os.environ, {"PHYSIO_DEFAULT_GENERATOR": "invalid_mode_xyz"}):
        check("invalid override falls back to flan_t5 (D)", D.get_default_generator() == "flan_t5")
        check("invalid override falls back to flan_t5 (P)", P.get_default_generator() == "flan_t5")
        ready = D.check_readiness()
        check("readiness invalid fallback is flan_t5", ready["default_generator"] == "flan_t5")

    # 4. Interface dropdown adapts default value to PHYSIO_DEFAULT_GENERATOR
    with patch.dict(os.environ, {"PHYSIO_DEFAULT_GENERATOR": "template_calibrated"}):
        demo = D.build_interface()
        blocks = getattr(demo, "blocks", {}) or {}
        dd = [c for c in blocks.values() if isinstance(c, gr.Dropdown)]
        check("dropdown default matches template_calibrated", dd and dd[0].value == D.GENERATOR_LABELS["template_calibrated"])

    # 5. Lazy loading verification: template default does not load FLAN
    calls = {"flan": 0}
    orig_ltg = D.load_text_generator
    def spy(*a, **kw):
        calls["flan"] += 1
        raise AssertionError("FLAN must not load when template default is used")
    D.load_text_generator = spy
    D._models_ready = False
    try:
        with patch.dict(os.environ, {"PHYSIO_DEFAULT_GENERATOR": "template_calibrated"}), \
             patch.object(D, "load_semantic_artifacts"), \
             patch.dict(sys.modules, {"physiognomy": SimpleNamespace(ensure_heavy_models=lambda: None)}):
            D.load_models_once()
            check("template default skips FLAN loading", calls["flan"] == 0)
    finally:
        D.load_text_generator = orig_ltg
        D._models_ready = False


def run_tests():
    test_1_import()
    test_2_interface()
    test_3_reuses_inference()
    test_4_five_sections()
    test_5_tech_details_separate()
    test_6_missing_image()
    test_7_error_mapping()
    test_8_no_reload()
    test_9_selector_wiring()
    test_10_generator_metadata()
    test_11_template_modes_skip_flan()
    test_12_missing_calibrated_thresholds_errors()
    test_12b_missing_concise_thresholds_errors()
    test_13_explicit_concept_fixture()
    test_14_generator_choices()
    test_15_dropdown_present()
    test_16_pipeline_dispatch()
    test_17_ui_requests_and_provenance()
    test_18_resource_limits_and_readiness()
    test_19_default_generator_env_override()
    print(f"\nPASS {len(PASS)} FAIL {len(FAIL)}")
    return 0 if not FAIL else 1


def main():
    with tempfile.TemporaryDirectory(prefix="physiognomy_demo_test_") as td, patch.object(D, "DEMO_TMP", Path(td)):
        return run_tests()


if __name__ == "__main__":
    sys.exit(main())
