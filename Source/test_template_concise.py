# -*- coding: utf-8 -*-
"""Regression checks for template_concise. No FLAN, no retrain."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).parent
sys.path.insert(0, str(SRC))

from generate_text_template import (  # noqa: E402
    CONCISE_FALLBACK,
    TRAIT_PHRASES,
    generate_text,
    generate_text_concise,
    load_thresholds,
)
from predict_image_to_text import (  # noqa: E402
    GENERATOR_CHOICES,
    PipelineError,
    TEMPLATE_THRESHOLDS_PATH,
    _generate_template,
    _generate_template_concise,
)
from text_gen_common import parse_sections, validate_structure  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

PASS = []
FAIL = []


def check(name, cond, detail=""):
    line = f"{'PASS' if cond else 'FAIL'} {name} {detail}"
    try:
        print(line)
    except UnicodeEncodeError:
        print(line.encode("ascii", "backslashreplace").decode("ascii"))
    (PASS if cond else FAIL).append(name)


def _fixture_probs():
    # Explicit concept fixture, independent of facial-image inference artifacts.
    return dict.fromkeys(TRAIT_PHRASES, 0.9)


def test_deterministic():
    probs = {"warm": 0.8, "calm": 0.7, "confident": 0.6, "loyal": 0.9}
    t1, a1, tr1 = generate_text_concise(probs, threshold=0.5, thresholds=None)
    t2, a2, tr2 = generate_text_concise(probs, threshold=0.5, thresholds=None)
    check("identical input identical text", t1 == t2)
    check("identical active", a1 == a2)
    check("identical selected personality", tr1["personality"]["selected_traits"] == tr2["personality"]["selected_traits"])


def test_content_follows_selected():
    a = {"warm": 0.9, "calm": 0.1, "loyal": 0.9, "patient": 0.1}
    b = {"warm": 0.1, "calm": 0.9, "loyal": 0.1, "patient": 0.9}
    ta, _, tra = generate_text_concise(a, threshold=0.5, thresholds=None)
    tb, _, trb = generate_text_concise(b, threshold=0.5, thresholds=None)
    check("different selected change text", ta != tb)
    check("warm realized in a", "warm" in " ".join(tra["personality"]["realized_concepts"]))
    check("calm not in a selected", "calm" not in tra["personality"]["selected_traits"])
    check("calm realized in b", "calm" in " ".join(trb["personality"]["realized_concepts"]))


def test_no_inactive_introduced():
    probs = {"warm": 0.9, "calm": 0.1, "loyal": 0.8}
    text, active, trace = generate_text_concise(probs, threshold=0.5, thresholds=None)
    selected = [t for cat in trace for t in trace[cat]["selected_traits"]]
    check("selected subset of active", set(selected) <= set(active))
    check("inactive calm omitted", "calm" not in selected)
    check("no fabricated trait", all(t in TRAIT_PHRASES for t in selected))
    check("text has 5 headings", validate_structure(text)["valid"])


def test_fallback():
    text, active, trace = generate_text_concise({"warm": 0.1}, threshold=0.5, thresholds=None)
    check("empty active", active == [])
    for cat in trace:
        check(f"fallback {cat}", trace[cat]["fallback_used"] is True)
        check(f"fallback text {cat}", trace[cat]["section_text"] == CONCISE_FALLBACK)
    check("5 sections remain", validate_structure(text)["valid"])


def test_existing_template_unchanged():
    probs = {"warm": 0.8, "calm": 0.7, "loyal": 0.7, "patient": 0.6}
    t_old, a_old = generate_text(probs, threshold=0.5, thresholds=None)
    t_old2, a_old2 = generate_text(probs, threshold=0.5, thresholds=None)
    check("legacy generate_text stable", t_old == t_old2 and a_old == a_old2)
    check("legacy still semicolon join", "; " in t_old)
    t_new, _, _ = generate_text_concise(probs, threshold=0.5, thresholds=None)
    check("concise differs from legacy", t_new != t_old)


def test_pipeline_concise():
    probs = _fixture_probs()
    text, sections, v, prompt, active, strategy, tmap, trace = _generate_template_concise(probs)
    check("pipeline valid 5-section", v["valid"], str(v))
    check("pipeline no prompt", prompt is None)
    check("pipeline strategy per_trait", strategy == "per_trait", str(strategy))
    check("trace 5 cats", set(trace) == {"personality", "career", "relationships", "future_destiny", "wealth"})
    selected = {t for cat in trace for t in trace[cat]["selected_traits"]}
    check("no inactive in selected", selected <= set(active))
    for cat, rec in trace.items():
        check(f"{cat} selected <=4", len(rec["selected_traits"]) <= 4, str(len(rec["selected_traits"])))
        if rec["active_traits"]:
            check(f"{cat} not empty when active", rec["fallback_used"] is False)


def test_calibrated_path_untouched():
    probs = _fixture_probs()
    t1, s1, v1, p1, a1, st1, tm1 = _generate_template(probs, "calibrated")
    t2, s2, v2, p2, a2, st2, tm2 = _generate_template(probs, "calibrated")
    check("calibrated deterministic", t1 == t2)
    check("calibrated still semicolon", "; " in t1)
    tc, *_ = _generate_template_concise(probs)
    check("concise not equal calibrated", tc != t1)


def test_choices_and_missing():
    check("choices include concise", "template_concise" in GENERATOR_CHOICES)
    orig = TEMPLATE_THRESHOLDS_PATH
    import predict_image_to_text as P
    P.TEMPLATE_THRESHOLDS_PATH = Path("C:/tmp/definitely_missing_trait_thresholds.json")
    try:
        try:
            P._generate_template_concise({"warm": 0.9})
            check("missing thresholds raises", False, "no exception")
        except PipelineError as e:
            check("missing thresholds MODEL_MISSING", e.code == "MODEL_MISSING", e.code)
    finally:
        P.TEMPLATE_THRESHOLDS_PATH = orig


def test_synonym_dedup():
    probs = {"reliable": 0.9, "dependable": 0.89, "warm": 0.8, "sociable": 0.79}
    text, active, trace = generate_text_concise(probs, threshold=0.5, thresholds=None)
    sel = set(trace["personality"]["selected_traits"])
    check("dedup reliable/dependable", not ({"reliable", "dependable"} <= sel), str(sel))
    check("dedup warm/sociable", not ({"warm", "sociable"} <= sel), str(sel))


def main():
    test_deterministic()
    test_content_follows_selected()
    test_no_inactive_introduced()
    test_fallback()
    test_existing_template_unchanged()
    test_pipeline_concise()
    test_calibrated_path_untouched()
    test_choices_and_missing()
    test_synonym_dedup()
    print(f"\nPASS {len(PASS)} FAIL {len(FAIL)}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
