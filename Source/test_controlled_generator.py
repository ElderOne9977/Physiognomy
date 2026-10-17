# -*- coding: utf-8 -*-
"""Targeted tests for the controlled two-stage generator. No FLAN required.

Covers: semantic input alignment, correct prompt construction, category-specific
conditioning, counterfactual sensitivity, unsupported concept detection,
deterministic fallback, generator provenance, five-section validation.

Run: python Source/test_controlled_generator.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).parent
sys.path.insert(0, str(SRC))

from generate_text_controlled import (  # noqa: E402
    build_content_plan,
    build_plan_prompt,
    extract_concepts,
    generate_text_controlled,
    plan_faithfulness,
    realize_deterministic,
    validate_against_plan,
)
from generate_text_template import TRAIT_PHRASES, load_thresholds  # noqa: E402
from text_gen_common import parse_sections, validate_structure  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

PASS, FAIL = [], []


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


def test_plan_alignment_and_bounds():
    """Planner selects only supported concepts and respects category ownership."""
    probs = {"warm": 0.9, "calm": 0.1, "confident": 0.8, "leadership": 0.85,
             "loyal": 0.7, "financial_stability": 0.6}
    plan = build_content_plan(probs, threshold=0.5)
    check("plan has 5 categories", set(plan["categories"]) ==
          {"personality", "career", "relationships", "future_destiny", "wealth"})
    selected = set(plan["selected_all"])
    check("selected subset of active", selected <= {t for t, p in probs.items() if p >= 0.5},
          str(selected))
    check("inactive calm excluded", "calm" not in selected)
    # category ownership
    ok = all(TRAIT_PHRASES[t][0] == cat
             for cat, rec in plan["categories"].items() for t in rec["selected"])
    check("category ownership preserved", ok)
    check("omitted recorded", plan["categories"]["personality"]["omitted"] == []
          or isinstance(plan["categories"]["personality"]["omitted"], list))


def test_empty_section_handling():
    probs = {"warm": 0.9}
    plan = build_content_plan(probs, threshold=0.5)
    empty = [c for c, r in plan["categories"].items() if r["empty"]]
    check("empty sections flagged", set(empty) == {"career", "relationships",
                                                   "future_destiny", "wealth"}, str(empty))
    text = realize_deterministic(plan)
    check("empty section uses fallback text", validate_structure(text)["valid"])


def test_determinism():
    probs = _fixture_probs()
    t1, p1, v1 = generate_text_controlled(probs)
    t2, p2, v2 = generate_text_controlled(probs)
    check("identical input identical text", t1 == t2)
    check("identical input identical plan", json.dumps(p1, sort_keys=True) == json.dumps(p2, sort_keys=True))
    check("deterministic realizer provenance", v1["realizer"] == "deterministic" and v1["fallback_used"] is True)


def test_counterfactual_sensitivity():
    """Changing only the semantic input must change the output text."""
    a = {"warm": 0.9, "calm": 0.1, "loyal": 0.9, "patient": 0.1}
    b = {"warm": 0.1, "calm": 0.9, "loyal": 0.1, "patient": 0.9}
    ta, _, _ = generate_text_controlled(a)
    tb, _, _ = generate_text_controlled(b)
    check("different vectors different text", ta != tb)
    check("warm realized in a", "warm" in extract_concepts(ta))
    check("calm realized in b", "calm" in extract_concepts(tb))
    check("calm absent in a", "calm" not in extract_concepts(ta))


def test_category_conditioning():
    """A trait must appear only in its own category section."""
    probs = {"warm": 0.9, "leadership": 0.9, "loyal": 0.9}
    text, plan, _ = generate_text_controlled(probs)
    secs, _ = parse_sections(text)
    check("warm in Personality", "warm" in extract_concepts(secs.get("Personality", "")))
    check("leadership in Career", "leadership" in extract_concepts(secs.get("Career", "")))
    check("loyal in Relationships", "loyal" in extract_concepts(secs.get("Relationships", "")))
    check("leadership not in Personality", "leadership" not in extract_concepts(secs.get("Personality", "")))
    check("warm not in Wealth", "warm" not in extract_concepts(secs.get("Wealth", "")))


def test_unsupported_detection_and_validator():
    probs = {"warm": 0.9, "leadership": 0.9}
    plan = build_content_plan(probs, threshold=0.5)
    # unsupported: mentions wealth concepts not in plan
    bad = ("Personality: warm. Career: leadership. Relationships: x. "
           "Future Destiny: y. Wealth: financial stability.")
    ok, reasons, _ = validate_against_plan(bad, plan)
    check("validator rejects unsupported", not ok, str(reasons))
    check("reason names unsupported", any(r.startswith("unsupported:") for r in reasons), str(reasons))
    good = "Personality: warm. Career: leadership. Relationships: x. Future Destiny: y. Wealth: z."
    ok2, reasons2, _ = validate_against_plan(good, plan)
    check("validator accepts faithful plan", ok2, str(reasons2))


def test_lexicon_collision_is_not_unsupported():
    """The canonical phrase bank is not injective; collisions must not fail validation."""
    probs = {"opportunities": 0.9}
    plan = build_content_plan(probs, threshold=0.5)
    # canonical phrase for opportunities contains the word "consistent"
    text = realize_deterministic(plan)
    ok, reasons, _ = validate_against_plan(text, plan)
    check("collision does not fail validation", ok, str(reasons))
    check("collision reported", any(r.startswith("collision:") for r in reasons), str(reasons))
    f = plan_faithfulness(text, plan)
    check("collision classified as collision", "consistent" in f["lexicon_collisions"], str(f))
    check("no genuine unsupported from collision", f["genuine_unsupported"] == [], str(f))


def test_guarded_fallback_provenance():
    probs = {"warm": 0.9, "leadership": 0.9}
    # neural returns drifting output -> must fall back and flag it
    bad_neural = lambda prompt: ("Personality: financial stability and saving. Career: x. "
                                 "Relationships: y. Future Destiny: z. Wealth: w.")
    text, plan, prov = generate_text_controlled(probs, realizer="flan_guarded",
                                                neural_generate=bad_neural)
    check("bad neural triggers fallback", prov["realizer"] == "flan_t5_rejected_fallback")
    check("fallback flagged", prov["fallback_used"] is True)
    check("neural attempted flagged", prov["neural_attempted"] is True)
    check("fallback text is deterministic plan text", text == realize_deterministic(plan))
    # neural returns faithful output -> accepted
    good_neural = lambda prompt: ("Personality: warm. Career: leadership. "
                                  "Relationships: x. Future Destiny: y. Wealth: z.")
    _, _, prov2 = generate_text_controlled(probs, realizer="flan_guarded",
                                           neural_generate=good_neural)
    check("faithful neural accepted", prov2["realizer"] == "flan_t5" and prov2["fallback_used"] is False)
    # no realizer supplied -> deterministic fallback, never claims neural
    _, _, prov3 = generate_text_controlled(probs, realizer="flan_guarded", neural_generate=None)
    check("no realizer -> no neural claim", prov3["realizer"] == "deterministic"
          and prov3["fallback_used"] is True)


def test_prompt_construction():
    probs = {"warm": 0.9, "leadership": 0.9}
    plan = build_content_plan(probs, threshold=0.5)
    prompt = build_plan_prompt(plan)
    check("prompt lists selected concepts", "warm" in prompt and "leadership" in prompt)
    check("prompt marks empty sections", "(none)" in prompt)
    check("prompt requests five sections", prompt.count("Return exactly these five sections") == 1)
    for title in ("Personality", "Career", "Relationships", "Future Destiny", "Wealth"):
        check(f"prompt mentions {title}", title in prompt)


def test_five_section_validity_and_faithfulness():
    probs = _fixture_probs()
    text, plan, prov = generate_text_controlled(probs)
    check("controlled output valid 5-section", validate_structure(text)["valid"], str(validate_structure(text)))
    f = plan_faithfulness(text, plan)
    check("faithfulness precision 1.0 (excluding collisions)",
          f["precision_excluding_collisions"] == 1.0, str(f))
    check("no genuine unsupported concepts", f["genuine_unsupported"] == [], str(f["genuine_unsupported"]))
    check("any extra concepts are lexicon collisions",
          set(f["unsupported"]) == set(f["lexicon_collisions"]), str(f))
    # no fabricated traits outside the canonical phrase bank
    check("no fabricated traits", all(t in TRAIT_PHRASES for t in plan["selected_all"]))


def test_threshold_policy_used():
    tmap, strategy = load_thresholds(str(SRC / "models_semantic" / "trait_thresholds.json"))
    check("calibrated thresholds load", tmap is not None and len(tmap) == 45, str(strategy))
    probs = _fixture_probs()
    plan = build_content_plan(probs, thresholds=tmap, threshold=0.5)
    ok = all(TRAIT_PHRASES[t][0] == cat for cat, r in plan["categories"].items()
             for t in r["selected"])
    check("threshold policy keeps category ownership", ok)
    # per-trait thresholds recorded for active concepts
    for cat, r in plan["categories"].items():
        for t in r["active"]:
            if t not in r["thresholds"]:
                check("thresholds recorded", False, f"{cat}:{t}")
                return
    check("thresholds recorded", True)


def main():
    test_plan_alignment_and_bounds()
    test_empty_section_handling()
    test_determinism()
    test_counterfactual_sensitivity()
    test_category_conditioning()
    test_unsupported_detection_and_validator()
    test_lexicon_collision_is_not_unsupported()
    test_guarded_fallback_provenance()
    test_prompt_construction()
    test_five_section_validity_and_faithfulness()
    test_threshold_policy_used()
    print(f"\nPASS {len(PASS)} FAIL {len(FAIL)}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
