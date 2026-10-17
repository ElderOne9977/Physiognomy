# -*- coding: utf-8 -*-
"""Controlled two-stage semantic generator: planner -> realization -> validator.

45 semantic probabilities
    -> calibrated content planner      (deterministic, category-bounded)
    -> explicit content plan           (selected / omitted / empty per category)
    -> text realization                (deterministic planner OR guarded FLAN-T5)
    -> semantic consistency validator  (rejects neural output that drifts)
    -> five-section English output

Guarantees:
  * Planner selects ONLY concepts whose probability >= their calibrated threshold,
    keeps category ownership, and records omitted + empty sections.
  * Deterministic: identical probs -> identical plan -> identical text.
  * The neural realizer is GATED: its output must pass the validator; otherwise a
    deterministic fallback is used and provenance["fallback_used"] is True.
  * Fallback output is never reported as neural (provenance["realizer"]).

No training. The semantic predictor, feature extractor and frozen FLAN-T5
checkpoint are untouched. Reuses canonical TRAIT_PHRASES / category titles /
section parser; no new trait mappings.

Usage (library):
  from generate_text_controlled import generate_text_controlled, build_content_plan
  text, plan, prov = generate_text_controlled(probs, thresholds=tmap)
"""
from __future__ import annotations

import re
from functools import lru_cache

from build_semantic_labels import TRAIT_PATTERNS
from generate_text_template import (
    CONCISE_FALLBACK,
    CONCISE_PREFIX,
    CONCISE_SYNONYM_GROUPS,
    TRAIT_PHRASES,
    _strip_prefix,
)
from text_gen_common import (
    CATEGORY_ORDER,
    CATEGORY_TITLES,
    OUTPUT_TITLES,
    parse_sections,
    validate_structure,
)

DEFAULT_THRESHOLD = 0.5
DEFAULT_K = 3
PLAN_GAP = 0.02  # margin gap rule shared with the concise selector (deterministic)

REALIZERS = ("deterministic", "flan_guarded")

PLAN_HEADER = (
    "Realize the following content plan as a five-section English face-reading "
    "description. Express ONLY the listed concepts in each section. If a section "
    "has no listed concepts, write a short neutral sentence."
)


@lru_cache(maxsize=200000)
def _extract_concepts(text):
    """Canonical lexicon extraction (same regexes that built semantic_labels.csv)."""
    return frozenset(
        t for t, (_cat, pats) in TRAIT_PATTERNS.items()
        if any(re.search(p, text, re.I) for p in pats)
    )


def extract_concepts(text):
    return set(_extract_concepts(text or ""))


def lexicon_collisions(section_text, selected, cat):
    """Expressed concepts that are only triggered by a *selected* phrase's wording.

    The canonical lexicon (TRAIT_PATTERNS) is not injective: e.g. the canonical
    phrase for `opportunities` ("...through consistent effort") also matches the
    `consistent` trigger. Such a concept is a lexicon collision, not a fabricated
    trait, so it is reported separately from genuine unsupported concepts.
    """
    expressed = extract_concepts(section_text)
    extra = expressed - set(selected)
    phrases = [TRAIT_PHRASES[t][1] for t in selected if t in TRAIT_PHRASES]
    collisions, genuine = set(), set()
    for t in extra:
        pats = TRAIT_PATTERNS.get(t, (None, []))[1]
        if pats and any(re.search(p, ph, re.I) for ph in phrases for p in pats):
            collisions.add(t)
        else:
            genuine.add(t)
    return collisions, genuine


def _threshold_for(trait, thresholds, scalar):
    if thresholds is not None and trait in thresholds:
        return float(thresholds[trait])
    return float(scalar)


def _dedup_by_synonym(sorted_traits):
    kept, kept_set = [], set()
    for t in sorted_traits:
        skip = False
        for g in CONCISE_SYNONYM_GROUPS:
            if t in g and any(k in g for k in kept_set):
                skip = True
                break
        if not skip:
            kept.append(t)
            kept_set.add(t)
    return kept


def _select(sorted_active, probs, thresholds, scalar, k):
    """Top-k with synonym dedup and the deterministic margin-gap rule."""
    deduped = _dedup_by_synonym(sorted_active)
    if len(deduped) <= k:
        return deduped
    def margin(t):
        return float(probs[t]) - _threshold_for(t, thresholds, scalar)
    gap = margin(deduped[k - 1]) - margin(deduped[k])
    if gap < PLAN_GAP:
        return deduped[:k + 1]
    return deduped[:k]


def build_content_plan(probs, thresholds=None, threshold=DEFAULT_THRESHOLD, k=DEFAULT_K):
    """Deterministic category-bounded content plan.

    Returns a JSON-serializable dict:
      {"categories": {cat: {active, selected, omitted, empty, realized_concepts,
                            probabilities, thresholds}},
       "selected_all": [...], "omitted_all": [...], "n_selected": int}
    """
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError(f"threshold out of range: {threshold}")
    plan = {"categories": {}, "selected_all": [], "omitted_all": []}
    for cat in CATEGORY_ORDER:
        members = [t for t in TRAIT_PHRASES if TRAIT_PHRASES[t][0] == cat]
        active = []
        for t in members:
            p = probs.get(t)
            if p is None:
                continue
            thr = _threshold_for(t, thresholds, threshold)
            if float(p) >= thr:
                active.append(t)

        def sort_key(t):
            thr = _threshold_for(t, thresholds, threshold)
            return (-(float(probs[t]) - thr), -float(probs[t]), t)

        active.sort(key=sort_key)
        selected = _select(active, probs, thresholds, threshold, k) if active else []
        omitted = [t for t in active if t not in selected]
        plan["categories"][cat] = {
            "active": list(active),
            "selected": list(selected),
            "omitted": list(omitted),
            "empty": len(selected) == 0,
            "realized_concepts": [_strip_prefix(TRAIT_PHRASES[t][1]) for t in selected],
            "probabilities": {t: float(probs[t]) for t in active},
            "thresholds": {t: _threshold_for(t, thresholds, threshold) for t in active},
        }
        plan["selected_all"].extend(selected)
        plan["omitted_all"].extend(omitted)
    plan["n_selected"] = len(plan["selected_all"])
    return plan


# ------------------------------------------------------------------ realization

def _realize_section(cat, selected):
    """One hedged sentence per section from the selected concepts (deterministic)."""
    if not selected:
        return CONCISE_FALLBACK
    frags = [_strip_prefix(TRAIT_PHRASES[t][1]) for t in selected]
    prefix = CONCISE_PREFIX[cat]
    if len(frags) == 1:
        body = prefix + frags[0] + "."
    elif len(frags) == 2:
        body = prefix + frags[0] + " and " + frags[1] + "."
    else:
        body = prefix + ", ".join(frags[:-1]) + ", and " + frags[-1] + "."
    return body


def realize_deterministic(plan):
    """Render the plan to five-section text using the canonical phrase bank."""
    sections = []
    for cat in CATEGORY_ORDER:
        body = _realize_section(cat, plan["categories"][cat]["selected"])
        sections.append(f"{CATEGORY_TITLES[cat]}: {body}")
    return "\n".join(sections)


def build_plan_prompt(plan):
    """Explicit content-plan prompt for a neural realizer."""
    lines = [PLAN_HEADER, ""]
    for cat in CATEGORY_ORDER:
        concepts = plan["categories"][cat]["selected"]
        lines.append(f"{OUTPUT_TITLES[cat]}: " +
                     (", ".join(concepts) if concepts else "(none)"))
    lines.append("")
    lines.append("Return exactly these five sections:")
    lines.extend(OUTPUT_TITLES[c] for c in CATEGORY_ORDER)
    return "\n".join(lines)


# ------------------------------------------------------------------ validation

def validate_against_plan(text, plan, require_recall=True):
    """Semantic consistency validator.

    Accepts a candidate text only when it is structurally valid, expresses no
    concept outside the plan's selected set, keeps each concept in its own
    category section, and (optionally) realizes at least one selected concept
    per non-empty section. Returns (ok, reasons, expressed).

    Lexicon collisions (a selected phrase's wording triggering another trait's
    trigger word, see lexicon_collisions) are reported as `collision:` reasons and
    are NOT treated as failures; only genuine unsupported concepts fail.
    """
    reasons = []
    v = validate_structure(text)
    if not v["valid"]:
        reasons.append("structure_invalid")
    sections, _ = parse_sections(text)

    expressed = {}
    collisions = {}
    for cat in CATEGORY_ORDER:
        title = CATEGORY_TITLES[cat]
        sel = set(plan["categories"][cat]["selected"])
        body = sections.get(title, "")
        sec_exp = extract_concepts(body)
        expressed[cat] = sorted(sec_exp)
        # category boundary: concepts must belong to this category
        foreign = {t for t in sec_exp
                   if t in TRAIT_PHRASES and TRAIT_PHRASES[t][0] != cat}
        if foreign:
            reasons.append(f"category_leak:{cat}:{sorted(foreign)}")
        coll, genuine = lexicon_collisions(body, sel, cat)
        collisions[cat] = sorted(coll)
        if coll:
            reasons.append(f"collision:{cat}:{sorted(coll)}")
        if genuine:
            reasons.append(f"unsupported:{cat}:{sorted(genuine)}")
        if require_recall and sel and not (sec_exp & sel):
            reasons.append(f"missing_selected:{cat}")
    failing = [r for r in reasons if not r.startswith("collision:")]
    return (len(failing) == 0), reasons, expressed


def plan_faithfulness(text, plan):
    """Precision/recall of expressed concepts vs the plan's selected concepts.

    Precision counts lexicon collisions as expressed (conservative), but reports
    them separately so a collision is never confused with a fabricated trait.
    """
    expressed = set()
    collisions = set()
    sections, _ = parse_sections(text)
    for cat in CATEGORY_ORDER:
        body = sections.get(CATEGORY_TITLES[cat], "")
        sel = set(plan["categories"][cat]["selected"])
        coll, _genuine = lexicon_collisions(body, sel, cat)
        collisions |= coll
        expressed |= (extract_concepts(body) & set(TRAIT_PHRASES))
    planned = set(plan["selected_all"])
    tp, fp, fn = len(expressed & planned), len(expressed - planned), len(planned - expressed)
    p = tp / (tp + fp) if (tp + fp) else 1.0
    r = tp / (tp + fn) if (tp + fn) else 1.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    unsupported = expressed - planned
    # collision-adjusted precision: drop collisions from the false-positive set
    tp2 = len((expressed - collisions) & planned)
    fp2 = len((expressed - collisions) - planned)
    p2 = tp2 / (tp2 + fp2) if (tp2 + fp2) else 1.0
    return {"precision": p, "recall": r, "f1": f,
            "precision_excluding_collisions": p2,
            "unsupported": sorted(unsupported),
            "lexicon_collisions": sorted(collisions),
            "genuine_unsupported": sorted(unsupported - collisions)}


# ------------------------------------------------------------------------ API

def generate_text_controlled(probs, thresholds=None, threshold=DEFAULT_THRESHOLD,
                             k=DEFAULT_K, realizer="deterministic",
                             neural_generate=None):
    """Planner -> realization -> validator.

    realizer="deterministic"  : phrase-bank realization (always succeeds).
    realizer="flan_guarded"   : call neural_generate(prompt)->text, accept only if
                                the validator passes, else deterministic fallback.

    Returns (text, plan, provenance). provenance["realizer"] is
    "deterministic" or "flan_t5" or "flan_t5_rejected_fallback", and
    provenance["fallback_used"] is True whenever neural output was rejected or
    no neural realizer was supplied.
    """
    if realizer not in REALIZERS:
        raise ValueError(f"unknown realizer {realizer!r} choices={REALIZERS}")
    plan = build_content_plan(probs, thresholds=thresholds, threshold=threshold, k=k)
    det_text = realize_deterministic(plan)

    provenance = {
        "realizer": "deterministic",
        "fallback_used": True,
        "neural_attempted": False,
        "rejection_reasons": [],
        "plan_selected": list(plan["selected_all"]),
        "plan_omitted": list(plan["omitted_all"]),
        "empty_sections": [c for c in CATEGORY_ORDER if plan["categories"][c]["empty"]],
    }

    if realizer == "deterministic":
        provenance["faithfulness"] = plan_faithfulness(det_text, plan)
        return det_text, plan, provenance

    # flan_guarded
    if neural_generate is None:
        provenance["rejection_reasons"] = ["no_neural_realizer"]
        provenance["faithfulness"] = plan_faithfulness(det_text, plan)
        return det_text, plan, provenance

    provenance["neural_attempted"] = True
    prompt = build_plan_prompt(plan)
    provenance["plan_prompt_preview"] = prompt[:800]
    try:
        cand = neural_generate(prompt)
    except Exception as e:  # neural failure must never break the demo
        provenance["rejection_reasons"] = [f"neural_error:{type(e).__name__}"]
        provenance["faithfulness"] = plan_faithfulness(det_text, plan)
        return det_text, plan, provenance

    ok, reasons, _ = validate_against_plan(cand, plan)
    if ok:
        provenance["realizer"] = "flan_t5"
        provenance["fallback_used"] = False
        provenance["faithfulness"] = plan_faithfulness(cand, plan)
        return cand, plan, provenance

    provenance["realizer"] = "flan_t5_rejected_fallback"
    provenance["rejection_reasons"] = reasons
    provenance["faithfulness"] = plan_faithfulness(det_text, plan)
    return det_text, plan, provenance