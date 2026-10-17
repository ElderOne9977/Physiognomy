"""
Deterministic template text generator over semantic trait probabilities.

Covers the five provider categories: Personality, Career, Relationships,
Future & Destiny, Wealth. Central DEFAULT_THRESHOLD=0.5; no hallucinated
traits; category ownership preserved; deterministic output per prob vector.

Usage:
  python Source/generate_text_template.py --input semantic_predictions.jsonl --output generated.jsonl
  python Source/generate_text_template.py --probs '{"warm":0.8,"calm":0.2}' --output -

The generator is intentionally boring: each section concatenates its active
traits' canned phrases. No LLM, no sampling.
"""
import argparse
import json
from pathlib import Path

DEFAULT_THRESHOLD = 0.5
DEFAULT_THRESHOLDS_PATH = Path(__file__).parent / "models_semantic" / "trait_thresholds.json"

# Canonical trait -> (category, phrase) used when prob >= threshold.
# Phrases mirror the provider's hedged style.
TRAIT_PHRASES = {
    # personality
    "warm": ("personality", "May suggest a warm, approachable presence"),
    "calm": ("personality", "May suggest calm composure under pressure"),
    "confident": ("personality", "May suggest quiet confidence"),
    "perceptive": ("personality", "May suggest perceptive attention to detail"),
    "resilient": ("personality", "May suggest resilient follow-through"),
    "reliable": ("personality", "May suggest reliable, steady character"),
    "practical": ("personality", "May suggest practical, grounded judgment"),
    "empathetic": ("personality", "May suggest empathetic responsiveness"),
    "adaptable": ("personality", "May suggest adaptable flexibility"),
    "disciplined": ("personality", "May suggest disciplined consistency"),
    "curious": ("personality", "May suggest curious, learning-oriented outlook"),
    "focused": ("personality", "May suggest focused diligence"),
    "grounded": ("personality", "May suggest grounded stability"),
    "dependable": ("personality", "May suggest dependable steadiness"),
    "sociable": ("personality", "May suggest sociable ease with others"),
    "thoughtful": ("personality", "May suggest thoughtful consideration"),
    "poised": ("personality", "May suggest poised self-presentation"),
    "observant": ("personality", "May suggest observant noticing"),
    # career
    "collaborative": ("career", "May be associated with collaborative, people-centered work"),
    "leadership": ("career", "May be associated with leadership or responsibility for coordination"),
    "creative": ("career", "May be associated with creative or design-leaning tasks"),
    "client_facing": ("career", "May be associated with client-facing roles where presence matters"),
    "research": ("career", "May be associated with research or close analysis"),
    "learning_rich": ("career", "May be associated with learning-rich environments"),
    "public_roles": ("career", "May be associated with public or visible roles"),
    "responsibility_driven": ("career", "May be associated with responsibility-driven positions"),
    "results_oriented": ("career", "May be associated with results-oriented, pragmatic execution"),
    # relationships
    "loyal": ("relationships", "Could favor loyal, committed bonds"),
    "patient": ("relationships", "Could favor patient, attentive relating"),
    "trust_based": ("relationships", "Could favor trust-based connection"),
    "stable_rel": ("relationships", "Could favor stable, enduring relationships"),
    "intellectual_rel": ("relationships", "Could favor intellectual exchange in close ties"),
    # future_destiny
    "steady_growth": ("future_destiny", "May suggest a path of steady growth"),
    "favorable": ("future_destiny", "May suggest favorable conditions with careful choices"),
    "opportunities": ("future_destiny", "May suggest opportunities through consistent effort"),
    "breakthroughs": ("future_destiny", "May suggest quiet breakthroughs over time"),
    "consistent": ("future_destiny", "May suggest consistent, incremental progress"),
    "influence": ("future_destiny", "May suggest growing influence with experience"),
    "social_connections": ("future_destiny", "May suggest supportive social connections"),
    "meaningful": ("future_destiny", "May suggest meaningful direction when aligned with values"),
    # wealth
    "financial_stability": ("wealth", "May be associated with financial stability through steadiness"),
    "generous_spending": ("wealth", "May be associated with generous but mindful spending"),
    "mindful_spending": ("wealth", "May be associated with mindful, balanced spending"),
    "saving": ("wealth", "May be associated with saving and gradual accumulation"),
    "conservative": ("wealth", "May be associated with conservative financial choices"),
}

CATEGORY_ORDER = ["personality", "career", "relationships", "future_destiny", "wealth"]
CATEGORY_TITLES = {
    "personality": "Personality",
    "career": "Career",
    "relationships": "Relationships",
    "future_destiny": "Future & Destiny",
    "wealth": "Wealth",
}

# ponytail: concise planner uses margin-rank + synonym dedup + 2-4 cap; upgrade to learned ranker only after gap measured
CONCISE_FALLBACK = "No sufficiently supported semantic concepts were selected for this section."
CONCISE_PREFIX = {
    "personality": "May suggest ",
    "career": "May be associated with ",
    "relationships": "Could favor ",
    "future_destiny": "May suggest ",
    "wealth": "May be associated with ",
}
CONCISE_SYNONYM_GROUPS = [
    {"reliable", "dependable"},
    {"practical", "grounded"},
    {"warm", "sociable"},
    {"financial_stability", "saving"},
    {"mindful_spending", "generous_spending"},
    {"steady_growth", "consistent"},
    {"opportunities", "favorable"},
]
CONCISE_TARGET_K = 3

def load_thresholds(thresholds_path):
    """Load threshold policy from trait_thresholds.json.

    Returns (threshold_map, strategy_name) or (None, None) if missing.
    threshold_map is trait_name -> float when available.
    Resolution order: trait_thresholds > category_thresholds > global_threshold > 0.5.
    """
    if thresholds_path is None:
        return None, None
    p = Path(thresholds_path)
    if not p.exists():
        return None, None
    d = json.loads(p.read_text(encoding="utf-8"))
    known = set(TRAIT_PHRASES)
    for k in ("trait_thresholds", "thresholds"):
        if k in d and isinstance(d[k], dict):
            for trait in d[k]:
                if trait not in known and trait not in (
                    "global", "personality", "career", "relationships",
                    "future_destiny", "wealth"):
                    raise ValueError(f"threshold file contains unknown key {trait!r}")
    strategy = d.get("strategy", "unknown")
    tmap = None
    if "trait_thresholds" in d and isinstance(d["trait_thresholds"], dict):
        tmap = {t: float(d["trait_thresholds"].get(t, d.get("global_threshold", DEFAULT_THRESHOLD)))
                for t in TRAIT_PHRASES}
        if "category_thresholds" in d:
            for t in TRAIT_PHRASES:
                if t not in d["trait_thresholds"]:
                    cat = TRAIT_PHRASES[t][0]
                    tmap[t] = float(d["category_thresholds"].get(cat, tmap[t]))
    elif "thresholds" in d and isinstance(d["thresholds"], dict):
        th = d["thresholds"]
        if any(k in th for k in TRAIT_PHRASES):
            tmap = {t: float(th.get(t, DEFAULT_THRESHOLD)) for t in TRAIT_PHRASES}
        elif any(k in th for k in ("personality", "career", "relationships", "future_destiny", "wealth")):
            tmap = {t: float(th.get(TRAIT_PHRASES[t][0], DEFAULT_THRESHOLD)) for t in TRAIT_PHRASES}
        elif "global" in th:
            tmap = {t: float(th["global"]) for t in TRAIT_PHRASES}
    elif "global_threshold" in d:
        tmap = {t: float(d["global_threshold"]) for t in TRAIT_PHRASES}
    if tmap is not None:
        for t in tmap:
            assert 0.0 <= tmap[t] <= 1.0, f"threshold {t}={tmap[t]} out of [0,1]"
    return tmap, strategy

def _strip_prefix(phrase: str) -> str:
    for pref in ("May suggest ", "May be associated with ", "Could favor "):
        if phrase.startswith(pref):
            return phrase[len(pref):]
    return phrase

def _dedup_by_synonym(sorted_traits):
    kept = []
    kept_set = set()
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

def _select_concise(sorted_active, probs, thresholds, threshold_val):
    deduped = _dedup_by_synonym(sorted_active)
    n = len(deduped)
    if n <= 3:
        return deduped
    def margin(t):
        thr = thresholds[t] if thresholds is not None and t in thresholds else threshold_val
        return float(probs[t]) - float(thr)
    gap = margin(deduped[2]) - margin(deduped[3])
    if gap < 0.02:
        return deduped[:4]
    return deduped[:3]

def generate_text_concise(probs, threshold=DEFAULT_THRESHOLD, thresholds=None):
    """Deterministic concise generator. Returns (text, active, trace)."""
    active = []
    for t, p in probs.items():
        if t not in TRAIT_PHRASES:
            continue
        thr = thresholds[t] if thresholds is not None and t in thresholds else threshold
        assert 0.0 <= thr <= 1.0
        if float(p) >= thr:
            active.append(t)
    by_cat_active = {c: [] for c in CATEGORY_ORDER}
    for t in active:
        cat = TRAIT_PHRASES[t][0]
        by_cat_active[cat].append(t)
    trace = {}
    by_cat_selected = {c: [] for c in CATEGORY_ORDER}
    for cat in CATEGORY_ORDER:
        lst = by_cat_active[cat]
        def sort_key(t, _thr=threshold, _thresholds=thresholds, _probs=probs):
            thr = _thresholds[t] if _thresholds is not None and t in _thresholds else _thr
            margin = float(_probs[t]) - float(thr)
            return (-margin, -float(_probs[t]), t)
        sorted_active = sorted(lst, key=sort_key)
        selected = _select_concise(sorted_active, probs, thresholds, threshold) if sorted_active else []
        by_cat_selected[cat] = selected
        omitted = [t for t in sorted_active if t not in selected]
        realized = [_strip_prefix(TRAIT_PHRASES[t][1]) for t in selected]
        fallback_used = len(selected) == 0
        trace[cat] = {
            "active_traits": sorted_active,
            "selected_traits": selected,
            "omitted_active_traits": omitted,
            "probabilities": {t: float(probs[t]) for t in sorted_active},
            "realized_concepts": realized,
            "fallback_used": fallback_used,
        }
    sections = []
    for cat in CATEGORY_ORDER:
        title = CATEGORY_TITLES[cat]
        sel = by_cat_selected[cat]
        if not sel:
            body = CONCISE_FALLBACK
        else:
            frags = [_strip_prefix(TRAIT_PHRASES[t][1]) for t in sel]
            prefix = CONCISE_PREFIX[cat]
            if len(frags) == 1:
                body = prefix + frags[0] + "."
            elif len(frags) == 2:
                body = prefix + frags[0] + " and " + frags[1] + "."
            else:
                body = prefix + ", ".join(frags[:-1]) + ", and " + frags[-1] + "."
        sections.append(f"{title}: {body}")
        trace[cat]["section_text"] = body
        trace[cat]["section_length_words"] = len(body.split())
        trace[cat]["section_length_chars"] = len(body)
    text = "\n".join(sections)
    return text, active, trace

def generate_text(probs, threshold=DEFAULT_THRESHOLD, thresholds=None):
    """Return (text, active_traits) for a probs dict.

    If thresholds is a dict trait->float, per-trait thresholds take precedence
    over the scalar threshold. Scalar threshold is the fallback.
    """
    active = []
    for t, p in probs.items():
        if t not in TRAIT_PHRASES:
            continue
        thr = thresholds[t] if thresholds is not None and t in thresholds else threshold
        assert 0.0 <= thr <= 1.0, f"threshold for {t} out of range: {thr}"
        if float(p) >= thr:
            active.append(t)
    by_cat = {c: [] for c in CATEGORY_ORDER}
    for t in active:
        cat, phrase = TRAIT_PHRASES[t]
        by_cat[cat].append(phrase)
    fallbacks = {
        "personality": "May suggest a balanced, even-keeled disposition",
        "career": "May be associated with steady, reliable work habits",
        "relationships": "Could favor respectful, considerate relating",
        "future_destiny": "May suggest gradual development with continued effort",
        "wealth": "May be associated with measured financial habits",
    }
    sections = []
    for cat in CATEGORY_ORDER:
        phrases = by_cat[cat] if by_cat[cat] else [fallbacks[cat]]
        title = CATEGORY_TITLES[cat]
        body = "; ".join(phrases) + "."
        sections.append(f"{title}: {body}")
    text = "\n".join(sections)
    return text, active

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="input jsonl with {image, probs} per line")
    ap.add_argument("--probs", help="inline JSON dict of trait->prob")
    ap.add_argument("--output", required=True, help="output jsonl path or - for stdout")
    ap.add_argument("--threshold", type=float, default=None,
                    help=f"global threshold override (default {DEFAULT_THRESHOLD})")
    ap.add_argument("--thresholds", type=str, default=None,
                    help="path to trait_thresholds.json (per-trait/category/global policy)")
    args = ap.parse_args()
    tmap, strategy = load_thresholds(args.thresholds)
    if tmap is None and args.thresholds is not None:
        raise SystemExit(f"thresholds file not found: {args.thresholds}")
    if tmap is None and args.threshold is None and DEFAULT_THRESHOLDS_PATH.exists():
        tmap, strategy = load_thresholds(str(DEFAULT_THRESHOLDS_PATH))
        if tmap is not None:
            print(f"auto-loaded thresholds from {DEFAULT_THRESHOLDS_PATH} (strategy={strategy})")
    scalar_thr = args.threshold if args.threshold is not None else DEFAULT_THRESHOLD
    if tmap is not None:
        print(f"using threshold policy strategy={strategy} "
              f"scalar_fallback={scalar_thr} "
              f"traits_with_policy={len(tmap)}")
    if args.probs:
        probs = json.loads(args.probs)
        text, active = generate_text(probs, threshold=scalar_thr, thresholds=tmap)
        rec = {"probs": probs, "active_traits": active, "generated_text": text}
        if args.output == "-":
            print(json.dumps(rec, ensure_ascii=False, indent=2))
        else:
            Path(args.output).write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
        return
    if not args.input:
        raise SystemExit("provide --input or --probs")
    inp = Path(args.input)
    out = args.output
    is_stdout = out == "-"
    fh = open(out, "w", encoding="utf-8") if not is_stdout else None
    n = 0
    for line in open(inp, encoding="utf-8"):
        line=line.strip()
        if not line:
            continue
        rec=json.loads(line)
        probs=rec.get("probs") or rec.get("trait_probs") or {}
        img=rec.get("image") or rec.get("image_id") or ""
        text, active = generate_text(probs, threshold=scalar_thr, thresholds=tmap)
        out_rec={"image": img, "active_traits": active, "generated_text": text, "probs": probs}
        s=json.dumps(out_rec, ensure_ascii=False)
        if is_stdout:
            print(s)
        else:
            fh.write(s+"\n")
        n+=1
    if fh:
        fh.close()
    thr_desc = f"thresholds={args.thresholds or 'auto/' + str(scalar_thr)}"
    print(f"Wrote {n} generated texts to {out} ({thr_desc})")


if __name__=="__main__":
    main()
