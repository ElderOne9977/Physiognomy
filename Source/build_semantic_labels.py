"""
Stage A: audit provider text and build the semantic label dataset.

Reads labels_enriched.jsonl, scans the five prediction categories for
semantic concepts with a deterministic lexicon, then emits:

  - Source/semantic_vocab_report.json : corpus audit + vocab decisions
  - semantic_labels.csv               : binary label matrix (image + traits)
  - semantic_schema.json              : canonical trait list + categories

Usage:
  python Source/build_semantic_labels.py

The provider text is template-generated, so concepts appear as fixed tokens
in fixed syntactic slots. A deterministic regex lexicon is enough; no NLP
dependency is needed. Only clear morphological variants are merged.
"""

import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
JSONL_PATH = ROOT / "labels_enriched.jsonl"
VOCAB_REPORT_PATH = Path(__file__).parent / "semantic_vocab_report.json"
LABELS_CSV_PATH = ROOT / "semantic_labels.csv"
SCHEMA_PATH = ROOT / "semantic_schema.json"

PRED_CATS = ["personality", "career", "relationships", "future_destiny", "wealth"]

# Thresholds: keep traits with enough positives but not near-universal ones.
MIN_POS = 300
MAX_POS = 5500
MIN_PREVALENCE = 0.03
MAX_PREVALENCE = 0.92

# Canonical lexicon: name -> (category, [regex, ...]).
# Patterns are searched case-insensitively against the combined record text.
TRAIT_PATTERNS: Dict[str, Tuple[str, List[str]]] = {
    # personality
    "warm":            ("personality", [r"\bwarm(?:th)?\b"]),
    "calm":            ("personality", [r"\bcalm\b"]),
    "confident":       ("personality", [r"\bconfiden(?:ce|t)\b"]),
    "perceptive":      ("personality", [r"\bpercepti(?:ve|on)\b"]),
    "resilient":       ("personality", [r"\bresilien(?:ce|t)\b"]),
    "reliable":        ("personality", [r"\breliab(?:le|ility)\b"]),
    "practical":       ("personality", [r"\bpractical(?:ity)?\b"]),
    "empathetic":      ("personality", [r"\bempath(?:etic|y)\b"]),
    "adaptable":       ("personality", [r"\badaptab(?:le|ility)|\badaptive\b"]),
    "disciplined":     ("personality", [r"\bdisciplin(?:ed|e)\b"]),
    "curious":         ("personality", [r"\bcuriou(?:s|sity)\b"]),
    "focused":         ("personality", [r"\bfocus(?:ed|)\b"]),
    "grounded":        ("personality", [r"\bgrounded\b"]),
    "dependable":      ("personality", [r"\bdependab(?:le|ility)\b"]),
    "sociable":        ("personality", [r"\bsociab(?:le|ility)\b"]),
    "thoughtful":      ("personality", [r"\bthoughtful(?:ness)?\b"]),
    "poised":          ("personality", [r"\bpoised\b"]),
    "observant":       ("personality", [r"\bobservan(?:t|ce)\b"]),
    # career
    "collaborative":   ("career", [r"\bcollaborat(?:ive|ion)\b"]),
    "leadership":      ("career", [r"\bleadership\b"]),
    "analytical":      ("career", [r"\banalytic(?:al)?\b"]),
    "creative":        ("career", [r"\bcreat(?:ive|ivity)\b"]),
    "client_facing":   ("career", [r"\bclient.?facing\b"]),
    "research":        ("career", [r"\bresearch\b"]),
    "learning_rich":   ("career", [r"\blearning.?rich\b"]),
    "public_roles":    ("career", [r"\bpublic\b"]),
    "responsibility_driven": ("career", [r"\bresponsibility.?driven\b"]),
    "results_oriented":("career", [r"\bresults.?oriented\b|\bpragmatic\b"]),
    # relationships
    "loyal":           ("relationships", [r"\bloyal(?:ty)?\b"]),
    "patient":         ("relationships", [r"\bpatien(?:ce|t)\b"]),
    "trust_based":     ("relationships", [r"\btrust.?based\b"]),
    "stable_rel":      ("relationships", [r"\bstable\b"]),
    "attentive_partner":("relationships", [r"\battentive\s+partner\b"]),
    "intellectual_rel":("relationships", [r"\bintellectual\s+exchange\b"]),
    # future_destiny
    "steady_growth":   ("future_destiny", [r"\bsteady\s+(?:growth|ascent|upward|arc|progress)\b"]),
    "favorable":       ("future_destiny", [r"\bfavorable\b"]),
    "opportunities":   ("future_destiny", [r"\bopportunit(?:ies|y)\b"]),
    "breakthroughs":   ("future_destiny", [r"\bbreakthroughs?\b"]),
    "consistent":      ("future_destiny", [r"\bconsistent\b"]),
    "influence":       ("future_destiny", [r"\binfluence\b"]),
    "social_connections":("future_destiny", [r"\bsocial\b"]),
    "meaningful":      ("future_destiny", [r"\bmeaningful\b"]),
    # wealth
    "financial_stability":("wealth", [r"\bfinancial\s+stability\b"]),
    "generous_spending":  ("wealth", [r"\bgenerous\b"]),
    "mindful_spending":   ("wealth", [r"\bmindful\b"]),
    "saving":             ("wealth", [r"\bsaving\b"]),
    "conservative":       ("wealth", [r"\bconservative\b"]),
}


def norm_name(value) -> str:
    """Normalize an image identifier (matches build_training_dataset.py)."""
    s = os.path.basename(str(value).strip())
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\)\s+\(", ")(", s)
    return s.lower()


def match_trait(text: str, patterns: List[str]) -> bool:
    for pat in patterns:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


def record_text(rec: dict) -> str:
    pred = rec.get("predictions") or {}
    parts = []
    for cat in PRED_CATS:
        vals = pred.get(cat) or []
        if isinstance(vals, str):
            vals = [vals]
        parts.extend(str(v) for v in vals)
    return " ".join(parts)


def load_jsonl(path: Path) -> List[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def audit_corpus(records: List[dict]) -> dict:
    total = len(records)
    cat_counts = {c: 0 for c in PRED_CATS}
    trait_raw = {t: Counter() for t in TRAIT_PATTERNS}
    trait_examples = {t: [] for t in TRAIT_PATTERNS}
    for rec in records:
        pred = rec.get("predictions") or {}
        for c in PRED_CATS:
            vals = pred.get(c) or []
            if isinstance(vals, str):
                vals = [vals]
            if vals:
                cat_counts[c] += 1
        text = record_text(rec)
        for t, (cat, pats) in TRAIT_PATTERNS.items():
            if match_trait(text, pats):
                trait_raw[t][cat] += 1
                if len(trait_examples[t]) < 3:
                    trait_examples[t].append({"category": cat, "snippet": text[:200]})
    return {
        "total_records": total,
        "records_per_category": cat_counts,
        "trait_raw_counts": {t: dict(c) for t, c in trait_raw.items()},
        "trait_examples": trait_examples,
    }


def select_traits(audit: dict):
    total = audit["total_records"] or 1
    selected, excluded = [], []
    for t, (cat, pats) in TRAIT_PATTERNS.items():
        raw = audit["trait_raw_counts"].get(t, {})
        pos = sum(raw.values())
        prev = pos / total
        info = {
            "name": t,
            "category": cat,
            "patterns": pats,
            "positive_count": pos,
            "prevalence": round(prev, 4),
            "per_category": raw,
            "examples": audit["trait_examples"].get(t, [])[:3],
        }
        reasons = []
        if pos < MIN_POS:
            reasons.append(f"positive_count={pos} < MIN_POS={MIN_POS}")
        if pos > MAX_POS:
            reasons.append(f"positive_count={pos} > MAX_POS={MAX_POS}")
        if prev < MIN_PREVALENCE:
            reasons.append(f"prevalence={prev:.4f} < MIN_PREVALENCE={MIN_PREVALENCE}")
        if prev > MAX_PREVALENCE:
            reasons.append(f"prevalence={prev:.4f} > MAX_PREVALENCE={MAX_PREVALENCE}")
        if reasons:
            info["exclusion_reasons"] = reasons
            excluded.append(info)
        else:
            selected.append(info)
    return [s["name"] for s in selected], selected, excluded


def build_label_matrix(records, trait_names):
    rows = []
    cooccur = defaultdict(lambda: defaultdict(int))
    for rec in records:
        text = record_text(rec)
        labels = {}
        active = []
        for t in trait_names:
            _, pats = TRAIT_PATTERNS[t]
            v = 1 if match_trait(text, pats) else 0
            labels[t] = v
            if v:
                active.append(t)
        for i, t1 in enumerate(active):
            for t2 in active[i + 1:]:
                cooccur[t1][t2] += 1
        rows.append({"image": norm_name(rec.get("image", "")), **labels})
    return rows, {t: dict(c) for t, c in cooccur.items()}


def compute_stats(rows, trait_names):
    n = len(rows)
    per_trait = {}
    for t in trait_names:
        s = sum(r[t] for r in rows)
        per_trait[t] = {
            "positive_count": s,
            "negative_count": n - s,
            "prevalence": round(s / n, 4) if n else 0.0,
        }
    dist = Counter(sum(r[t] for t in trait_names) for r in rows)
    return {
        "n_samples": n,
        "per_trait": per_trait,
        "labels_per_sample_distribution": dict(sorted(dist.items())),
        "traits_with_zero_positives": [t for t in trait_names if per_trait[t]["positive_count"] == 0],
        "traits_universal": [t for t in trait_names if per_trait[t]["prevalence"] > 0.98],
    }


def main():
    print(f"Loading {JSONL_PATH} ...")
    records = load_jsonl(JSONL_PATH)
    print(f"  {len(records)} records")
    audit = audit_corpus(records)
    selected_names, selected_info, excluded_info = select_traits(audit)
    print(f"Selected {len(selected_names)} traits; excluded {len(excluded_info)}")
    for e in excluded_info:
        print(f"  excluded {e['name']}: {e['exclusion_reasons']}")

    rows, cooccur = build_label_matrix(records, selected_names)
    stats = compute_stats(rows, selected_names)

    header = ["image"] + selected_names
    with open(LABELS_CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([r[c] for c in header])
    print(f"Wrote {LABELS_CSV_PATH} ({len(rows)} rows x {len(selected_names)} traits)")

    schema = {
        "version": 1,
        "traits": [{"name": s["name"], "category": s["category"]} for s in selected_info],
        "thresholds": {
            "MIN_POS": MIN_POS,
            "MAX_POS": MAX_POS,
            "MIN_PREVALENCE": MIN_PREVALENCE,
            "MAX_PREVALENCE": MAX_PREVALENCE,
        },
        "total_candidates": len(TRAIT_PATTERNS),
        "selected_count": len(selected_names),
        "excluded_count": len(excluded_info),
    }
    SCHEMA_PATH.write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {SCHEMA_PATH}")

    report = {
        "generated_by": "build_semantic_labels.py",
        "source_jsonl": str(JSONL_PATH),
        "source_records": len(records),
        "records_per_category": audit["records_per_category"],
        "total_candidate_concepts": len(TRAIT_PATTERNS),
        "thresholds": schema["thresholds"],
        "selected_concepts": selected_info,
        "excluded_concepts": excluded_info,
        "label_statistics": stats,
        "cooccurrence_top": {
            t1: dict(sorted(c.items(), key=lambda x: -x[1])[:10])
            for t1, c in sorted(cooccur.items())[:15]
        },
    }
    VOCAB_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {VOCAB_REPORT_PATH}")

    print("\n=== Summary ===")
    for s in selected_info:
        print(f"  {s['name']:22s} | {s['category']:18s} | +{s['positive_count']:5d} ({s['prevalence']:.3f})")


if __name__ == "__main__":
    main()