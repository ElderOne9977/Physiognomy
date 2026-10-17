"""
Shared deterministic helpers for the text-generation experiment.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = ROOT / "semantic_schema.json"

CATEGORY_ORDER = ["personality", "career", "relationships", "future_destiny", "wealth"]
CATEGORY_TITLES = {
    "personality": "Personality",
    "career": "Career",
    "relationships": "Relationships",
    "future_destiny": "Future & Destiny",
    "wealth": "Wealth",
}
TITLE_TO_CATEGORY = {v: k for k, v in CATEGORY_TITLES.items()}

OUTPUT_TITLES = {
    "personality": "Personality",
    "career": "Career",
    "relationships": "Relationships",
    "future_destiny": "Future Destiny",
    "wealth": "Wealth",
}

INSTRUCTION = (
    "Generate a structured English face-reading description from the semantic "
    "probability profile below."
)

BIN_LABELS = [f"P{i}" for i in range(10)]

def prob_to_bin(p: float) -> str:
    try:
        v = float(p)
    except Exception:
        raise ValueError(f"prob_to_bin: non-numeric {p!r}")
    if v < 0:
        v = 0.0
    if v > 1:
        v = 1.0
    idx = int(v * 10)
    if idx >= 10:
        idx = 9
    return f"P{idx}"

def bin_to_prob_center(bin_label: str) -> float:
    idx = int(bin_label[1:])
    return idx * 0.1 + 0.05

def load_schema(path=DEFAULT_SCHEMA):
    schema = json.loads(Path(path).read_text(encoding="utf-8"))
    traits = schema["traits"]
    trait_names = [t["name"] for t in traits]
    cat_of = {t["name"]: t["category"] for t in traits}
    return trait_names, cat_of

def format_input(probs, trait_names, cat_of, decimals=2, representation="continuous"):
    if representation not in ("continuous", "bins"):
        raise ValueError(f"unknown representation {representation!r}")
    lines = [INSTRUCTION, ""]
    for cat in CATEGORY_ORDER:
        lines.append(f"{OUTPUT_TITLES[cat]}:")
        for t in trait_names:
            if cat_of[t] == cat:
                if representation == "bins":
                    lines.append(f"{t}={prob_to_bin(probs[t])}")
                else:
                    lines.append(f"{t}={probs[t]:.{decimals}f}")
        lines.append("")
    lines.append("Return exactly these five sections:")
    lines.extend(OUTPUT_TITLES[c] for c in CATEGORY_ORDER)
    return "\n".join(lines)

def format_target(predictions):
    blocks = []
    for cat in CATEGORY_ORDER:
        vals = predictions.get(cat) or []
        vals = [x.strip() for x in vals if isinstance(x, str) and x.strip()]
        body = " ".join(vals) if vals else "(no provider text)"
        blocks.append(f"{OUTPUT_TITLES[cat]}:\n{body}")
    return "\n\n".join(blocks)

_SECTION_RE = re.compile(
    r"(Personality|Career|Relationships|Future\s*&\s*Destiny|Future\s+Destiny|Wealth)\s*:",
    re.IGNORECASE,
)

def parse_sections(text):
    text = text or ""
    matches = list(_SECTION_RE.finditer(text))
    out = {}
    seen_order = []
    for i, m in enumerate(matches):
        raw = m.group(1)
        title = re.sub(r"\s*&\s*", " & ", raw)
        title = re.sub(r"\s+", " ", title).strip()
        title = title.title().replace("Future & Destiny", "Future & Destiny")
        norm = None
        for canon in CATEGORY_TITLES.values():
            if canon.lower().replace(" ", "") == title.lower().replace(" ", ""):
                norm = canon
                break
        if norm is None and title.lower().startswith("future"):
            norm = "Future & Destiny"
        if norm is None:
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        out[norm] = body
        seen_order.append(norm)
    return out, seen_order

def validate_structure(text):
    sections, order = parse_sections(text)
    present = [c for c in CATEGORY_TITLES.values() if c in sections]
    missing = [c for c in CATEGORY_TITLES.values() if c not in sections]
    empty = [c for c in present if not sections[c].strip()]
    expected = set(CATEGORY_TITLES.values())
    unexpected = [s for s in order if s not in expected]
    duplicates = len(order) - len(set(order))
    return {
        "sections_present": len(present),
        "valid": len(present) == 5 and not empty and duplicates == 0,
        "missing_sections": missing,
        "empty_sections": empty,
        "duplicate_sections": duplicates,
        "unexpected_sections": unexpected,
        "n_section_headers": len(order),
    }

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)