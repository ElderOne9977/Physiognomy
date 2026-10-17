"""
Measure template realization fidelity: activated semantic concepts -> generated text.

For each generated record, checks whether every active trait's canned phrase
survives in the generated text and whether any non-active trait phrase appears
(spurious). This isolates template/parser loss from semantic-predictor error.

Usage:
  python Source/measure_realization.py --generated generated_text_calibrated.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from generate_text_template import TRAIT_PHRASES

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated", default=str(ROOT / "generated_text_calibrated.jsonl"))
    ap.add_argument("--output", default=str(Path(__file__).parent / "semantic_realization.json"))
    args = ap.parse_args()

    tp = fp = fn = 0
    n = misses = extra = 0
    for line in open(args.generated, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        text = rec.get("generated_text") or ""
        active = set(rec.get("active_traits") or [])
        found = {t for t, (_, phrase) in TRAIT_PHRASES.items() if phrase in text}
        tp += len(active & found)
        fp += len(found - active)
        fn += len(active - found)
        n += 1
        if active - found:
            misses += 1
        if found - active:
            extra += 1

    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    out = {
        "generated_file": str(Path(args.generated).resolve()),
        "n": n, "tp": tp, "fp": fp, "fn": fn,
        "realization_precision": round(p, 6),
        "realization_recall": round(r, 6),
        "realization_f1": round(f1, 6),
        "rows_with_miss": misses, "rows_with_extra": extra,
    }
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()