# -*- coding: utf-8 -*-
"""Real-image E2E: 3 generators, one extraction pipeline, no retrain.

Checks per image: all 3 modes emit valid 5-section text, generator metadata
provenance correct, semantic vector + feature CSV identical across modes
(generator must not touch upstream), warm per-stage CPU timings recorded.
Contrast pair: 2 real images -> per-method output difference report.

Usage: python Source/validate_generator_modes.py
Writes: outputs/validate_generator_modes.json
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from predict_image_to_text import (  # noqa: E402
    predict_image_to_text,
)

IMAGES = [ROOT / "Humans" / "1 (1).jpg", ROOT / "Humans" / "1 (10).jpg"]
OUT_JSON = ROOT / "outputs" / "validate_generator_modes.json"

# Scope pinned to the original four modes so this historical validator is
# reproducible; the controlled generators are covered by
# Source/test_controlled_generator.py and Source/evaluate_controlled_generator.py.
GENERATOR_CHOICES = ("flan_t5", "template_0.5", "template_calibrated", "template_concise")

EXPECTED_POLICY = {
    "flan_t5": "n/a",
    "template_0.5": "threshold_0.5",
    "template_calibrated": "per_trait",
    "template_concise": "per_trait",
    "controlled_deterministic": "per_trait",
    "controlled_flan_guarded": "per_trait",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def run_once(img: Path, generator: str, outdir: Path, warm: bool) -> dict:
    stem = f"{img.stem}_{generator}{'_warm' if warm else ''}"
    res = predict_image_to_text(
        img,
        features_output=str(outdir / f"{stem}_features.csv"),
        output=str(outdir / f"{stem}_result.json"),
        feature_source="extract",
        generator=generator,
    )
    return res


def main() -> int:
    for img in IMAGES:
        if not img.exists():
            print(f"missing image: {img}")
            return 2
    outdir = ROOT / "outputs"
    checks: list[dict] = []

    def check(name: str, cond: bool, detail=""):
        checks.append({"name": name, "pass": bool(cond), "detail": str(detail)[:500]})
        print(f"{'PASS' if cond else 'FAIL'} {name} {detail}")

    modes: dict[str, dict] = {}
    for img in IMAGES:
        per_image = {}
        for gen in GENERATOR_CHOICES:
            cold = run_once(img, gen, outdir, warm=False)
            warm = run_once(img, gen, outdir, warm=True)
            per_image[gen] = {"cold": cold, "warm": warm}
        modes[img.name] = per_image

    # -- per mode checks on primary image
    primary = IMAGES[0].name
    for gen in GENERATOR_CHOICES:
        res = modes[primary][gen]["warm"]
        g = res.get("generator") or {}
        check(f"{gen} valid 5-section", res["validation"]["valid"], str(res["validation"]))
        check(f"{gen} provenance method", g.get("method") == gen, str(g.get("method")))
        check(f"{gen} provenance threshold_policy",
              g.get("threshold_policy") == EXPECTED_POLICY[gen],
              str(g.get("threshold_policy")))
        check(f"{gen} 45 probs", len(res["semantic"]["probabilities"]) == 45,
              len(res["semantic"]["probabilities"]))

    # -- upstream unchanged across generators (same image, same extraction)
    ref = modes[primary][GENERATOR_CHOICES[0]]["warm"]
    ref_vec = ref["semantic"]["probabilities"]
    ref_csv = Path(ref["features"]["csv_path"])
    for gen in GENERATOR_CHOICES[1:]:
        res = modes[primary][gen]["warm"]
        vec = res["semantic"]["probabilities"]
        maxdiff = max(abs(vec[t] - ref_vec[t]) for t in ref_vec)
        check(f"vector identical {gen} vs flan", maxdiff == 0.0, f"maxdiff={maxdiff}")
        csv = Path(res["features"]["csv_path"])
        check(f"csv path per-mode {gen}", csv != ref_csv, csv.name)
    # CSV content identical? extraction runs independently; compare hashes of warm CSVs
    csv_hashes = {gen: sha256(Path(modes[primary][gen]["warm"]["features"]["csv_path"]))
                  for gen in GENERATOR_CHOICES}
    check("csv bytes identical across generators",
          len(set(csv_hashes.values())) == 1, str(csv_hashes))

    # -- contrast pair: does each method react to a different vector?
    second = IMAGES[1].name
    diffs = {}
    for gen in GENERATOR_CHOICES:
        a = modes[primary][gen]["warm"]
        b = modes[second][gen]["warm"]
        from text_gen_common import parse_sections
        sa, _ = parse_sections(a["text"])
        sb, _ = parse_sections(b["text"])
        changed = sum(1 for k in sa if sa.get(k, "").strip() != sb.get(k, "").strip())
        vec_maxdiff = max(abs(a["semantic"]["probabilities"][t] - b["semantic"]["probabilities"][t])
                          for t in a["semantic"]["probabilities"])
        diffs[gen] = {"sections_changed": changed, "vector_maxdiff": round(vec_maxdiff, 6)}
        check(f"{gen} vector differs across images", vec_maxdiff > 0.0, f"maxdiff={vec_maxdiff:.4f}")
    # template must change text when vector changes (collapse fix); FLAN may not - report only
    check("template_0.5 text changes across images", diffs["template_0.5"]["sections_changed"] > 0,
          diffs["template_0.5"])
    check("template_concise text changes across images",
          diffs["template_concise"]["sections_changed"] > 0, diffs["template_concise"])
    check("template_calibrated text changes across images",
          diffs["template_calibrated"]["sections_changed"] > 0, diffs["template_calibrated"])

    # -- warm timings per stage per method
    timings = {}
    for gen in GENERATOR_CHOICES:
        res = modes[primary][gen]["warm"]
        timings[gen] = res["timing"]
        print(f"warm {gen}: {res['timing']}")

    payload = {
        "checks": checks,
        "all_pass": all(c["pass"] for c in checks),
        "images": [str(i) for i in IMAGES],
        "warm_timings_primary": timings,
        "contrast_pair": diffs,
        "cold_timings": {gen: modes[primary][gen]["cold"]["timing"] for gen in GENERATOR_CHOICES},
    }
    # fin(): no NaN/Infinity
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
                        encoding="utf-8")
    print(f"wrote {OUT_JSON} all_pass={payload['all_pass']} "
          f"({sum(c['pass'] for c in checks)}/{len(checks)})")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())