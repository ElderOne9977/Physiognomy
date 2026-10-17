# -*- coding: utf-8 -*-
# Stage H focused tests for Phase 3.10 semantic label reliability
# Run: py Source/test_semantic_reliability_phase310.py
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import analyze_semantic_label_reliability as mod

PASS = []
FAIL = []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'} {name} {detail}")


def test_entropy():
    check("entropy p=0.5", abs(mod.entropy(0.5) - 1.0) < 1e-12)
    check("entropy p=0", mod.entropy(0) == 0.0)
    check("entropy p=1", mod.entropy(1) == 0.0)
    check("entropy bounds", 0 < mod.entropy(0.2) < 1)


def test_fin():
    check("fin normal", mod.fin("1.5") == 1.5)
    check("fin nan -> None", mod.fin(float("nan")) is None)
    check("fin inf -> None", mod.fin(float("inf")) is None)
    check("fin garbage -> None", mod.fin("abc") is None)


def test_norm_stem():
    check("norm lower", mod.norm_name("A (1).JPG") == "a (1).jpg")
    check("stem ext stripped", mod.stem_name("a (1).jpg") == "a (1)")
    check("stem equal across ext", mod.stem_name("a (1).jpeg") == mod.stem_name("a (1).jpg"))


def test_ece_score():
    import numpy as np
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    check("ece perfect low", mod.ece_score(y, p) <= 0.16, str(mod.ece_score(y, p)))
    p_bad = np.array([0.9, 0.9, 0.1, 0.1])
    check("ece bad high", mod.ece_score(y, p_bad) > 0.5)
    w = np.array([1.0, 1.0, 1.0, 1.0])
    check("ece weighted finite", math.isfinite(mod.ece_score(y, p, sample_weight=w)))


def test_bootstrap_reproducible():
    import numpy as np
    groups = {0: [0, 1], 1: [2], 2: [3, 4, 5]}
    comp_ids = list(groups.keys())

    def sample(seed):
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(20):
            ids = [comp_ids[k] for k in rng.integers(0, len(comp_ids), size=len(comp_ids))]
            idx = []
            for g in ids:
                idx.extend(groups[g])
            out.append(tuple(idx))
        return out

    check("bootstrap seed stable", sample(42) == sample(42))
    check("bootstrap seed differs", sample(42) != sample(1))


def test_fold_isolation_on_disk():
    p = mod.OUT_DIR / "artifact_integrity_report.json"
    if not p.exists():
        check("integrity report exists", False)
        return
    j = json.loads(p.read_text(encoding="utf-8"))
    lc = j.get("fold_integrity", {}).get("leakage_checks", {})
    check("all folds pass", lc.get("all_folds_pass") is True)
    for k, v in lc.get("per_fold", {}).items():
        check(f"{k} group overlap 0", v.get("group_overlap") == 0)
        check(f"{k} sha overlap 0", v.get("sha_overlap") == 0)
    c = j.get("counts", {})
    check("feature rows 6988", c.get("feature_rows") == 6988, str(c.get("feature_rows")))
    check("humans files 7472", c.get("humans_files") == 7472)
    check("traits 45", c.get("trait_count") == 45)
    check("components 3203", c.get("safe_group_n") == 3203)
    check("largest 610", c.get("audit_components", {}).get("largest_component_size") == 610)


def test_stage_b_conflicts():
    p = mod.OUT_DIR / "label_conflict_analysis.json"
    if not p.exists():
        check("conflict report exists", False)
        return
    j = json.loads(p.read_text(encoding="utf-8"))
    ex = j.get("exact_content_duplicates", {})
    check("dup groups 1452", ex.get("n_groups") == 1452, str(ex.get("n_groups")))
    check("all dup groups conflict", ex.get("conflicting_groups") == 1452)
    mean_c = ex.get("mean_conflicting_traits", 0)
    check("mean conflicting ~18.2", 15 < mean_c < 22, str(mean_c))
    check("n traits 45", j.get("dataset", {}).get("n_traits") == 45)


def test_stage_d_bootstrap_full():
    p = mod.OUT_DIR / "semantic_trait_performance_report.json"
    if not p.exists():
        check("stage D report exists", False)
        return
    j = json.loads(p.read_text(encoding="utf-8"))
    check("stage D bootstrap 1000", j.get("bootstrap", {}).get("n_bootstrap") == 1000, str(j.get("bootstrap")))
    macro = j.get("macro_roc_auc")
    check("macro roc matches corrected", abs(macro - 0.6437791161952097) < 1e-9, str(macro))
    check("n traits 45", j.get("n_traits") == 45)
    check("per-trait CIs present", all(t.get("roc_ci_low") is not None and t.get("roc_ci_high") is not None for t in j.get("weakest_traits", [])))


def test_all_outputs_finite_json():
    for fn in [
        "artifact_integrity_report.json",
        "label_conflict_analysis.json",
        "label_uncertainty_report.json",
        "semantic_trait_performance_report.json",
        "component_influence_analysis.json",
    ]:
        p = mod.OUT_DIR / fn
        if not p.exists():
            check(f"{fn} exists", False)
            continue
        txt = p.read_text(encoding="utf-8")
        check(f"{fn} no NaN/Infinity", "NaN" not in txt and "Infinity" not in txt)
        try:
            json.loads(txt)
            check(f"{fn} valid JSON", True)
        except Exception as e:
            check(f"{fn} valid JSON", False, str(e))


def test_trait_reliability_rows():
    p = mod.OUT_DIR / "trait_label_reliability.csv"
    if not p.exists():
        check("trait reliability csv exists", False)
        return
    lines = p.read_text(encoding="utf-8").splitlines()
    check("trait reliability 45 rows", len(lines) - 1 == 45, str(len(lines) - 1))
    check("trait reliability boot 1000", ",1000," in lines[1])


def test_component_influence_small():
    p = mod.OUT_DIR / "component_influence_analysis.json"
    if not p.exists():
        check("component influence exists", False)
        return
    j = json.loads(p.read_text(encoding="utf-8"))
    delta = abs(j.get("delta_full_vs_excl_largest", {}).get("macro_roc_delta", 1))
    check("excl largest delta < 0.01", delta < 0.01, str(delta))


def test_isolation():
    check("phase310 workspace separate", mod.OUT_DIR.name == "semantic_reliability_phase310")
    check("hash_grouped untouched path", mod.OOF_HASH_CSV.parent.name == "hash_grouped_semantic")
    check("production OOF path distinct", mod.OOF_CSV != mod.OOF_HASH_CSV)


def main():
    test_entropy()
    test_fin()
    test_norm_stem()
    test_ece_score()
    test_bootstrap_reproducible()
    test_fold_isolation_on_disk()
    test_stage_b_conflicts()
    test_stage_d_bootstrap_full()
    test_all_outputs_finite_json()
    test_trait_reliability_rows()
    test_component_influence_small()
    test_isolation()
    print(f"\nPASS {len(PASS)} FAIL {len(FAIL)}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())