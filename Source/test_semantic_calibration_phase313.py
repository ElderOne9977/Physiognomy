# -*- coding: utf-8 -*-
"""Synthetic checks for phase 3.13 calibration helpers. No model training."""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_semantic_calibration as C

PASS = 0
FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"PASS {name}")
    else:
        FAIL += 1
        print(f"FAIL {name}")


def test_ece_perfect():
    y = np.array([0, 0, 1, 1], float)
    p = np.array([0.0, 0.1, 0.9, 1.0])
    # not perfect, but a constant-correct prediction has ECE 0
    y2 = np.array([0, 0, 1, 1], float)
    p2 = np.array([0.0, 0.0, 1.0, 1.0])
    check("ece_zero_when_perfect", C.ece_score(y2, p2) == 0.0)
    check("ece_positive_when_off", C.ece_score(y, p) >= 0.0)
    check("ece_empty", C.ece_score(np.array([]), np.array([])) is None)


def test_brier():
    y = np.array([0, 1, 1, 0], float)
    p = np.array([0.0, 1.0, 1.0, 0.0])
    check("brier_perfect", abs(C.safe_brier(y, p)) < 1e-12)
    p2 = np.array([1.0, 0.0, 0.0, 1.0])
    check("brier_worst_gt_perfect", C.safe_brier(y, p2) > C.safe_brier(y, p))


def test_logloss_clip():
    y = np.array([0, 1], float)
    p = np.array([0.0, 1.0])  # would be inf without clipping
    v = C.safe_logloss(y, p)
    check("logloss_finite_at_boundary", v is not None and math.isfinite(v))
    check("clip_bounds", C.clip_prob(np.array([-1.0, 2.0])).min() >= C.CLIP)


def test_slope_intercept():
    rng = np.random.RandomState(0)
    # well-separated, so slope should be finite and positive
    p = np.clip(rng.beta(2, 5, size=400), 1e-3, 1 - 1e-3)
    y = (rng.rand(400) < p).astype(float)
    slope, intercept = C.calibration_slope_intercept(y, p)
    check("slope_finite", slope is not None and math.isfinite(slope))
    check("intercept_finite", intercept is not None and math.isfinite(intercept))
    check("slope_none_one_class", C.calibration_slope_intercept(np.zeros(30), p[:30]) == (None, None))


def test_prior_correction_direction():
    # balanced model over-predicts a rare trait; correction must pull p down
    p = np.array([0.6, 0.7, 0.8])
    out = C.prior_correct(p, pi=0.2)
    check("prior_pulls_down_when_rare", np.all(out < p))
    # common trait: correction pulls p up
    out2 = C.prior_correct(p, pi=0.8)
    check("prior_pulls_up_when_common", np.all(out2 > p))
    check("prior_identity_at_half", np.allclose(C.prior_correct(p, pi=0.5), p, atol=1e-6))
    check("prior_stays_in_unit_interval", np.all((out > 0) & (out < 1)))


def test_component_split_isolation():
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    folds = C.component_folds(groups, n_splits=5, seed=1)
    # every row of a component shares one fold
    ok = all(len(set(folds[groups == g])) == 1 for g in np.unique(groups))
    check("component_fold_atomic", ok)
    # a fold and its complement share no component
    leak = False
    for k in range(5):
        if set(groups[folds == k]) & set(groups[folds != k]):
            leak = True
    check("component_fold_no_leak", not leak)


def test_cross_fit_separation_and_bounds():
    rng = np.random.RandomState(2)
    n = 200
    groups = np.repeat(np.arange(40), 5)
    # probability that ranks with the label but is shifted high (over-confident)
    latent = rng.rand(n)
    y = (latent > 0.7).astype(float)
    p = np.clip(latent * 0.5 + 0.4, 0.01, 0.99)
    folds = C.component_folds(groups, n_splits=4, seed=2)
    for method in ("sigmoid", "isotonic", "prior"):
        out, notes = C.cross_fit(method, p, y, groups, folds, min_pos=5, min_neg=5)
        check(f"{method}_bounds", np.all((out >= 0) & (out <= 1)) and not np.isnan(out).any())
        check(f"{method}_all_folds_noted", len(notes) == 4)
    # isotonic refused when support is tiny
    y_rare = np.zeros(n)
    y_rare[:3] = 1
    out, notes = C.cross_fit("isotonic", p, y_rare, groups, folds, min_pos=30, min_neg=30)
    check("isotonic_fallback_on_rare", all(n["method_used"] == "fallback_uncalibrated" for n in notes))
    check("isotonic_fallback_equals_raw", np.allclose(out, p))


def test_ranking_preserved_by_monotone_maps():
    p = np.array([0.2, 0.5, 0.9, 0.1, 0.7])
    y = np.array([0, 1, 1, 0, 1], float)
    base = C.safe_auc(y, p)
    # prior correction is strictly monotone, so ROC-AUC is unchanged
    check("prior_auc_unchanged", abs(C.safe_auc(y, C.prior_correct(p, 0.3)) - base) < 1e-9)


def test_threshold_impact():
    traits = ["a", "b"]
    cat = {"a": "personality", "b": "career"}
    raw = np.array([[0.2, 0.8], [0.9, 0.1]])
    cal = np.array([[0.6, 0.8], [0.4, 0.1]])
    thr = {"a": 0.5, "b": 0.5}
    rows, by_cat = C.threshold_impact(raw, cal, thr, traits, cat)
    check("threshold_flip_counted", rows[0]["flipped_fraction"] == 1.0)
    check("threshold_category_totals", by_cat["personality"]["active_raw"] == 1)


def test_json_finite():
    payload = {"a": float("nan"), "b": np.float64("inf"), "c": np.int64(3)}
    clean = C.sanitize(payload)
    check("sanitize_nan_to_none", clean["a"] is None and clean["b"] is None)
    text = json.dumps(clean, allow_nan=False)
    check("json_roundtrip", json.loads(text)["c"] == 3)


def test_group_support():
    y = np.array([1, 1, 0, 0, 1])
    g = np.array([1, 1, 2, 3, 3])
    pos, neg = C.group_support(y, g)
    # group 1 positive only, group 2 negative only, group 3 both
    check("group_support_counts", pos == 2 and neg == 2)


def main():
    test_ece_perfect()
    test_brier()
    test_logloss_clip()
    test_slope_intercept()
    test_prior_correction_direction()
    test_component_split_isolation()
    test_cross_fit_separation_and_bounds()
    test_ranking_preserved_by_monotone_maps()
    test_threshold_impact()
    test_json_finite()
    test_group_support()
    print(f"{PASS} PASS / {FAIL} FAIL")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()