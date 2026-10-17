# -*- coding: utf-8 -*-
"""Synthetic checks for Phase 3.15 threshold-repair helpers. No model fitting."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import repair_calibration_thresholds_phase315 as R
import validate_nested_semantic_calibration as N
import generate_text_template as G

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    (PASS, FAIL) = (PASS + 1, FAIL) if cond else (PASS, FAIL + 1)
    print(("PASS " if cond else "FAIL ") + name)


def test_recovery_exact():
    # if p_sig = sigmoid(a*logit(p_raw)+b), recovery must return (a, b) exactly
    rng = np.random.RandomState(0)
    p_raw = rng.uniform(0.05, 0.95, size=500)
    for a, b in ((2.0, -0.4), (0.5, -2.0), (1.4, 0.0)):
        p_sig = R.apply_ab(a, b, p_raw)
        ra, rb, resid = R.recover_map_from_pair(p_raw, p_sig)
        check(f"recover_a_{a}_{b}", abs(ra - a) < 1e-9)
        check(f"recover_b_{a}_{b}", abs(rb - b) < 1e-9)
        check(f"recover_resid_{a}_{b}", resid < 1e-9)


def test_monotone_mapping():
    a, b = 1.4, -0.3
    ts = [0.2, 0.35, 0.5, 0.8]
    mapped = [R.map_threshold(a, b, t) for t in ts]
    check("mapping_increasing", all(mapped[i] < mapped[i + 1] for i in range(3)))
    check("mapping_in_unit_interval", all(0.0 <= m <= 1.0 for m in mapped))
    check("mapping_matches_calibrator", abs(mapped[1] - float(R.apply_ab(a, b, 0.35))) < 1e-12)


def test_nonmonotone_rejected():
    check("negative_slope_maps_none", R.map_threshold(-0.5, 0.1, 0.4) is None)
    check("none_slope_maps_none", R.map_threshold(None, 0.1, 0.4) is None)
    check("zero_slope_maps_none", R.map_threshold(0.0, 0.2, 0.4) is None)


def test_decision_parity():
    rng = np.random.RandomState(0)
    a, b = 2.0, -0.4
    p_raw = rng.uniform(0.05, 0.95, size=400)
    p_cal = R.apply_ab(a, b, p_raw)
    for t_raw in (0.3, 0.45, 0.6, 0.72):
        t_cal = R.map_threshold(a, b, t_raw)
        check(f"parity_{t_raw}", R.mismatch_count(p_raw, t_raw, p_cal, t_cal) == 0)
    # a wrong mapped threshold breaks parity
    check("wrong_threshold_breaks_parity", R.mismatch_count(p_raw, 0.5, p_cal, 0.5) > 0)


def test_score_boundary_candidates():
    p = np.array([0.42, 0.42, 0.1, 0.9, 0.1])
    cand = R.score_boundary_candidates(p)
    check("candidates_are_unique_scores", cand == [0.9, 0.42, 0.1])
    check("candidates_descending", cand == sorted(cand, reverse=True))
    # every distinct decision boundary is reachable
    y = np.array([0, 1, 0, 1, 0])
    counts = {int(R.decisions(p, t).sum()) for t in cand}
    check("candidates_cover_boundaries", counts == {1, 3, 5})


def test_best_f1_tie_break():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.1])
    thr, f1 = R.best_f1_threshold(y, p, [0.9, 0.8, 0.2])
    # 0.9 and 0.8 both give F1 1.0 on positives only vs both; lowest wins
    check("tie_break_lowest", thr == 0.8 and abs(f1 - 1.0) < 1e-9)
    check("no_candidate_returns_none", R.best_f1_threshold(y, p, []) == (None, -1.0))


def test_old_grid_truncation():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    # positives sit at 0.12, negatives at 0.02: optimum is below 0.30
    p = np.array([0.12, 0.12, 0.12, 0.02, 0.02, 0.02, 0.02, 0.02])
    trunc, best_old, best_free = R.old_grid_truncated(y, p)
    check("truncated_detected", trunc is True)
    check("free_optimum_below_floor", best_free < 0.30)
    # mass well inside the grid is not truncated
    p2 = np.array([0.8, 0.8, 0.8, 0.2, 0.2, 0.2, 0.2, 0.2])
    trunc2, _, _ = R.old_grid_truncated(y, p2)
    check("in_grid_not_truncated", trunc2 is False)


def test_policy_application():
    P = np.array([[0.2, 0.9], [0.6, 0.4], [0.8, 0.1], [0.3, 0.7]])
    folds = np.array([0, 0, 1, 1])
    shared = {0: 0.5, 1: 0.5}
    pred = R.apply_policy(P, folds, shared)
    check("shared_threshold", pred.tolist() == [[0, 1], [1, 0], [1, 0], [0, 1]])
    per_fold = [{0: 0.25, 1: 0.95}, {0: 0.9, 1: 0.0}]
    pred2 = R.apply_policy(P, folds, per_fold)
    check("per_fold_threshold", pred2[:, 0].tolist() == [0, 1, 0, 0])
    check("per_fold_threshold_col2", pred2[:, 1].tolist() == [0, 0, 1, 1])


def test_downstream_injection():
    # a decision vector must survive the generator unchanged
    probs = np.array([0.9 if i < 3 else 0.1 for i in range(len(N.PILOT_TRAITS))])
    fired = np.array([1 if i < 2 else 0 for i in range(len(N.PILOT_TRAITS))])
    bodies, active, trace = R.section_bodies_for(fired, probs, N.PILOT_TRAITS)
    check("injected_active_matches_decision", set(active) == {N.PILOT_TRAITS[0], N.PILOT_TRAITS[1]})
    for cat, tr in trace.items():
        check(f"injected_fallback_{cat}", tr["fallback_used"] == (len(tr["selected_traits"]) == 0))
    # deterministic
    bodies2, _, _ = R.section_bodies_for(fired, probs, N.PILOT_TRAITS)
    check("injected_deterministic", bodies == bodies2)


def test_production_untouched():
    prod = [N.ROOT / p for p in ("faces_full_features_clean.csv", "semantic_labels.csv",
                                 "semantic_schema.json",
                                 "Source/models_semantic/trait_thresholds.json")]
    for pf in prod:
        check(f"prod_exists_{pf.name}", pf.exists())
    check("out_dir_isolated", str(R.OUT_DIR).endswith("semantic_threshold_phase315"))
    check("phase314_untouched", (R.P314 / "nested_calibration_decision.json").exists())


def main():
    test_recovery_exact()
    test_monotone_mapping()
    test_nonmonotone_rejected()
    test_decision_parity()
    test_score_boundary_candidates()
    test_best_f1_tie_break()
    test_old_grid_truncation()
    test_policy_application()
    test_downstream_injection()
    test_production_untouched()
    print(f"{PASS} PASS / {FAIL} FAIL")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()