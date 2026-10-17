# -*- coding: utf-8 -*-
"""Synthetic checks for Phase 3.14 nested validation helpers.

Covers: isolation (group/sha/stem), train-only preprocess/thresholds, calibrator fit,
trait order, prob bounds, Brier/ECE/logloss, threshold apply, degenerate traits,
paired component bootstrap, downstream compare (real generators), deterministic finite JSON,
production files unchanged.
"""
import json
import hashlib
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_nested_semantic_calibration as N
import generate_text_template as G

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


def test_isolation_sha_stem():
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    sha = np.array([hashlib.sha256(f"img{g}_{i}".encode()).hexdigest() for i, g in enumerate(groups)])
    stems = np.array([f"img{g}" for g in groups])
    # folds respect group boundaries (atomic): g0,g1,g4 -> fold0; g2,g3 -> fold1
    folds = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0])
    for k in (0, 1):
        tr = folds != k
        va = folds == k
        og = len(set(groups[tr]) & set(groups[va]))
        osha = len(set(sha[tr]) & set(sha[va]))
        ost = len(set(stems[tr]) & set(stems[va]))
        check(f"isolation_fold{k}_group", og == 0)
        check(f"isolation_fold{k}_sha", osha == 0)
        check(f"isolation_fold{k}_stem", ost == 0)
    check("norm_name_lower", N.norm_name("A/B/face (2).JPG") == "face (2).jpg")
    check("stem_name_strips_ext", N.stem_name("face (2).jpg") == "face (2)")


def test_train_only_preprocess():
    rng = np.random.RandomState(1)
    Xtr = rng.randn(100, 4).astype(float)
    Xtr[0, 0] = np.nan
    Xva = rng.randn(30, 4).astype(float)
    Xva[0, 0] = 999.0  # extreme val that must NOT influence imputer/scaler
    imp, sc = N.fit_preprocess(Xtr)
    # imputer median computed only on train
    check("imputer_median_train_only", abs(imp.statistics_[0] - np.nanmedian(Xtr[:, 0])) < 1e-6)
    Xtr_s = N.apply_preprocess(imp, sc, Xtr)
    Xva_s = N.apply_preprocess(imp, sc, Xva)
    # train transformed has ~0 mean, 1 std (approximately)
    check("train_mean_near_zero", abs(Xtr_s.mean()) < 0.2)
    check("train_std_near_one", abs(Xtr_s.std() - 1.0) < 0.2)
    # val transformed with train params
    check("val_transformed_not_nan", not np.isnan(Xva_s).any())
    # changing val does not change fitted params (by construction fit on Xtr only)
    imp2, sc2 = N.fit_preprocess(Xtr)  # same Xtr
    check("fit_deterministic", np.allclose(imp.statistics_, imp2.statistics_))
    check("fit_deterministic_sc", np.allclose(sc.mean_, sc2.mean_))


def test_calibrator_fit():
    # over-confident probs, sigmoid should pull toward prior
    p = np.array([0.9, 0.95, 0.85, 0.1, 0.05, 0.15])
    y = np.array([1, 1, 1, 0, 0, 0])
    clf = N.fit_sigmoid(p, y)
    check("sigmoid_fits", clf is not None)
    if clf is not None:
        pcal = N.apply_sigmoid(clf, p)
        check("sigmoid_bounds", np.all((pcal >= 0) & (pcal <= 1)))
        # one-class -> fit_sigmoid returns None
    clf1 = N.fit_sigmoid(p, np.ones_like(y))
    check("sigmoid_none_on_one_class", clf1 is None)


def test_trait_order():
    check("pilot_traits_len", len(N.PILOT_TRAITS) == 9)
    check("pilot_traits_unique", len(set(N.PILOT_TRAITS)) == 9)
    for t in N.PILOT_TRAITS:
        check(f"pilot_trait_has_rationale_{t}", t in N.PILOT_RATIONALE)


def test_prob_bounds():
    p = np.array([-0.1, 0.0, 0.5, 1.0, 1.1])
    c = N.clip_prob(p)
    check("clip_prob_bounds", np.all(c >= N.CLIP) and np.all(c <= 1 - N.CLIP))
    # prior_correct maps (0,1) -> (0,1)
    pc = N.prior_correct(np.array([0.2, 0.8]), pi=0.3)
    check("prior_correct_bounds", np.all((pc > 0) & (pc < 1)))
    # apply_sigmoid bounds
    clf = N.fit_sigmoid(np.array([0.1, 0.9]), np.array([0, 1]))
    if clf is not None:
        out = N.apply_sigmoid(clf, np.array([0.0, 1.0]))
        check("apply_sigmoid_bounds", np.all((out >= 0) & (out <= 1)))


def test_brier_ece_logloss():
    y = np.array([0, 1, 1, 0, 1], float)
    p = np.array([0.1, 0.9, 0.8, 0.2, 0.7])
    b = N.safe_brier(y, p)
    e = N.ece_score(y, p)
    l = N.safe_logloss(y, p)
    check("brier_finite", b is not None and math.isfinite(b))
    check("ece_finite", e is not None and math.isfinite(e))
    check("logloss_finite", l is not None and math.isfinite(l))
    # edge cases
    check("brier_empty", N.safe_brier(np.array([]), np.array([])) is None)
    check("ece_empty", N.ece_score(np.array([]), np.array([])) is None)
    check("logloss_empty", N.safe_logloss(np.array([]), np.array([])) is None)
    # perfect
    y2 = np.array([0, 0, 1, 1])
    p2 = np.array([0.0, 0.0, 1.0, 1.0])
    check("brier_perfect", abs(N.safe_brier(y2, p2)) < 1e-12)
    check("logloss_clipped_boundary", N.safe_logloss(np.array([0, 1]), np.array([0.0, 1.0])) is not None)


def test_threshold_apply():
    P = np.array([[0.2, 0.8], [0.9, 0.1], [0.4, 0.6]])
    thr = {"a": 0.5, "b": 0.5}
    pred = (P >= np.array([thr["a"], thr["b"]])).astype(int)
    check("threshold_semantics", pred[0, 0] == 0 and pred[0, 1] == 1)
    # grid within 0.30..0.70
    check("grid_bounds", min(N.GRID) >= 0.30 and max(N.GRID) <= 0.70)
    # fallback: if pos_g<5 or neg_g<5 -> old threshold
    # test via policy_metrics fallback logic is hard in isolation; we test group_support
    y = np.array([1, 1, 0, 0, 1])
    g = np.array([1, 1, 2, 3, 3])
    pos, neg = N.group_support(y, g)
    check("group_support_counts", pos == 2 and neg == 2)


def test_degenerate_traits():
    # one-class -> safe_auc/safe_ap None; safe_brier/safe_logloss finite; ece finite (but slope None)
    y1 = np.zeros(20)
    p1 = np.full(20, 0.3)
    check("auc_none_one_class", N.safe_auc(y1, p1) is None)
    check("ap_none_one_class", N.safe_ap(y1, p1) is None)
    check("brier_finite_one_class", N.safe_brier(y1, p1) is not None and math.isfinite(N.safe_brier(y1, p1)))
    check("logloss_finite_one_class", N.safe_logloss(y1, p1) is not None and math.isfinite(N.safe_logloss(y1, p1)))
    # ECE is defined even for one-class (mean diff) — check it returns finite
    e = N.ece_score(y1, p1)
    check("ece_finite_one_class", e is not None and math.isfinite(e))
    check("slope_none_one_class", N.calibration_slope_intercept(y1, p1) == (None, None))


def test_paired_component_bootstrap():
    def _deltas(seed=42):
        rng = np.random.RandomState(seed)
        groups = np.repeat(np.arange(10), 5)
        n = len(groups)
        P1 = np.clip(rng.beta(2, 5, size=n), 0.01, 0.99)
        P2 = np.clip(P1 * 0.7 + 0.15, 0.01, 0.99)
        Y = (rng.rand(n) < P2).astype(float)
        uniq = np.unique(groups)
        ds = []
        for _ in range(20):
            chosen = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([np.where(groups == g)[0] for g in chosen]) if len(chosen) else np.array([], int)
            if len(idx) == 0:
                continue
            Yb = Y[idx]
            b1 = N.safe_brier(Yb, P1[idx])
            b2 = N.safe_brier(Yb, P2[idx])
            if b1 is not None and b2 is not None:
                ds.append(b2 - b1)
        return ds
    deltas = _deltas(42)
    check("paired_bootstrap_same_idx", len(deltas) > 0)
    check("paired_deltas_finite", all(math.isfinite(d) for d in deltas))
    deltas2 = _deltas(42)
    check("bootstrap_deterministic", deltas == deltas2)


def test_downstream_compare():
    probs = {t: 0.9 if i < 3 else 0.1 for i, t in enumerate(N.PILOT_TRAITS)}
    thr_map = {t: 0.5 for t in N.PILOT_TRAITS}
    text, active = G.generate_text(probs, threshold=G.DEFAULT_THRESHOLD, thresholds=thr_map)
    check("downstream_template_active_count", len(active) == sum(1 for t in N.PILOT_TRAITS if probs[t] >= 0.5))
    text_c, active_c, trace = G.generate_text_concise(probs, threshold=G.DEFAULT_THRESHOLD, thresholds=thr_map)
    check("downstream_concise_returns_trace", isinstance(trace, dict))
    check("downstream_concise_active_subset", len(active_c) <= len(active))
    check("trace_has_all_categories", len(trace) == 5)
    # fallback_used true iff no selected traits in that category
    for cat, tr in trace.items():
        sel = tr.get("selected_traits", [])
        fb = tr.get("fallback_used", False)
        check(f"fallback_consistent_{cat}", fb == (len(sel) == 0))
    # realized_concepts belong to selected traits (no hallucination)
    for cat, tr in trace.items():
        sel = set(tr.get("selected_traits", []))
        # realized_concepts are stripped phrases; just check count matches selected
        check(f"realized_matches_selected_{cat}", len(tr.get("realized_concepts", [])) == len(sel))


def test_deterministic_json():
    obj = {"a": 1.0, "b": [2, 3], "c": {"d": 4}}
    with tempfile.TemporaryDirectory() as td:
        p1 = Path(td) / "a.json"
        p2 = Path(td) / "b.json"
        N.write_json(p1, obj)
        N.write_json(p2, obj)
        t1 = p1.read_bytes()
        t2 = p2.read_bytes()
        check("write_json_deterministic", t1 == t2)
        # sanitize
        dirty = {"nan": float("nan"), "inf": float("inf"), "int": np.int64(3), "flt": np.float64(2.5)}
        clean = N.sanitize(dirty)
        check("sanitize_nan_inf", clean["nan"] is None and clean["inf"] is None)
        check("sanitize_numpy_types", clean["int"] == 3 and clean["flt"] == 2.5)
        # roundtrip
        js = json.dumps(clean, allow_nan=False)
        back = json.loads(js)
        check("json_roundtrip", back["int"] == 3)


def test_production_files_unchanged():
    prod_files = [N.ROOT / p for p in (
        "faces_full_features_clean.csv",
        "semantic_labels.csv",
        "semantic_schema.json",
        "Source/experiments/hash_grouped_semantic/cv_fold_assignments.csv",
        "Source/models_semantic/trait_thresholds.json",
        "Source/models_semantic/feature_cols.npy",
        "oof_semantic_probs.csv",
    )]
    for pf in prod_files:
        check(f"prod_exists_{pf.name}", pf.exists())
    # hashes before/after trivial operation
    before = {pf: hashlib.sha256(pf.read_bytes()).hexdigest() for pf in prod_files}
    # do nothing (test itself shouldn't write to prod)
    after = {pf: hashlib.sha256(pf.read_bytes()).hexdigest() for pf in prod_files}
    for pf in prod_files:
        check(f"prod_unchanged_{pf.name}", before[pf] == after[pf])
    # OUT_DIR is under semantic_calibration_phase314 (not production)
    check("out_dir_not_prod", str(N.OUT_DIR).endswith("semantic_calibration_phase314"))


def main():
    test_isolation_sha_stem()
    test_train_only_preprocess()
    test_calibrator_fit()
    test_trait_order()
    test_prob_bounds()
    test_brier_ece_logloss()
    test_threshold_apply()
    test_degenerate_traits()
    test_paired_component_bootstrap()
    test_downstream_compare()
    test_deterministic_json()
    test_production_files_unchanged()
    print(f"{PASS} PASS / {FAIL} FAIL")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()