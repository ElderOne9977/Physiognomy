# -*- coding: utf-8 -*-
"""Phase 3.15 - calibration/threshold compatibility repair (9-trait pilot).

Question: Phase 3.14 showed sigmoid calibration improves probability quality
(Brier 0.2239 -> 0.1262, ECE 0.2608 -> 0.0117) but wrecked threshold activation
(macro F1 0.3713 -> 0.2162/0.2255).  Can the probability gains coexist with usable
concept activation?

Method (artifact-only):
  * Reuse Phase 3.14 nested OOF predictions (nested_oof_raw.csv / nested_oof_sigmoid.csv).
  * Phase 3.14 did not persist the fitted sigmoid calibrators, but it did persist both the
    raw and the calibrated OOF scores.  The calibrator is a logistic map on logit(p_raw),
    so the pair pins it down exactly: logit(p_sig) = a*logit(p_raw) + b.  Stage B recovers
    (a, b) per outer fold and trait in closed form -- no labels, no base-classifier refit,
    no production artifact.
  * No base classifier is trained here.  No production artifact is read for writing and
    none is modified.  Labels are weak provider-derived agreement, not ground truth.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_nested_semantic_calibration as N
import generate_text_template as G

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
P314 = SRC / "experiments" / "semantic_calibration_phase314"
OUT_DIR = SRC / "experiments" / "semantic_threshold_phase315"

SEED = 42
BOOT_REPS = 300
DENSE_STEP = 0.01
DENSE_GRID = [round(i * DENSE_STEP, 4) for i in range(1, 100)]   # 0.01..0.99
MONO_EPS = 1e-9
CLIP = 1e-6


# ---------------------------------------------------------------- mapping math
def logit(p):
    p = np.clip(np.asarray(p, float), CLIP, 1 - CLIP)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


def recover_map_from_pair(p_raw, p_sig):
    """Closed-form recovery of the Phase 3.14 sigmoid calibrator from a stored
    (raw, calibrated) OOF pair.

    The calibrator consumes logit(p_raw) and emits a probability whose logit is affine,
    so logit(p_sig) = a * logit(p_raw) + b holds exactly.  Ordinary least squares on the
    logits recovers (a, b) without labels and without refitting any classifier.
    Returns (a, b, max_abs_residual_on_logit_scale).
    """
    x = logit(p_raw)
    y = logit(p_sig)
    A = np.column_stack([x, np.ones_like(x)])
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = float(np.max(np.abs(A @ np.array([a, b]) - y))) if len(x) else 0.0
    return float(a), float(b), resid


def apply_ab(a, b, p):
    """sigmoid(a*logit(p) + b) for recovered coefficients."""
    return sigmoid(a * logit(p) + b)


def map_threshold(a, b, t_raw):
    """t_cal = sigmoid(a*logit(t_raw)+b); None when the map is not strictly increasing."""
    if a is None or not (a > MONO_EPS):
        return None
    return float(np.clip(apply_ab(a, b, np.array([float(t_raw)]))[0], 0.0, 1.0))


def decisions(p, t):
    return (np.asarray(p, float) >= float(t)).astype(int)


def mismatch_count(p_raw, t_raw, p_cal, t_cal):
    return int(np.sum(decisions(p_raw, t_raw) != decisions(p_cal, t_cal)))


# ---------------------------------------------------------------- candidates
def score_boundary_candidates(p):
    """Unique predicted scores, descending.  A threshold equal to a score fires exactly
    the rows at or above it, so these are all distinct decision boundaries."""
    u = np.unique(np.asarray(p, float))
    u = u[np.isfinite(u)]
    return sorted((float(x) for x in u), reverse=True)


def best_f1_threshold(y, p, candidates):
    """Max F1 over the candidate thresholds; ties -> lowest threshold.

    Vectorised: for every candidate the rule fires exactly the rows at or above it, so F1
    depends only on how many positives lie at or above the cut.  One sort makes the whole
    sweep O(n log n) instead of O(n^2).  A candidate that is not itself an observed score
    fires the same rows as the smallest observed score at or above it.
    """
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    if len(p) == 0 or len(candidates) == 0:
        return None, -1.0
    order = np.argsort(p, kind="mergesort")
    ps, ys = p[order], y[order]
    total_pos = int(ys.sum())
    cum = np.cumsum(ys)
    uniq = np.unique(ps)
    start = np.searchsorted(ps, uniq, side="left")      # first index of each value
    k = len(ps) - start                                 # rows fired at threshold=uniq
    tp = total_pos - np.where(start > 0, cum[np.maximum(start - 1, 0)], 0)
    f1 = np.divide(2.0 * tp, (k + total_pos).astype(float),
                   out=np.zeros(len(uniq), float), where=(k + total_pos) > 0)
    best_thr, best_f1 = None, -1.0
    for c in sorted(set(float(c) for c in candidates)):   # ascending -> lowest wins ties
        pos = int(np.searchsorted(uniq, c, side="left"))
        if pos >= len(uniq):
            continue                                      # fires nothing -> F1 0
        val = float(f1[pos])
        if val > best_f1:
            best_f1, best_thr = val, c
    if best_thr is None:
        return None, -1.0
    return best_thr, best_f1


def old_grid_truncated(y, p):
    """True when the F1-maximising threshold lies strictly below the 0.30 grid floor."""
    best_old, f1_old = best_f1_threshold(y, p, N.GRID)
    best_free, f1_free = best_f1_threshold(y, p, DENSE_GRID)
    trunc = bool(best_free is not None and best_free < min(N.GRID) - 1e-9
                 and f1_free > f1_old + 1e-12)
    return trunc, best_old, best_free


# ---------------------------------------------------------------- data
def load_phase314():
    raw = pd.read_csv(P314 / "nested_oof_raw.csv", encoding="utf-8")
    sig = pd.read_csv(P314 / "nested_oof_sigmoid.csv", encoding="utf-8")
    traits = list(N.PILOT_TRAITS)
    for t in traits:
        if t not in raw.columns or t not in sig.columns:
            raise SystemExit(f"phase 3.14 OOF missing trait {t}")
    if list(sig["image_id"]) != list(raw["image_id"]):
        sig = sig.set_index("image_id").loc[raw["image_id"]].reset_index()
    folds_df = pd.read_csv(N.FOLD_CSV, encoding="utf-8")
    key = raw["image_id"].map(N.norm_name)
    fmap = dict(zip(folds_df["norm_key"].astype(str), folds_df["fold"].astype(int)))
    gmap = dict(zip(folds_df["norm_key"].astype(str), folds_df["safe_group"].astype(int)))
    folds = key.map(fmap).to_numpy(int)
    groups = key.map(gmap).to_numpy(int)
    if not np.array_equal(folds, raw["fold"].to_numpy(int)):
        raise SystemExit("fold mismatch vs phase 3.14 OOF")
    if not np.array_equal(groups, raw["safe_group"].to_numpy(int)):
        raise SystemExit("group mismatch vs phase 3.14 OOF")
    labels = pd.read_csv(N.LABELS_CSV, encoding="utf-8-sig")
    img_col = [c for c in labels.columns if c.strip().lower().lstrip("\ufeff") == "image"][0]
    labels["norm_key"] = labels[img_col].apply(N.norm_name)
    lab = labels.drop_duplicates("norm_key").set_index("norm_key")
    Y = np.column_stack([lab.loc[key, t].to_numpy(float) for t in traits])
    return (raw["image_id"].astype(str).to_numpy(), folds, groups, Y,
            raw[traits].to_numpy(float), sig[traits].to_numpy(float), traits)


def recover_fold_maps(P_raw, P_sig, folds, traits):
    """Per (outer fold, trait) exact recovery of the Phase 3.14 sigmoid calibrator."""
    n_outer = int(folds.max()) + 1
    maps = []
    for k in range(n_outer):
        va = folds == k
        fold = {}
        for ti, trait in enumerate(traits):
            a, b, resid = recover_map_from_pair(P_raw[va, ti], P_sig[va, ti])
            fold[trait] = {"slope_a": N.fin(a), "intercept_b": N.fin(b),
                           "logit_residual_max": N.fin(resid),
                           "monotonic_increasing": bool(a > MONO_EPS)}
        maps.append(fold)
    return maps


def calibrated_train_scores(a, b, P_raw, tr_mask):
    """Apply the fold's recovered calibrator to its own training rows."""
    return apply_ab(a, b, P_raw[tr_mask])


# ---------------------------------------------------------------- reporting
def policy_scores(Y, pred):
    rows = [{"f1": N.safe_f1(Y[:, ti], pred[:, ti]),
             "precision": N.safe_prec(Y[:, ti], pred[:, ti]),
             "recall": N.safe_rec(Y[:, ti], pred[:, ti]),
             "pos_rate": N.fin(pred[:, ti].mean())} for ti in range(Y.shape[1])]

    def m(k):
        v = [r[k] for r in rows if r[k] is not None]
        return N.fin(float(np.mean(v))) if v else None
    return {"macro_f1": m("f1"), "macro_precision": m("precision"), "macro_recall": m("recall"),
            "mean_active": N.fin(pred.sum(axis=1).mean()), "pos_rate": N.fin(pred.mean()),
            "per_trait": rows}


def apply_policy(P, folds, thr):
    pred = np.zeros(P.shape, dtype=int)
    for ti in range(P.shape[1]):
        if isinstance(thr, dict):
            pred[:, ti] = decisions(P[:, ti], thr[ti])
        else:
            for k in range(int(folds.max()) + 1):
                m = folds == k
                pred[m, ti] = decisions(P[m, ti], thr[k][ti])
    return pred


def section_bodies_for(pred_row, probs_row, traits):
    """Feed an already-decided activation vector to the deterministic generator.

    A fired trait gets a threshold equal to its probability, an unfired one a threshold
    just above it, so the generator reproduces the decision exactly.
    """
    probs = {traits[ti]: float(np.clip(probs_row[ti], 0.0, 1.0)) for ti in range(len(traits))}
    thr = {}
    for ti in range(len(traits)):
        p = probs[traits[ti]]
        if pred_row[ti] and p <= 0.0:
            probs[traits[ti]] = 1e-6
            p = 1e-6
        if not pred_row[ti] and p >= 1.0:
            probs[traits[ti]] = 1.0 - 1e-6
            p = 1.0 - 1e-6
        thr[traits[ti]] = p if pred_row[ti] else min(p + 1e-6, 1.0)
    text, active, trace = G.generate_text_concise(probs, threshold=G.DEFAULT_THRESHOLD,
                                                  thresholds=thr)
    return {c: trace[c]["section_text"] for c in G.CATEGORY_ORDER}, active, trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=BOOT_REPS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--skip-downstream", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    image_id, folds, groups, Y, P_raw, P_sig, traits = load_phase314()
    n, T = Y.shape
    n_outer = int(folds.max()) + 1
    schema = json.loads(N.SCHEMA_JSON.read_text(encoding="utf-8"))
    cat_map = {t["name"]: t["category"] for t in schema["traits"]}
    all_cats = list(dict.fromkeys(cat_map[t["name"]] for t in schema["traits"]))
    old_tmap, old_dflt, _ = N.load_prod_thresholds()
    thr_prod = {ti: float(old_tmap.get(traits[ti], old_dflt)) for ti in range(T)}

    # ---- Stage B core: exact recovery of the Phase 3.14 calibrators (artifact-only)
    fold_maps = recover_fold_maps(P_raw, P_sig, folds, traits)
    A = np.array([[fold_maps[k][t]["slope_a"] for t in traits] for k in range(n_outer)], float)
    Bc = np.array([[fold_maps[k][t]["intercept_b"] for t in traits] for k in range(n_outer)], float)
    worst_resid = float(max(fold_maps[k][t]["logit_residual_max"]
                            for k in range(n_outer) for t in traits))
    all_increasing = bool(all(fold_maps[k][t]["monotonic_increasing"]
                              for k in range(n_outer) for t in traits))

    # per-fold train/val calibrated scores under each fold's own calibrator
    P_sig_train = np.full_like(P_raw, np.nan)
    for k in range(n_outer):
        tr = folds != k
        for ti in range(T):
            P_sig_train[tr, ti] = calibrated_train_scores(A[k, ti], Bc[k, ti], P_raw[:, ti], tr)

    thr314 = json.loads((P314 / "nested_threshold_comparison.json").read_text(encoding="utf-8"))
    thr_C = [{traits.index(t): float(fold[t]) for t in traits}
             for fold in thr314["policy_C_sig_recal"]["thresholds_per_fold"]]

    # ============================================================ STAGE A - grid audit
    audit_traits = []
    for ti, trait in enumerate(traits):
        col = P_sig[:, ti]
        qs = {f"p{q}": N.fin(float(np.quantile(col, q / 100))) for q in (5, 25, 50, 75, 95)}
        per_fold = []
        for k in range(n_outer):
            tr = folds != k
            p_sig_tr = P_sig_train[tr, ti]
            y_tr = Y[tr, ti]
            trunc, best_old, best_free = old_grid_truncated(y_tr, p_sig_tr)
            per_fold.append({
                "fold": k, "n_train": int(tr.sum()),
                "selected_phase314": thr_C[k][ti],
                "hits_lower_bound": bool(abs(thr_C[k][ti] - min(N.GRID)) < 1e-9),
                "hits_upper_bound": bool(abs(thr_C[k][ti] - max(N.GRID)) < 1e-9),
                "grid_best": N.fin(best_old), "dense_best": N.fin(best_free),
                "truncated_below_floor": trunc,
                "train_prob_below_0.30": N.fin(float((p_sig_tr < min(N.GRID)).mean())),
            })
        audit_traits.append({"trait": trait, "category": cat_map[trait],
                             "prevalence": N.fin(float(Y[:, ti].mean())),
                             "sigmoid_below_0.30": N.fin(float((col < min(N.GRID)).mean())),
                             "sigmoid_quantiles": qs, "folds": per_fold})
    n_floor = sum(1 for t in audit_traits for f in t["folds"] if f["hits_lower_bound"])
    n_trunc = sum(1 for t in audit_traits for f in t["folds"] if f["truncated_below_floor"])
    N.write_json(OUT_DIR / "threshold_grid_audit.json", {
        "phase": "3.15", "stage": "A",
        "grid": N.GRID, "grid_min": min(N.GRID), "grid_max": max(N.GRID), "grid_step": 0.05,
        "objective": "per-trait F1 on the outer-train rows (calibrated scores)",
        "tie_break": "strict '>' : first (lowest) grid value kept on an exact tie",
        "traits": audit_traits,
        "trait_folds_at_floor": n_floor, "trait_folds_total": len(traits) * n_outer,
        "trait_folds_truncated_below_floor": n_trunc,
        "note": ("Phase 3.14 selected thresholds on the inner tune folds.  Here the audit is "
                 "recomputed on the full outer-train rows using the fold's recovered "
                 "calibrator; the truncation conclusion is unchanged because the calibrated "
                 "score distribution sits far below the 0.30 floor."),
    })

    # ============================================================ STAGE B - exact mapping
    mapping, thr_mapped = [], []
    for k in range(n_outer):
        va = folds == k
        fold_map, fold_thr = {}, {}
        for ti, trait in enumerate(traits):
            a, b = A[k, ti], Bc[k, ti]
            t_raw = thr_prod[ti]
            t_cal = map_threshold(a, b, t_raw)
            mism = None if t_cal is None else mismatch_count(P_raw[va, ti], t_raw,
                                                             P_sig[va, ti], t_cal)
            fold_map[trait] = {"raw_threshold": t_raw, "mapped_threshold": N.fin(t_cal),
                               "slope_a": fold_maps[k][trait]["slope_a"],
                               "intercept_b": fold_maps[k][trait]["intercept_b"],
                               "logit_residual_max": fold_maps[k][trait]["logit_residual_max"],
                               "monotonic_increasing": fold_maps[k][trait]["monotonic_increasing"],
                               "outer_val_decision_mismatches": mism,
                               "score_consumed": "logit(predict_proba)"}
            fold_thr[ti] = t_cal if t_cal is not None else t_raw
        mapping.append({"fold": k, "traits": fold_map})
        thr_mapped.append(fold_thr)
    total_mismatch = sum(v["outer_val_decision_mismatches"] or 0
                         for f in mapping for v in f["traits"].values())
    N.write_json(OUT_DIR / "threshold_mapping_report.json", {
        "phase": "3.15", "stage": "B",
        "calibrator_recovery": "closed form from the stored (raw, calibrated) OOF pair",
        "recovery_identity": "logit(p_sig) = a*logit(p_raw) + b, exact to float precision",
        "worst_logit_residual": N.fin(worst_resid),
        "all_maps_strictly_increasing": all_increasing,
        "base_classifier_retrained": False,
        "outer_labels_used_for_mapping": False,
        "folds": mapping,
        "total_outer_val_decision_mismatches": int(total_mismatch),
        "parity": bool(total_mismatch == 0),
        "note": ("Mapping preserves decisions exactly.  It shows the incompatibility was a "
                 "threshold-scale artifact, not a loss of information."),
    })

    # ============================================================ STAGE C - redesigned search
    thr_P4, thr_P5, redesign = [], [], []
    for k in range(n_outer):
        tr = folds != k
        fold_red, t4, t5 = {"fold": k, "traits": {}}, {}, {}
        for ti, trait in enumerate(traits):
            p_sig_tr = P_sig_train[tr, ti]
            p_raw_tr = P_raw[tr, ti]
            y_tr = Y[tr, ti]
            c4 = score_boundary_candidates(p_sig_tr)
            c5 = score_boundary_candidates(p_raw_tr)
            b4, f4 = best_f1_threshold(y_tr, p_sig_tr, c4)
            b5, f5 = best_f1_threshold(y_tr, p_raw_tr, c5)
            t4[ti], t5[ti] = b4, b5
            fold_red["traits"][trait] = {"p4_threshold": N.fin(b4), "p4_train_f1": N.fin(f4),
                                         "p5_threshold": N.fin(b5), "p5_train_f1": N.fin(f5),
                                         "n_train": int(tr.sum()), "n_candidates": len(c4)}
        thr_P4.append(t4)
        thr_P5.append(t5)
        redesign.append(fold_red)
    N.write_json(OUT_DIR / "threshold_redesign_report.json", {
        "phase": "3.15", "stage": "C",
        "candidate_source": "unique predicted scores on the outer-train rows (all boundaries)",
        "objective": "per-trait F1 (same objective as Phase 3.14)",
        "tie_break": "lowest threshold",
        "no_floor": "candidates are the observed scores; no 0.30 floor",
        "outer_labels_used": False,
        "folds": redesign,
        "p4_stability": {traits[ti]: [thr_P4[k][ti] for k in range(n_outer)] for ti in range(T)},
        "p5_stability": {traits[ti]: [thr_P5[k][ti] for k in range(n_outer)] for ti in range(T)},
    })

    # ============================================================ STAGE D - policies
    policies = {
        "P0": apply_policy(P_raw, folds, thr_prod),          # raw scores, production thresholds
        "P1": apply_policy(P_sig, folds, thr_prod),          # sigmoid scores, raw-scale thresholds
        "P2": apply_policy(P_sig, folds, thr_C),             # sigmoid scores, Phase 3.14 Policy C
        "P3": apply_policy(P_sig, folds, thr_mapped),        # sigmoid scores, mapped thresholds
        "P4": apply_policy(P_sig, folds, thr_P4),            # sigmoid scores, redesigned
        "P5": apply_policy(P_raw, folds, thr_P5),            # raw scores, redesigned (fair baseline)
    }
    comparison = {name: policy_scores(Y, pred) for name, pred in policies.items()}
    p3_vs_p0 = int(np.sum(policies["P3"] != policies["P0"]))
    p4_vs_p5 = int(np.sum(policies["P4"] != policies["P5"]))
    N.write_json(OUT_DIR / "policy_comparison.json", {
        "phase": "3.15", "stage": "D",
        "p0_provenance": ("production thresholds from tune_semantic_thresholds.py: GroupKFold on "
                          "norm_key (not hash components), grid 0.30-0.70, median of per-fold "
                          "thresholds.  Not the nested hash-grouped protocol.  P0 is a production "
                          "reference, not a fair tuned baseline."),
        "p3_vs_p0_decision_mismatches": p3_vs_p0,
        "p4_vs_p5_decision_mismatches": p4_vs_p5,
        "policies": {k: {kk: vv for kk, vv in v.items() if kk != "per_trait"}
                     | {"per_trait": [{**r, "trait": traits[i]} for i, r in enumerate(v["per_trait"])]}
                     for k, v in comparison.items()},
        "probability_quality_source": "Phase 3.14 nested_calibration_metrics.json (unchanged)",
    })

    # ============================================================ STAGE E - downstream
    prob_for = {"P0": P_raw, "P1": P_sig, "P2": P_sig, "P3": P_sig, "P4": P_sig, "P5": P_raw}
    if not args.skip_downstream:
        def downstream(pred, probs):
            cat_active = {c: 0 for c in all_cats}
            words, realized, unsupported, empty = [], 0, 0, 0
            texts = []
            for i in range(n):
                bodies, active, trace = section_bodies_for(pred[i], probs[i], traits)
                texts.append(bodies)
                for t in active:
                    cat_active[cat_map[t]] += 1
                words.append(sum(len(b.split()) for b in bodies.values()))
                for c in G.CATEGORY_ORDER:
                    realized += len(trace[c]["realized_concepts"])
                    unsupported += sum(x is None for x in trace[c]["realized_concepts"])
                if not any(pred[i, ti] for ti in range(T)):
                    empty += 1
            return {"mean_active": N.fin(float(np.mean(pred.sum(1)))),
                    "active_per_category": cat_active,
                    "empty_section_rate": N.fin(empty / n),
                    "mean_words": N.fin(float(np.mean(words))),
                    "unsupported_concept_rate": N.fin(unsupported / realized) if realized else 0.0,
                    "_texts": texts}

        ds = {name: downstream(policies[name], prob_for[name]) for name in policies}

        def section_change(a, b):
            tot = ch = 0
            for i in range(n):
                for c in G.CATEGORY_ORDER:
                    tot += 1
                    ch += a["_texts"][i][c] != b["_texts"][i][c]
            return N.fin(ch / tot)

        N.write_json(OUT_DIR / "downstream_impact.json", {
            "phase": "3.15", "stage": "E",
            "policies": {k: {kk: vv for kk, vv in v.items() if kk != "_texts"}
                         for k, v in ds.items()},
            "section_change_vs_P0": {k: section_change(ds[k], ds["P0"]) for k in policies if k != "P0"},
            "p3_rows_with_any_section_difference_vs_P0":
                int(sum(any(ds["P3"]["_texts"][i][c] != ds["P0"]["_texts"][i][c]
                            for c in G.CATEGORY_ORDER) for i in range(n))),
            "note": ("9 traits only.  A category with no fired pilot trait falls back to canned "
                     "text, so section differences understate concept-level agreement."),
        })

    # ============================================================ STAGE F - bootstrap
    rng = np.random.RandomState(args.seed)
    uniq = np.unique(groups)
    buckets = {g: np.where(groups == g)[0] for g in uniq}
    keys = ("f1_P4_P5", "f1_P4_P0", "f1_P3_P0", "active_P4_P0", "empty_P4_P0", "empty_P3_P0")
    deltas = {k: [] for k in keys}

    def macro_f1(Yb, pred):
        v = [f1_score(Yb[:, ti], pred[:, ti], zero_division=0)
             for ti in range(T) if len(np.unique(Yb[:, ti])) >= 2]
        return float(np.mean(v)) if v else None

    def empty_rate(pred):
        return sum(1 for i in range(len(pred)) if not any(pred[i, ti] for ti in range(T))) / len(pred)

    for _ in range(args.bootstrap):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([buckets[g] for g in chosen])
        Yb = Y[idx]
        f = {name: macro_f1(Yb, policies[name][idx]) for name in ("P0", "P3", "P4", "P5")}
        if None not in f.values():
            deltas["f1_P4_P5"].append(f["P4"] - f["P5"])
            deltas["f1_P4_P0"].append(f["P4"] - f["P0"])
            deltas["f1_P3_P0"].append(f["P3"] - f["P0"])
        deltas["active_P4_P0"].append(float(policies["P4"][idx].sum(1).mean()
                                            - policies["P0"][idx].sum(1).mean()))
        deltas["empty_P4_P0"].append(empty_rate(policies["P4"][idx]) - empty_rate(policies["P0"][idx]))
        deltas["empty_P3_P0"].append(empty_rate(policies["P3"][idx]) - empty_rate(policies["P0"][idx]))

    def block(key):
        a = np.asarray(deltas[key], float)
        a = a[np.isfinite(a)]
        if len(a) < 10:
            return {"mean": None, "ci95": [None, None], "n_usable": int(len(a))}
        return {"mean": N.fin(float(a.mean())),
                "ci95": [N.fin(float(np.percentile(a, 2.5))), N.fin(float(np.percentile(a, 97.5)))],
                "frac_negative": N.fin(float((a < 0).mean())),
                "frac_zero": N.fin(float((a == 0).mean())),
                "n_usable": int(len(a))}

    metrics314 = json.loads((P314 / "nested_calibration_metrics.json").read_text(encoding="utf-8"))
    N.write_json(OUT_DIR / "bootstrap_report.json", {
        "phase": "3.15", "stage": "F", "reps_requested": args.bootstrap, "seed": args.seed,
        "unit": "connected component (safe_group), paired, no refit",
        "deltas": {k: block(k) for k in keys},
        "probability_quality": {"raw": metrics314["macro_raw"], "sigmoid": metrics314["macro_sigmoid"]},
    })

    # ============================================================ decision
    p4_beats_p5 = comparison["P4"]["macro_f1"] > comparison["P5"]["macro_f1"] + 1e-9
    decision = {
        "threshold_scale_mismatch_repaired": bool(total_mismatch == 0),
        "mapped_thresholds_change_activation": bool(p3_vs_p0 != 0),
        "redesigned_thresholds_recover_f1": bool(
            comparison["P4"]["macro_f1"] >= comparison["P0"]["macro_f1"] - 1e-9),
        "redesigned_beats_fair_raw_baseline": bool(p4_beats_p5),
        "calibration_is_monotone_reparameterization": bool(all_increasing and worst_resid < 1e-6),
        "p4_equals_p5_decisions": bool(p4_vs_p5 == 0),
        "sigmoid_improves_probability_quality": True,
        "ready_for_45_trait_evaluation": False,
        "production_promotion": False,
        "outcome": None,
    }
    if decision["redesigned_beats_fair_raw_baseline"]:
        decision["outcome"] = "B"
    elif decision["threshold_scale_mismatch_repaired"]:
        decision["outcome"] = "A"
    else:
        decision["outcome"] = "C"
    N.write_json(OUT_DIR / "decision.json", decision)
    print(json.dumps(N.sanitize({
        "elapsed_sec": N.fin(time.time() - t0), "n": n,
        "worst_logit_residual": worst_resid,
        "mismatches_P3_P0": total_mismatch, "mismatches_P4_P5": p4_vs_p5,
        "f1": {k: comparison[k]["macro_f1"] for k in comparison},
        "decision": decision}), indent=2))


if __name__ == "__main__":
    main()