"""
Semantic error analysis over cross-fitted predictions.

Inputs: ground-truth semantic labels, cross-fitted probabilities,
threshold policy, semantic schema. Pure stdlib + sklearn metrics.

Usage:
  python Source/analyze_semantic_errors.py
  python Source/analyze_semantic_errors.py --oof oof_semantic_probs.csv \
      --labels semantic_labels.csv --thresholds Source/models_semantic/trait_thresholds.json \
      --schema semantic_schema.json --output Source/semantic_error_analysis.json
"""
import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             f1_score, precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OOF = ROOT / "oof_semantic_probs.csv"
DEFAULT_LABELS = ROOT / "semantic_labels.csv"
DEFAULT_SCHEMA = ROOT / "semantic_schema.json"
DEFAULT_THRESHOLDS = Path(__file__).parent / "models_semantic" / "trait_thresholds.json"
DEFAULT_OUTPUT = Path(__file__).parent / "semantic_error_analysis.json"
DEFAULT_REPORT = Path(__file__).parent / "semantic_train_report.json"

CATEGORIES = ["personality", "career", "relationships", "future_destiny", "wealth"]
WEAK_AUC = 0.55
BUCKETS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]


def norm_name(v):
    s = os.path.basename(str(v).strip())
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\)\s+\(", ")(", s)
    return s.lower()


def load_threshold_map(path, trait_names):
    if path is None or not Path(path).exists():
        return {t: 0.5 for t in trait_names}, None
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    strategy = d.get("strategy", "unknown")
    tmap = {}
    if "trait_thresholds" in d:
        tmap = {t: float(d["trait_thresholds"].get(t, 0.5)) for t in trait_names}
    elif "thresholds" in d and isinstance(d["thresholds"], dict):
        th = d["thresholds"]
        if any(k in th for k in trait_names):
            tmap = {t: float(th.get(t, 0.5)) for t in trait_names}
        elif "global" in th:
            tmap = {t: float(th["global"]) for t in trait_names}
        elif any(c in th for c in CATEGORIES):
            schema = json.loads((ROOT / "semantic_schema.json").read_text(encoding="utf-8"))
            cat_map = {x["name"]: x["category"] for x in schema["traits"]}
            tmap = {t: float(th.get(cat_map[t], 0.5)) for t in trait_names}
        else:
            tmap = {t: 0.5 for t in trait_names}
    elif "global_threshold" in d:
        tmap = {t: float(d["global_threshold"]) for t in trait_names}
    else:
        tmap = {t: 0.5 for t in trait_names}
    return tmap, strategy


def ece_score(y_true, y_prob, n_bins=10):
    """Expected calibration error (equal-width bins)."""
    bins = np.linspace(0, 1, n_bins + 1)
    total = len(y_true)
    e = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        e += abs(acc - conf) * mask.sum() / total
    return float(e)


def main():
    ap = argparse.ArgumentParser(description="Semantic error analysis")
    ap.add_argument("--oof", default=str(DEFAULT_OOF))
    ap.add_argument("--labels", default=str(DEFAULT_LABELS))
    ap.add_argument("--schema", default=str(DEFAULT_SCHEMA))
    ap.add_argument("--thresholds", default=str(DEFAULT_THRESHOLDS))
    ap.add_argument("--train-report", default=str(DEFAULT_REPORT))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--threshold", type=float, default=None,
                    help="override single global threshold (ignores thresholds file)")
    args = ap.parse_args()

    schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
    trait_names = [t["name"] for t in schema["traits"]]
    trait_cat = {t["name"]: t["category"] for t in schema["traits"]}
    cat_idx = {c: [i for i, t in enumerate(trait_names) if trait_cat[t] == c]
               for c in CATEGORIES}

    if args.threshold is not None:
        thr_map = {t: float(args.threshold) for t in trait_names}
        strategy = f"global_{args.threshold}"
    else:
        thr_path = args.thresholds if args.thresholds and Path(args.thresholds).exists() else None
        thr_map, strategy = load_threshold_map(thr_path, trait_names)

    print(f"threshold strategy: {strategy}")
    print(f"traits: {len(trait_names)}, thresholds: "
          f"min {min(thr_map.values()):.3f} max {max(thr_map.values()):.3f}")

    labels = pd.read_csv(args.labels, encoding="utf-8-sig")
    img_col = [c for c in labels.columns if c.strip().lower().lstrip("\ufeff") == "image"][0]
    if img_col != "image":
        labels = labels.rename(columns={img_col: "image"})
    labels["norm_key"] = labels["image"].apply(norm_name)
    labels = labels.set_index("norm_key")

    oof = pd.read_csv(args.oof, encoding="utf-8-sig")
    oof["norm_key"] = oof["image_id"].apply(norm_name)
    # keep only rows with matching label
    oof["has_label"] = oof["norm_key"].isin(labels.index)
    oof_matched = oof[oof["has_label"]].copy()
    print(f"OOF {len(oof)} rows, matched {len(oof_matched)}, unmatched {len(oof) - len(oof_matched)}")
    assert len(oof_matched) > 0, "no overlapping rows"
    assert trait_names[0] in oof.columns, f"OOF missing trait {trait_names[0]}"
    assert (oof_matched[trait_names].values >= 0).all() and (oof_matched[trait_names].values <= 1).all(), "probs out of [0,1]"
    assert np.isfinite(oof_matched[trait_names].values).all(), "non-finite probs"

    y_true = np.array([labels.loc[k, trait_names].values.astype(int) for k in oof_matched["norm_key"]]).T
    y_prob = oof_matched[trait_names].values.T.astype(float)
    n_traits, n_rows = y_true.shape
    y_pred = np.zeros_like(y_true, dtype=int)
    for i, t in enumerate(trait_names):
        y_pred[i] = (y_prob[i] >= thr_map[t]).astype(int)

    # --- per-trait ---
    per_trait = {}
    for i, t in enumerate(trait_names):
        yt, yp, pp = y_true[i], y_pred[i], y_prob[i]
        tp = int(((yp == 1) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())
        tn = int(((yp == 0) & (yt == 0)).sum())
        try:
            auc = round(float(roc_auc_score(yt, pp)), 4)
        except Exception:
            auc = None
        try:
            prauc = round(float(average_precision_score(yt, pp)), 4)
        except Exception:
            prauc = None
        try:
            brier = round(float(brier_score_loss(yt, pp)), 4)
        except Exception:
            brier = None
        try:
            ece = round(ece_score(yt, pp, 10), 4)
        except Exception:
            ece = None
        per_trait[t] = {
            "category": trait_cat[t],
            "threshold": round(float(thr_map[t]), 4),
            "prevalence": round(float(yt.mean()), 4),
            "predicted_positive_rate": round(float(yp.mean()), 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(float(precision_score(yt, yp, zero_division=0)), 4),
            "recall": round(float(recall_score(yt, yp, zero_division=0)), 4),
            "f1": round(float(f1_score(yt, yp, zero_division=0)), 4),
            "roc_auc": auc,
            "pr_auc": prauc,
            "brier": brier,
            "ece": ece,
            "weak_signal": bool(auc is not None and auc < WEAK_AUC),
        }
        # probability buckets
        buckets = []
        for lo, hi in BUCKETS:
            if hi == 1.0:
                mask = (pp >= lo) & (pp <= hi)
            else:
                mask = (pp >= lo) & (pp < hi)
            cnt = int(mask.sum())
            rate = round(float(yt[mask].mean()), 4) if cnt else None
            buckets.append({"range": f"{lo:.1f}-{hi:.1f}", "count": cnt, "positive_rate": rate})
        per_trait[t]["buckets"] = buckets

    # --- per-category ---
    per_cat = {}
    for c in CATEGORIES:
        idx = cat_idx[c]
        ys, ps = y_true[idx], y_pred[idx]
        # macro over traits in category
        aucs = [per_trait[trait_names[i]]["roc_auc"] for i in idx if per_trait[trait_names[i]]["roc_auc"] is not None]
        prs = [per_trait[trait_names[i]]["pr_auc"] for i in idx if per_trait[trait_names[i]]["pr_auc"] is not None]
        briers = [per_trait[trait_names[i]]["brier"] for i in idx if per_trait[trait_names[i]]["brier"] is not None]
        f1s = [per_trait[trait_names[i]]["f1"] for i in idx]
        precs = [per_trait[trait_names[i]]["precision"] for i in idx]
        recs = [per_trait[trait_names[i]]["recall"] for i in idx]
        tp_sum = sum(per_trait[trait_names[i]]["tp"] for i in idx)
        fp_sum = sum(per_trait[trait_names[i]]["fp"] for i in idx)
        fn_sum = sum(per_trait[trait_names[i]]["fn"] for i in idx)
        tn_sum = sum(per_trait[trait_names[i]]["tn"] for i in idx)
        per_cat[c] = {
            "trait_count": len(idx),
            "traits": [trait_names[i] for i in idx],
            "macro_roc_auc": round(float(np.mean(aucs)), 4) if aucs else None,
            "macro_pr_auc": round(float(np.mean(prs)), 4) if prs else None,
            "mean_brier": round(float(np.mean(briers)), 4) if briers else None,
            "macro_precision": round(float(np.mean(precs)), 4),
            "macro_recall": round(float(np.mean(recs)), 4),
            "macro_f1": round(float(np.mean(f1s)), 4),
            "micro_precision": round(float(tp_sum / (tp_sum + fp_sum)) if tp_sum + fp_sum else 0, 4),
            "micro_recall": round(float(tp_sum / (tp_sum + fn_sum)) if tp_sum + fn_sum else 0, 4),
            "tp": tp_sum, "fp": fp_sum, "fn": fn_sum, "tn": tn_sum,
            "avg_true_traits_per_sample": round(float(ys.sum(0).mean()), 4),
            "avg_pred_traits_per_sample": round(float(ps.sum(0).mean()), 4),
        }

    # --- overall ---
    overall = {
        "macro_precision": round(float(np.mean([per_trait[t]["precision"] for t in trait_names])), 4),
        "macro_recall": round(float(np.mean([per_trait[t]["recall"] for t in trait_names])), 4),
        "macro_f1": round(float(np.mean([per_trait[t]["f1"] for t in trait_names])), 4),
        "micro_precision": round(float(y_true[y_pred == 1].size and 0), 4),
        "n_rows": int(n_rows),
        "n_traits": int(n_traits),
    }
    # correct micro
    tp_all = sum(per_trait[t]["tp"] for t in trait_names)
    fp_all = sum(per_trait[t]["fp"] for t in trait_names)
    fn_all = sum(per_trait[t]["fn"] for t in trait_names)
    overall["micro_precision"] = round(float(tp_all / (tp_all + fp_all)) if tp_all + fp_all else 0, 4)
    overall["micro_recall"] = round(float(tp_all / (tp_all + fn_all)) if tp_all + fn_all else 0, 4)
    overall["micro_f1"] = round(float(2 * overall["micro_precision"] * overall["micro_recall"]
                                      / (overall["micro_precision"] + overall["micro_recall"]))
                                 if overall["micro_precision"] + overall["micro_recall"] else 0, 4)
    overall["avg_true_traits_per_sample"] = round(float(y_true.sum(0).mean()), 4)
    overall["avg_pred_traits_per_sample"] = round(float(y_pred.sum(0).mean()), 4)

    # --- rankings ---
    fp_rank = sorted(trait_names, key=lambda t: -per_trait[t]["fp"])
    fn_rank = sorted(trait_names, key=lambda t: -per_trait[t]["fn"])
    by_f1 = sorted([t for t in trait_names if per_trait[t]["f1"] is not None],
                   key=lambda t: -per_trait[t]["f1"])
    by_auc = sorted([t for t in trait_names if per_trait[t]["roc_auc"] is not None],
                    key=lambda t: -per_trait[t]["roc_auc"])
    weak = [t for t in trait_names if per_trait[t]["weak_signal"]]

    # --- count distribution ---
    def dist(v):
        c = v.sum(0)
        d = {str(k): int((c == k).sum()) for k in [0, 1, 2, 3, 4]}
        d["5+"] = int((c >= 5).sum())
        return {"histogram": d, "mean": round(float(c.mean()), 3),
                "median": round(float(np.median(c)), 3),
                "p90": round(float(np.percentile(c, 90)), 3)}

    dists = {"true": dist(y_true), "predicted": dist(y_pred)}

    report = {
        "generated_by": "analyze_semantic_errors.py",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "oof": str(Path(args.oof).resolve()),
        "labels": str(Path(args.labels).resolve()),
        "schema": str(Path(args.schema).resolve()),
        "threshold_strategy": strategy,
        "n_rows": int(n_rows),
        "n_traits": int(n_traits),
        "overall": overall,
        "per_trait": per_trait,
        "per_category": per_cat,
        "rankings": {
            "most_fp": [{"trait": t, "fp": per_trait[t]["fp"],
                         "precision": per_trait[t]["precision"]} for t in fp_rank[:10]],
            "most_fn": [{"trait": t, "fn": per_trait[t]["fn"],
                         "recall": per_trait[t]["recall"]} for t in fn_rank[:10]],
            "best_by_f1": [{"trait": t, "f1": per_trait[t]["f1"],
                            "roc_auc": per_trait[t]["roc_auc"]} for t in by_f1[:10]],
            "worst_by_f1": [{"trait": t, "f1": per_trait[t]["f1"],
                             "roc_auc": per_trait[t]["roc_auc"]} for t in by_f1[-10:][::-1]],
            "best_by_auc": [{"trait": t, "roc_auc": per_trait[t]["roc_auc"],
                             "f1": per_trait[t]["f1"]} for t in by_auc[:10]],
            "worst_by_auc": [{"trait": t, "roc_auc": per_trait[t]["roc_auc"],
                              "f1": per_trait[t]["f1"]} for t in by_auc[-10:][::-1]],
        },
        "weak_signal_traits": weak,
        "weak_signal_criteria": f"roc_auc < {WEAK_AUC}",
        "trait_count_distribution": dists,
    }

    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"overall macro F1={overall['macro_f1']} P={overall['micro_precision']} "
          f"avg_pred {overall['avg_pred_traits_per_sample']}")
    print(f"weak {len(weak)}: {weak}")


if __name__ == "__main__":
    main()