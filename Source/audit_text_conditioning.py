"""
Frozen-checkpoint conditioning audit for FLAN-T5-small.

Question: does the trained checkpoint actually depend on the 45-d semantic
probability input, or does it mostly emit provider/template priors?

Phases (run in order; default = all):
  conditions : 5 input conditions x 702 test rows -> Source/generated_text_test_*.jsonl
  ablation   : per-category train-mean mask       -> Source/text_conditioning_category_ablation.json
  sweep      : 7 traits x 6 levels x 100 rows     -> Source/text_generator_sensitivity_sweep.json
  report     : metrics + deltas + decision gate   -> Source/text_conditioning_audit.json

Usage:
  python Source/audit_text_conditioning.py
  python Source/audit_text_conditioning.py --phase report
  python Source/audit_text_conditioning.py --phase conditions --limit 8 --outdir <temp>

Frozen (never changed by this script):
  model dir Source/models_text/flan_t5_small, tokenizer, text_generation_splits.json,
  prompt format (text_gen_common.format_input / INSTRUCTION), greedy decoding
  (max_new_tokens=256, do_sample=False, num_beams=1), BS=16, MAX_SOURCE_LEN=320,
  seed=42, semantic schema, OOF probabilities, target provider text.
Only the semantic input vector changes between conditions. No training here.
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

sys.path.insert(0, str(Path(__file__).parent))
from text_gen_common import (
    load_schema, format_input, parse_sections, validate_structure,
    CATEGORY_ORDER, CATEGORY_TITLES, INSTRUCTION,
)
from evaluate_text_model import norm_name, rouge_l_one, bleu_corpus, load_threshold_map, CALIB_PATH
from build_semantic_labels import TRAIT_PATTERNS

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).parent
DEFAULT_MODEL = SRC / "models_text" / "flan_t5_small"
DEFAULT_TRAIN = ROOT / "text_generation_train.jsonl"
DEFAULT_SPLITS = SRC / "text_generation_splits.json"

SEED = 42
BS = 16              # identical to generate_text_model.py
MAX_SOURCE_LEN = 320
MAX_NEW_TOKENS = 256

CONDITIONS = ["NORMAL", "SHUFFLED", "CONSTANT_MEAN", "CONSTANT_05", "ZERO"]
COND_FILE = {
    "NORMAL": "generated_text_test_normal.jsonl",
    "SHUFFLED": "generated_text_test_shuffled.jsonl",
    "CONSTANT_MEAN": "generated_text_test_constant_mean.jsonl",
    "CONSTANT_05": "generated_text_test_constant_05.jsonl",
    "ZERO": "generated_text_test_zero.jsonl",
}

def _cond_file(cond, representation):
    base = COND_FILE[cond]
    if representation == "bins":
        return base.replace(".jsonl", "_bins.jsonl")
    return base

SECTION_TITLES = [CATEGORY_TITLES[c] for c in CATEGORY_ORDER]

# Sensitivity sweep: representative schema traits across all 5 categories.
SWEEP_TRAITS = ["warm", "confident", "research", "trust_based",
                "opportunities", "steady_growth", "conservative"]
SWEEP_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
SWEEP_N = 100

# Documented interpretation thresholds (stored in the report JSON).
THRESH = {
    "delta_rouge_strong": 0.03,
    "delta_rouge_moderate": 0.01,
    "delta_semf1_strong": 0.05,
    "own_section_changed_strong": 0.50,
    "own_section_changed_weak": 0.25,
    "sweep_clear_concept_delta": 0.25,
    "sweep_clear_norm_change": 0.50,
}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def mean(xs):
    return float(sum(xs) / len(xs)) if xs else 0.0


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def norm_text(t):
    return re.sub(r"\s+", " ", (t or "").strip().lower())


_CONCEPT_CACHE = {}


def extract_concepts(text):
    """Canonical lexicon extraction (same regexes that built semantic_labels.csv)."""
    if text in _CONCEPT_CACHE:
        return _CONCEPT_CACHE[text]
    found = {t for t, (_cat, pats) in TRAIT_PATTERNS.items()
             if any(re.search(p, text, re.I) for p in pats)}
    if len(_CONCEPT_CACHE) < 500000:
        _CONCEPT_CACHE[text] = found
    return found


def prf(pred, true):
    """Macro P/R/F1 with the convention of evaluate_text_generation.py."""
    tp, fp, fn = len(pred & true), len(pred - true), len(true - pred)
    p = tp / (tp + fp) if (tp + fp) else 1.0
    r = tp / (tp + fn) if (tp + fn) else 1.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


# ---------------------------------------------------------------- data loading

def load_frozen_rows():
    """Load schema, splits, all rows; return (rows_test, trait_names, cat_of, splits)."""
    trait_names, cat_of = load_schema()
    schema = json.loads((ROOT / "semantic_schema.json").read_text(encoding="utf-8"))
    assert [t["name"] for t in schema["traits"]] == trait_names, "schema trait order mismatch"
    splits = json.loads(DEFAULT_SPLITS.read_text(encoding="utf-8"))
    test_set = set(splits["test_images"])
    train_set = set(splits["train_images"])
    val_set = set(splits["val_images"])
    # no group overlap between splits (norm_name keys)
    assert not (train_set & test_set), "train/test overlap"
    assert not (val_set & test_set), "val/test overlap"
    assert not (train_set & val_set), "train/val overlap"
    rows = load_jsonl(DEFAULT_TRAIN)
    test_rows = [r for r in rows if norm_name(r["image_id"]) in test_set]  # file order = frozen order
    assert len(test_rows) == 702, f"expected 702 test rows, got {len(test_rows)}"
    for r in test_rows:
        probs = r["trait_probs"]
        assert len(probs) == 45, f"dim != 45 for {r['image_id']}"
        assert list(probs.keys()) == trait_names, f"trait order mismatch for {r['image_id']}"
        for t in trait_names:
            v = float(probs[t])
            assert np.isfinite(v) and 0.0 <= v <= 1.0, f"prob out of range {t}={v}"
        assert r["target_text"].strip(), f"empty target for {r['image_id']}"
    return rows, test_rows, trait_names, cat_of, splits, train_set


def train_mean_vector(all_rows, train_set, trait_names):
    """Mean probability vector from TRAINING split rows only."""
    train_rows = [r for r in all_rows if norm_name(r["image_id"]) in train_set]
    assert train_rows, "no training rows for mean vector"
    mat = np.array([[float(r["trait_probs"][t]) for t in trait_names] for r in train_rows])
    assert mat.shape[1] == 45
    assert np.isfinite(mat).all()
    return mat.mean(axis=0), len(train_rows)


def build_condition_vectors(test_rows, trait_names, train_mean):
    """Original vectors plus the four counterfactual conditions."""
    normal = [[float(r["trait_probs"][t]) for t in trait_names] for r in test_rows]
    for vec in normal:
        assert len(vec) == 45 and all(np.isfinite(vec)) and all(0.0 <= v <= 1.0 for v in vec)
    n = len(test_rows)
    # SHUFFLED: seeded derangement, no row receives its own vector.
    rng = np.random.RandomState(SEED)
    idx = np.arange(n)
    perm = None
    for _ in range(10000):
        cand = rng.permutation(n)
        if not (cand == idx).any():
            perm = cand
            break
    assert perm is not None, "derangement not found"
    for i in range(n):
        src = int(perm[i])
        assert norm_name(test_rows[src]["image_id"]) != norm_name(test_rows[i]["image_id"]), \
            f"shuffle fixed point at row {i}"
    conditions = {
        "NORMAL": normal,
        "SHUFFLED": [normal[int(perm[i])][:] for i in range(n)],
        "CONSTANT_MEAN": [[float(v) for v in train_mean] for _ in range(n)],
        "CONSTANT_05": [[0.5] * 45 for _ in range(n)],
        "ZERO": [[0.0] * 45 for _ in range(n)],
    }
    shuffle_sources = [test_rows[int(perm[i])]["image_id"] for i in range(n)]
    for name, vecs in conditions.items():
        for vec in vecs:
            assert len(vec) == 45 and all(np.isfinite(vec)) and all(0.0 <= v <= 1.0 for v in vec)
    return conditions, shuffle_sources


def vec_to_probs(vec, trait_names):
    return {t: float(v) for t, v in zip(trait_names, vec)}


# ---------------------------------------------------------------- generation

def load_model(model_dir):
    log(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"gpu={torch.cuda.get_device_name(0)}")
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    torch.manual_seed(SEED)
    return model, tok, device


def generate_texts(model, tok, device, prompts):
    """Greedy decode, identical settings to generate_text_model.py."""
    texts = []
    with torch.no_grad():
        for i in range(0, len(prompts), BS):
            chunk = prompts[i:i + BS]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=MAX_SOURCE_LEN)
            gen = model.generate(**{k: v.to(device) for k, v in enc.items()},
                                 max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1)
            texts.extend(tok.batch_decode(gen, skip_special_tokens=True))
            if (i // BS) % 15 == 0:
                log(f"  batch {i // BS + 1}/{(len(prompts) + BS - 1) // BS}")
    return texts
def build_prompts(vecs, test_rows, trait_names, cat_of, representation="continuous"):
    prompts, probs_dicts = [], []
    for vec, row in zip(vecs, test_rows):
        probs = vec_to_probs(vec, trait_names)
        prompt = format_input(probs, trait_names, cat_of, representation=representation)
        # no target text in model input
        assert row["target_text"].strip() not in prompt, "target leaked into prompt"
        assert prompt.startswith(INSTRUCTION), "prompt must start with frozen instruction"
        if representation == "bins":
            assert f"{trait_names[0]}=P" in prompt, "bins representation missing P-labels"
        prompts.append(prompt)
        probs_dicts.append(probs)
    return prompts, probs_dicts


def write_condition_jsonl(path, test_rows, texts, probs_dicts, condition, sources=None):
    with open(path, "w", encoding="utf-8") as f:
        for i, (row, text, probs) in enumerate(zip(test_rows, texts, probs_dicts)):
            rec = {"image_id": row["image_id"], "generated_text": text, "probs": probs,
                   "condition": condition}
            if sources is not None:
                rec["source_image_id"] = sources[i]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log(f"Wrote {len(texts)} rows -> {path}")


# ---------------------------------------------------------------- evaluation

def evaluate_condition(gen_texts, test_rows, tgt_full, tgt_secs, trait_names, thr_cal):
    n = len(gen_texts)
    assert n == len(test_rows)
    rouge_ps, rouge_rs, rouge_fs = [], [], []
    hyp_toks, ref_toks = [], []
    tgt_conc = tgt_full
    gen_conc = []
    valid_count = missing_total = empty_total = malformed = 0
    sem_ps, sem_rs, sem_fs = [], [], []
    in05_ps, in05_rs, in05_fs, in05_unsup = [], [], [], 0
    incal_ps, incal_rs, incal_fs, incal_unsup = [], [], [], 0
    schema_unsup = 0
    per_sec = {t: {"rouge": [], "p": [], "r": [], "f": []} for t in SECTION_TITLES}
    schema_set = set(trait_names)

    for text, row, t_full, t_secs in zip(gen_texts, test_rows, tgt_full, tgt_secs):
        v = validate_structure(text)
        valid_count += int(v["valid"])
        missing_total += len(v["missing_sections"])
        empty_total += len(v["empty_sections"])
        malformed += int(not v["valid"])

        p, r, f = rouge_l_one(text, row["target_text"])
        rouge_ps.append(p); rouge_rs.append(r); rouge_fs.append(f)
        hyp_toks.append(text.lower().split())
        ref_toks.append(row["target_text"].lower().split())

        g_con = extract_concepts(text)
        gen_conc.append(g_con)
        schema_unsup += len(g_con - schema_set)
        sp, sr, sf = prf(g_con, t_full)
        sem_ps.append(sp); sem_rs.append(sr); sem_fs.append(sf)

        vec = row["_audit_probs"]
        sup05 = {t for t in trait_names if vec[t] >= 0.5}
        supcal = {t for t in trait_names if vec[t] >= thr_cal.get(t, 0.5)}
        a, b, c = prf(g_con, sup05)
        in05_ps.append(a); in05_rs.append(b); in05_fs.append(c)
        in05_unsup += len(g_con - sup05)
        a, b, c = prf(g_con, supcal)
        incal_ps.append(a); incal_rs.append(b); incal_fs.append(c)
        incal_unsup += len(g_con - supcal)

        g_secs, _ = parse_sections(text)
        for title in SECTION_TITLES:
            gb = g_secs.get(title, "")
            tb = t_secs.get(title, "")
            gp, gr, gf = rouge_l_one(gb, tb)
            per_sec[title]["rouge"].append(gf)
            sp2, sr2, sf2 = prf(extract_concepts(gb), extract_concepts(tb))
            per_sec[title]["p"].append(sp2)
            per_sec[title]["r"].append(sr2)
            per_sec[title]["f"].append(sf2)

    bleu, bleu_sm, bleu1, _prec, _bp = bleu_corpus(hyp_toks, ref_toks, max_n=4)
    return {
        "n": n,
        "rougeL_precision_mean": mean(rouge_ps),
        "rougeL_recall_mean": mean(rouge_rs),
        "rougeL_f1_mean": mean(rouge_fs),
        "bleu1": float(bleu1),
        "bleu4": float(bleu),
        "bleu4_smoothed": float(bleu_sm),
        "structure": {
            "valid_rate": valid_count / n,
            "valid_count": valid_count,
            "missing_sections_total": missing_total,
            "empty_sections_total": empty_total,
            "malformed_count": malformed,
        },
        "target_semantic": {
            "precision_mean": mean(sem_ps),
            "recall_mean": mean(sem_rs),
            "f1_mean": mean(sem_fs),
        },
        "input_faithfulness": {
            "threshold_0.5": {
                "precision_mean": mean(in05_ps),
                "recall_mean": mean(in05_rs),
                "f1_mean": mean(in05_fs),
                "unsupported_concepts_total": in05_unsup,
            },
            "calibrated": {
                "precision_mean": mean(incal_ps),
                "recall_mean": mean(incal_rs),
                "f1_mean": mean(incal_fs),
                "unsupported_concepts_total": incal_unsup,
            },
        },
        "unsupported_schema_concepts_total": schema_unsup,
        "n_unique_texts": len({t for t in gen_texts}),
        "per_section": {
            title: {
                "rougeL_f1_mean": mean(per_sec[title]["rouge"]),
                "semantic_precision_mean": mean(per_sec[title]["p"]),
                "semantic_recall_mean": mean(per_sec[title]["r"]),
                "semantic_f1_mean": mean(per_sec[title]["f"]),
            } for title in SECTION_TITLES
        },
    }


def change_rates(a_texts, b_texts):
    n = len(a_texts)
    return {
        "exact_changed_rate": mean([1.0 if x != y else 0.0 for x, y in zip(a_texts, b_texts)]),
        "normalized_changed_rate": mean([1.0 if norm_text(x) != norm_text(y) else 0.0
                                          for x, y in zip(a_texts, b_texts)]),
        "concepts_changed_rate": mean([1.0 if extract_concepts(x) != extract_concepts(y) else 0.0
                                       for x, y in zip(a_texts, b_texts)]),
    }


def precompute_targets(test_rows):
    tgt_full = [extract_concepts(r["target_text"]) for r in test_rows]
    tgt_secs = [parse_sections(r["target_text"])[0] for r in test_rows]
    return tgt_full, tgt_secs


# ---------------------------------------------------------------- phases

def phase_conditions(args, trait_names, cat_of, test_rows, train_mean):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    conds, shuffle_sources = build_condition_vectors(test_rows, trait_names, train_mean)
    model = tok = device = None
    rep = getattr(args, "representation", "continuous")
    for cond in CONDITIONS:
        out_path = outdir / _cond_file(cond, rep)
        expected = len(test_rows) if not args.limit else min(args.limit, len(test_rows))
        if out_path.exists() and not args.force:
            existing = load_jsonl(out_path)
            if len(existing) == expected:
                log(f"skip {cond}: {out_path} already has {len(existing)} rows")
                continue
        if model is None:
            model, tok, device = load_model(args.model)
        vecs = conds[cond][:expected]
        rows_sub = test_rows[:expected]
        prompts, probs_dicts = build_prompts(vecs, rows_sub, trait_names, cat_of, representation=rep)
        log(f"condition {cond}: {len(prompts)} prompts, generating on {device} ...")
        t0 = time.time()
        texts = generate_texts(model, tok, device, prompts)
        log(f"condition {cond}: done in {time.time() - t0:.0f}s")
        sources = None
        if cond == "SHUFFLED":
            sources = shuffle_sources[:expected]
            for i in range(len(rows_sub)):
                assert norm_name(sources[i]) != norm_name(rows_sub[i]["image_id"])
        write_condition_jsonl(out_path, rows_sub, texts, probs_dicts, cond, sources)
    return outdir


def phase_ablation(args, trait_names, cat_of, test_rows, train_mean, splits, train_set):
    outdir = Path(args.outdir)
    rep = getattr(args, "representation", "continuous")
    suffix = "_bins" if rep == "bins" else ""
    normal_path = outdir / _cond_file("NORMAL", rep)
    assert normal_path.exists(), "run conditions phase first"
    normal_recs = load_jsonl(normal_path)
    assert len(normal_recs) in (702, args.limit or 702), "normal jsonl incomplete"
    normal_texts = [r["generated_text"] for r in normal_recs]
    n = len(normal_texts)
    rows_sub = test_rows[:n]
    # input-faithfulness in evaluate_condition reads row["_audit_probs"]:
    # use the NORMAL condition's input vector as baseline for section metrics.
    for rec, row in zip(normal_recs, rows_sub):
        row["_audit_probs"] = rec["probs"]
    tgt_full, tgt_secs = precompute_targets(rows_sub)

    out_path = outdir / f"text_conditioning_category_ablation{suffix}.json"
    state = {"n_test": n, "mask_source": "train-split mean per category (train_images only)",
             "baseline": "NORMAL condition outputs", "per_category": {}}
    if out_path.exists() and not args.force:
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        if prev.get("n_test") == n:
            state = prev

    # baseline per-section metrics for deltas
    if "normal_per_section" not in state:
        base = evaluate_condition(normal_texts, rows_sub, tgt_full, tgt_secs,
                                  trait_names, _cal_thresholds())
        state["normal_per_section"] = base["per_section"]
        out_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    todo = [c for c in CATEGORY_ORDER if c not in state["per_category"]]
    if todo:
        model, tok, device = load_model(args.model)
    for cat in todo:
        vecs = []
        for r in rows_sub:
            vec = [float(r["trait_probs"][t]) for t in trait_names]
            for j, t in enumerate(trait_names):
                if cat_of[t] == cat:
                    vec[j] = float(train_mean[j])
            vecs.append(vec)
        prompts, _ = build_prompts(vecs, rows_sub, trait_names, cat_of, representation=rep)
        log(f"ablation {cat}: {len(prompts)} prompts ...")
        t0 = time.time()
        texts = generate_texts(model, tok, device, prompts)
        log(f"ablation {cat}: done in {time.time() - t0:.0f}s")
        masked_traits = [t for t in trait_names if cat_of[t] == cat]
        title = CATEGORY_TITLES[cat]
        sec = evaluate_condition(texts, rows_sub, tgt_full, tgt_secs, trait_names,
                                 _cal_thresholds())["per_section"]
        own = {
            "rougeL_f1_mean": sec[title]["rougeL_f1_mean"],
            "rougeL_f1_delta_vs_normal":
                sec[title]["rougeL_f1_mean"] - state["normal_per_section"][title]["rougeL_f1_mean"],
            "semantic_f1_mean": sec[title]["semantic_f1_mean"],
            "semantic_f1_delta_vs_normal":
                sec[title]["semantic_f1_mean"] -
                state["normal_per_section"][title]["semantic_f1_mean"],
        }
        # section-level text change vs NORMAL (own + cross sections)
        g_secs_all = [parse_sections(t)[0] for t in texts]
        n_secs_all = [parse_sections(t)[0] for t in normal_texts]
        changed = {}
        for other_title in SECTION_TITLES:
            a = [s.get(other_title, "") for s in g_secs_all]
            b = [s.get(other_title, "") for s in n_secs_all]
            changed[other_title] = change_rates(a, b)
        own.update(changed[title])
        cross = {ot: changed[ot] for ot in SECTION_TITLES if ot != title}
        cross_mean = mean([cross[ot]["normalized_changed_rate"] for ot in cross])
        diag = {
            "own_normalized_changed_rate": own["normalized_changed_rate"],
            "cross_mean_normalized_changed_rate": cross_mean,
            "own_section_top_change":
                own["normalized_changed_rate"] >= max(
                    cross[ot]["normalized_changed_rate"] for ot in cross) if cross else True,
            "weak_own_change": own["normalized_changed_rate"] < THRESH["own_section_changed_weak"],
            "entangled": cross_mean > own["normalized_changed_rate"],
        }
        state["per_category"][cat] = {
            "masked_traits": masked_traits,
            "corresponding_section": title,
            "own": own,
            "cross": cross,
            "diagnosis": diag,
        }
        out_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"ablation {cat}: own changed {own['normalized_changed_rate']:.3f} "
            f"cross mean {cross_mean:.3f}")
    # summary
    weak = [c for c in CATEGORY_ORDER
            if state["per_category"].get(c, {}).get("diagnosis", {}).get("weak_own_change")]
    entangled = [c for c in CATEGORY_ORDER
                 if state["per_category"].get(c, {}).get("diagnosis", {}).get("entangled")]
    state["diagnosis"] = {"weak_categories": weak, "entangled_categories": entangled}
    out_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Wrote {out_path}")


def phase_sweep(args, trait_names, cat_of, test_rows, train_mean):
    outdir = Path(args.outdir)
    rep = getattr(args, "representation", "continuous")
    suffix = "_bins" if rep == "bins" else ""
    out_path = outdir / f"text_generator_sensitivity_sweep{suffix}.json"
    n = min(SWEEP_N, len(test_rows)) if not args.limit else min(args.limit, SWEEP_N)
    rows_sub = test_rows[:n]
    state = {
        "n_probe": n,
        "row_selection": "first N canonical test rows (frozen file order)",
        "traits": SWEEP_TRAITS,
        "levels": SWEEP_LEVELS,
        "other_probs": "held at each row's original values",
        "classification_rules": {
            "no_response": "all six level outputs identical for every row",
            "clear_response": "concept_present@1.0 - @0.0 >= "
                              f"{THRESH['sweep_clear_concept_delta']} or "
                              f"normalized_changed(vs 0.0)@1.0 >= {THRESH['sweep_clear_norm_change']}",
            "non_monotonic_response": "outputs change but concept-present rate is not "
                                       "non-decreasing across levels and not clear",
            "weak_response": "some output change but below clear thresholds and monotone-ish",
        },
        "per_trait": {},
    }
    if out_path.exists() and not args.force:
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        if prev.get("n_probe") == n:
            state = prev
    todo = [t for t in SWEEP_TRAITS if t not in state["per_trait"]]
    if todo:
        assert all(t in trait_names for t in todo), "sweep trait missing from schema"
        model, tok, device = load_model(args.model)
    for trait in todo:
        cat = cat_of[trait]
        title = CATEGORY_TITLES[cat]
        level_texts = {}
        t0 = time.time()
        for lvl in SWEEP_LEVELS:
            vecs = []
            for r in rows_sub:
                vec = [float(r["trait_probs"][t]) for t in trait_names]
                vec[trait_names.index(trait)] = float(lvl)
                vecs.append(vec)
            prompts, _ = build_prompts(vecs, rows_sub, trait_names, cat_of, representation=rep)
            level_texts[lvl] = generate_texts(model, tok, device, prompts)
        log(f"sweep {trait}: {len(SWEEP_LEVELS)} levels x {n} rows in {time.time() - t0:.0f}s")
        base = level_texts[SWEEP_LEVELS[0]]
        base_secs = [parse_sections(t)[0].get(title, "") for t in base]
        prev_texts = None
        levels_data = {}
        for lvl in SWEEP_LEVELS:
            texts = level_texts[lvl]
            secs = [parse_sections(t)[0].get(title, "") for t in texts]
            present = [1.0 if any(re.search(p, s, re.I) for p in TRAIT_PATTERNS[trait][1])
                       else 0.0 for s in secs]
            rec = {
                "concept_present_rate": mean(present),
            }
            if lvl != SWEEP_LEVELS[0]:
                rec["changed_exact_vs_0"] = mean([1.0 if x != y else 0.0
                                                  for x, y in zip(texts, base)])
                rec["changed_norm_vs_0"] = mean([1.0 if norm_text(x) != norm_text(y) else 0.0
                                                 for x, y in zip(texts, base)])
                rec["section_changed_vs_0"] = mean([1.0 if norm_text(a) != norm_text(b) else 0.0
                                                    for a, b in zip(secs, base_secs)])
            else:
                rec.update({"changed_exact_vs_0": 0.0, "changed_norm_vs_0": 0.0,
                            "section_changed_vs_0": 0.0})
            if prev_texts is not None:
                rec["changed_exact_vs_prev"] = mean([1.0 if x != y else 0.0
                                                     for x, y in zip(texts, prev_texts)])
                rec["changed_norm_vs_prev"] = mean([1.0 if norm_text(x) != norm_text(y) else 0.0
                                                    for x, y in zip(texts, prev_texts)])
            levels_data[str(lvl)] = rec
            prev_texts = texts
        low, high = level_texts[0.0], level_texts[1.0]
        r01 = mean([rouge_l_one(a, b)[2] for a, b in zip(low, high)])
        present_seq = [levels_data[str(l)]["concept_present_rate"] for l in SWEEP_LEVELS]
        delta_present = present_seq[-1] - present_seq[0]
        non_mono = any(present_seq[i] < present_seq[i - 1] - 1e-9 for i in range(1, len(present_seq)))
        any_change = any(levels_data[str(l)].get("changed_norm_vs_prev", 0.0) > 0
                         for l in SWEEP_LEVELS[1:])
        norm_at_1 = levels_data["1.0"]["changed_norm_vs_0"]
        if not any_change:
            label = "no_response"
        elif delta_present >= THRESH["sweep_clear_concept_delta"] or \
                norm_at_1 >= THRESH["sweep_clear_norm_change"]:
            label = "clear_response"
        elif non_mono:
            label = "non_monotonic_response"
        else:
            label = "weak_response"
        state["per_trait"][trait] = {
            "category": cat, "section": title,
            "levels": levels_data,
            "rougeL_f1_0.0_vs_1.0": r01,
            "concept_present_delta_1.0_minus_0.0": delta_present,
            "concept_present_by_level": present_seq,
            "non_monotonic": non_mono,
            "response": label,
        }
        out_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "traits_tested": len(state["per_trait"]),
        "responsive": [t for t, d in state["per_trait"].items() if d["response"] == "clear_response"],
        "weak": [t for t, d in state["per_trait"].items() if d["response"] == "weak_response"],
        "no_response": [t for t, d in state["per_trait"].items() if d["response"] == "no_response"],
        "non_monotonic": [t for t, d in state["per_trait"].items()
                          if d["response"] == "non_monotonic_response"],
        "non_monotonic_flagged": [t for t, d in state["per_trait"].items() if d["non_monotonic"]],
    }
    state["summary"] = summary
    out_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Wrote {out_path}")


def _cal_thresholds():
    tmap, _strategy = load_threshold_map()
    return tmap or {}


def phase_report(args, trait_names, cat_of, test_rows, train_mean, n_train, splits):
    outdir = Path(args.outdir)
    rep = getattr(args, "representation", "continuous")
    suffix = "_bins" if rep == "bins" else ""
    if args.limit:
        test_rows = test_rows[:args.limit]
    tgt_full, tgt_secs = precompute_targets(test_rows)
    thr_cal = _cal_thresholds()
    gen_by_cond = {}
    report = {
        "audit": "frozen FLAN-T5-small conditioning audit",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_test": len(test_rows),
        "row_order": "frozen file order (identical across conditions)",
        "image_ids_sha256": hashlib.sha256(
            "\n".join(r["image_id"] for r in test_rows).encode()).hexdigest(),
        "frozen": {
            "model_dir": str(args.model),
            "model_files_sha256": {p.name: sha256_file(p) for p in sorted(Path(args.model).iterdir())
                                   if p.is_file()},
            "splits_sha256": sha256_file(DEFAULT_SPLITS),
            "schema_sha256": sha256_file(ROOT / "semantic_schema.json"),
            "prompt_instruction": INSTRUCTION,
            "prompt_format": f"format_input (text_gen_common), {rep}, category headers",
            "max_source_len": MAX_SOURCE_LEN,
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
            "num_beams": 1,
            "batch_size": BS,
            "seed": SEED,
            "representation": rep,
            "cuda_available": torch.cuda.is_available(),
            "train_meta": json.loads((Path(args.model) / "train_meta.json").read_text(encoding="utf-8")),
        },
        "train_mean_vector": {t: round(float(train_mean[i]), 6) for i, t in enumerate(trait_names)},
        "train_mean_source": f"train_images split only (n={n_train})",
        "interpretation_thresholds": THRESH,
        "conditions": {},
        "deltas_vs_normal": {},
        "pairwise_change_vs_normal": {},
    }
    # load + assert identical ids/order across condition files
    for cond in CONDITIONS:
        path = outdir / _cond_file(cond, rep)
        recs = load_jsonl(path)
        assert len(recs) == len(test_rows), f"{cond}: {len(recs)} rows != 702"
        ids = [r["image_id"] for r in recs]
        assert ids == [r["image_id"] for r in test_rows], f"{cond}: row order mismatch"
        # attach audit probs for input faithfulness of THIS condition
        for rec, row in zip(recs, test_rows):
            assert list(rec["probs"].keys()) == trait_names, f"{cond}: probs key order"
            row["_audit_probs"] = rec["probs"]
            for t in trait_names:
                v = float(rec["probs"][t])
                assert np.isfinite(v) and 0.0 <= v <= 1.0
        if cond == "SHUFFLED":
            for rec in recs:
                assert "source_image_id" in rec
                assert norm_name(rec["source_image_id"]) != norm_name(rec["image_id"]), \
                    "shuffle source == target"
        gen_by_cond[cond] = [r["generated_text"] for r in recs]
        report["conditions"][cond] = evaluate_condition(
            gen_by_cond[cond], test_rows, tgt_full, tgt_secs, trait_names, thr_cal)
        log(f"evaluated {cond}")

    # pairwise deltas + change rates vs NORMAL
    norm_m = report["conditions"]["NORMAL"]
    for cond in CONDITIONS[1:]:
        m = report["conditions"][cond]
        report["deltas_vs_normal"][cond] = {
            "rougeL_f1": norm_m["rougeL_f1_mean"] - m["rougeL_f1_mean"],
            "bleu1": norm_m["bleu1"] - m["bleu1"],
            "bleu4": norm_m["bleu4"] - m["bleu4"],
            "target_semantic_f1": norm_m["target_semantic"]["f1_mean"] - m["target_semantic"]["f1_mean"],
            "input_faithfulness_f1_0.5":
                norm_m["input_faithfulness"]["threshold_0.5"]["f1_mean"] -
                m["input_faithfulness"]["threshold_0.5"]["f1_mean"],
            "input_faithfulness_f1_calibrated":
                norm_m["input_faithfulness"]["calibrated"]["f1_mean"] -
                m["input_faithfulness"]["calibrated"]["f1_mean"],
        }
        report["pairwise_change_vs_normal"][cond] = change_rates(
            gen_by_cond["NORMAL"], gen_by_cond[cond])

    # sanity vs historical NORMAL run (generate_text_model_test.jsonl)
    hist_path = SRC / "generated_text_model_test.jsonl"
    if hist_path.exists():
        hist = {norm_name(r["image_id"]): r["generated_text"] for r in load_jsonl(hist_path)}
        normal_by_id = {norm_name(r["image_id"]): t
                        for r, t in zip(test_rows, gen_by_cond["NORMAL"])}
        same = sum(1 for k, t in normal_by_id.items() if hist.get(k) == t)
        report["historical_normal_exact_agreement"] = same / len(test_rows)
        log(f"historical NORMAL agreement {same}/{len(test_rows)}")

    # decision gate (needs ablation + sweep artifacts)
    abl_path = outdir / f"text_conditioning_category_ablation{suffix}.json"
    swp_path = outdir / f"text_generator_sensitivity_sweep{suffix}.json"
    ablation = json.loads(abl_path.read_text(encoding="utf-8")) if abl_path.exists() else None
    sweep = json.loads(swp_path.read_text(encoding="utf-8")) if swp_path.exists() else None
    report["signals"] = {}
    if ablation and sweep:
        d = report["deltas_vs_normal"]
        dR_shuf, dR_cm = d["SHUFFLED"]["rougeL_f1"], d["CONSTANT_MEAN"]["rougeL_f1"]
        dF_shuf, dF_cm = d["SHUFFLED"]["target_semantic_f1"], d["CONSTANT_MEAN"]["target_semantic_f1"]
        if (dR_shuf >= THRESH["delta_rouge_strong"] and dR_cm >= THRESH["delta_rouge_strong"]) or \
                (dF_shuf >= THRESH["delta_semf1_strong"] and dF_cm >= THRESH["delta_semf1_strong"]):
            s1 = "strong"
        elif dR_shuf >= THRESH["delta_rouge_moderate"] or dR_cm >= THRESH["delta_rouge_strong"] or \
                dF_shuf >= THRESH["delta_semf1_strong"]:
            s1 = "moderate"
        else:
            s1 = "weak"
        own_strong = sum(1 for c in CATEGORY_ORDER
                         if ablation["per_category"][c]["own"]["normalized_changed_rate"]
                         >= THRESH["own_section_changed_strong"])
        frac_abl = own_strong / len(CATEGORY_ORDER)
        s2 = "strong" if frac_abl >= 0.8 else "moderate" if frac_abl >= 0.4 else "weak"
        n_clear = len(sweep.get("summary", {}).get("responsive", []))
        n_total = max(1, len(sweep.get("per_trait", {})))
        frac_swp = n_clear / n_total
        s3 = "strong" if frac_swp >= 0.7 else "moderate" if frac_swp >= 0.4 else "weak"
        if s1 == "strong" and s2 in ("strong", "moderate") and s3 in ("strong", "moderate"):
            conclusion = "strong"
        elif s1 == "weak" and s2 == "weak" and s3 == "weak":
            conclusion = "weak"
        else:
            conclusion = "moderate"
        report["signals"] = {
            "conditioning_main": {
                "level": s1,
                "delta_rouge_shuffled": dR_shuf,
                "delta_rouge_constant_mean": dR_cm,
                "delta_semf1_shuffled": dF_shuf,
                "delta_semf1_constant_mean": dF_cm,
            },
            "category_mask": {"level": s2, "own_section_strong_fraction": frac_abl},
            "sensitivity_sweep": {"level": s3, "clear_fraction": frac_swp,
                                  "n_clear": n_clear, "n_total": n_total},
        }
        report["conclusion"] = conclusion
        report["path"] = "B" if conclusion == "weak" else "A"
        report["representation_retraining_triggered"] = conclusion == "weak"
    else:
        report["conclusion"] = "pending (ablation/sweep artifacts missing)"
        report["path"] = "pending"
        report["representation_retraining_triggered"] = False

    out_path = outdir / f"text_conditioning_audit{suffix}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Wrote {out_path}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["conditions", "ablation", "sweep", "report", "all"],
                    default="all")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--outdir", default=str(SRC))
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N rows only")
    ap.add_argument("--force", action="store_true", help="regenerate even if outputs exist")
    ap.add_argument("--representation", choices=["continuous","bins"], default="continuous")
    args = ap.parse_args()

    all_rows, test_rows, trait_names, cat_of, splits, train_set = load_frozen_rows()
    train_mean, n_train = train_mean_vector(all_rows, train_set, trait_names)
    log(f"frozen: test={len(test_rows)} train_mean_rows={n_train} dim=45 representation={args.representation}")

    if args.phase in ("conditions", "all"):
        phase_conditions(args, trait_names, cat_of, test_rows, train_mean)
    if args.phase in ("ablation", "all"):
        phase_ablation(args, trait_names, cat_of, test_rows, train_mean, splits, train_set)
    if args.phase in ("sweep", "all"):
        phase_sweep(args, trait_names, cat_of, test_rows, train_mean)
    if args.phase in ("report", "all"):
        report = phase_report(args, trait_names, cat_of, test_rows, train_mean, n_train, splits)
        if report.get("conclusion"):
            log(f"conclusion={report['conclusion']} path={report['path']} "
                f"retrain={report['representation_retraining_triggered']}")


if __name__ == "__main__":
    main()