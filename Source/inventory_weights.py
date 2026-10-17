# -*- coding: utf-8 -*-
"""Pretrained weights inventory and offline provisioning validator.

Audits presence, sizes, and paths of all models required for offline CPU execution:
1. Semantic preprocessing and schema artifacts (7 files in Source/models_semantic/)
2. Semantic classifier models (45 logreg_*.pkl in Source/models_semantic/)
3. Text generation checkpoint (FLAN-T5-small in Source/models_text/flan_t5_small/)
4. InsightFace detector & landmarks (buffalo_l: det_10g.onnx, 2d106det.onnx)
5. Facexlib parser weights (parsing_bisenet.pth)

Usage:
    python Source/inventory_weights.py [--check] [--json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Source"
SEMANTIC_DIR = SRC / "models_semantic"
TEXT_DIR = SRC / "models_text" / "flan_t5_small"

SEMANTIC_METADATA_FILES = [
    "feature_cols.npy",
    "trait_names.npy",
    "imputer.pkl",
    "scaler.pkl",
    "semantic_schema.json",
    "missing_indicator_cols.json",
    "trait_thresholds.json",
]

TEXT_CHECKPOINT_FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
]

INSIGHTFACE_FILES = [
    "det_10g.onnx",
    "2d106det.onnx",
]


def _file_info(p: Path) -> dict:
    if not p.exists():
        return {"exists": False, "bytes": 0}
    return {
        "exists": True,
        "bytes": p.stat().st_size,
    }


def find_insightface_dir() -> Path | None:
    env_home = os.environ.get("INSIGHTFACE_HOME")
    candidates = []
    if env_home:
        candidates.append(Path(env_home) / "models" / "buffalo_l")
        candidates.append(Path(env_home) / "buffalo_l")
    user_home = Path.home() / ".insightface" / "models" / "buffalo_l"
    candidates.append(user_home)
    app_dir = ROOT / ".insightface" / "models" / "buffalo_l"
    candidates.append(app_dir)
    for c in candidates:
        if c.exists() and (c / "det_10g.onnx").exists():
            return c
    return candidates[0] if candidates else None


def find_facexlib_dir() -> Path | None:
    try:
        import facexlib
        pkg_weights = Path(facexlib.__file__).parent / "weights"
        if pkg_weights.exists():
            return pkg_weights
    except Exception:
        pass
    app_weights = ROOT / "facexlib" / "weights"
    return app_weights


def audit_weights() -> dict:
    # 1. Semantic metadata
    sem_meta = {}
    sem_meta_missing = []
    for f in SEMANTIC_METADATA_FILES:
        p = SEMANTIC_DIR / f
        info = _file_info(p)
        sem_meta[f] = info
        if not info["exists"]:
            sem_meta_missing.append(f)

    # 2. 45 classifiers
    clfs = {}
    clf_missing = []
    trait_names_file = SEMANTIC_DIR / "trait_names.npy"
    if trait_names_file.exists():
        import numpy as np
        traits = np.load(trait_names_file, allow_pickle=True).tolist()
    else:
        traits = []
    for t in traits:
        fname = f"logreg_{t}.pkl"
        p = SEMANTIC_DIR / fname
        info = _file_info(p)
        clfs[fname] = info
        if not info["exists"]:
            clf_missing.append(fname)

    # 3. Text model
    text_info = {}
    text_missing = []
    for f in TEXT_CHECKPOINT_FILES:
        p = TEXT_DIR / f
        info = _file_info(p)
        text_info[f] = info
        if not info["exists"]:
            text_missing.append(f)

    # 3b. Controlled Text Model (Phase 4 fine-tuned)
    ctrl_dir = SRC / "models_text" / "flan_t5_small_controlled"
    ctrl_present = 0
    if ctrl_dir.exists():
        for f in TEXT_CHECKPOINT_FILES:
            if (ctrl_dir / f).exists():
                ctrl_present += 1

    # 4. Insightface
    if_dir = find_insightface_dir()
    if_info = {}
    if_missing = []
    for f in INSIGHTFACE_FILES:
        p = if_dir / f if if_dir else Path(f)
        info = _file_info(p)
        if_info[f] = info
        if not info["exists"]:
            if_missing.append(f)

    # 5. Facexlib
    fx_dir = find_facexlib_dir()
    bisenet_path = fx_dir / "parsing_bisenet.pth" if fx_dir else Path("parsing_bisenet.pth")
    fx_info = {"parsing_bisenet.pth": _file_info(bisenet_path)}
    fx_missing = [] if fx_info["parsing_bisenet.pth"]["exists"] else ["parsing_bisenet.pth"]

    all_ok = (
        len(sem_meta_missing) == 0
        and len(clf_missing) == 0
        and len(text_missing) == 0
        and len(if_missing) == 0
        and len(fx_missing) == 0
    )

    return {
        "status": "PASS" if all_ok else "INCOMPLETE",
        "all_weights_present": all_ok,
        "directories": {
            "semantic": str(SEMANTIC_DIR),
            "text": str(TEXT_DIR),
            "insightface": str(if_dir) if if_dir else None,
            "facexlib": str(fx_dir) if fx_dir else None,
        },
        "semantic_metadata": {
            "total_expected": len(SEMANTIC_METADATA_FILES),
            "present": len(SEMANTIC_METADATA_FILES) - len(sem_meta_missing),
            "missing": sem_meta_missing,
        },
        "semantic_classifiers": {
            "total_expected": len(traits) or 45,
            "present": (len(traits) or 45) - len(clf_missing),
            "missing": clf_missing,
        },
        "text_checkpoint": {
            "total_expected": len(TEXT_CHECKPOINT_FILES),
            "present": len(TEXT_CHECKPOINT_FILES) - len(text_missing),
            "missing": text_missing,
        },
        "text_controlled_checkpoint": {
            "total_expected": len(TEXT_CHECKPOINT_FILES),
            "present": ctrl_present,
            "directory": str(ctrl_dir) if ctrl_dir.exists() else None,
        },
        "insightface_buffalo_l": {
            "total_expected": len(INSIGHTFACE_FILES),
            "present": len(INSIGHTFACE_FILES) - len(if_missing),
            "missing": if_missing,
            "directory": str(if_dir) if if_dir else None,
        },
        "facexlib_parser": {
            "bisenet_present": fx_info["parsing_bisenet.pth"]["exists"],
            "path": str(bisenet_path),
            "missing": fx_missing,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Inventory model weights for offline execution")
    parser.add_argument("--json", action="store_true", help="Output JSON manifest")
    parser.add_argument("--check", action="store_true", help="Exit code 0 if all weights present, 1 otherwise")
    args = parser.parse_args()

    report = audit_weights()
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"Status: {report['status']}")
        print(f"  Semantic metadata: {report['semantic_metadata']['present']}/{report['semantic_metadata']['total_expected']}")
        print(f"  Semantic classifiers: {report['semantic_classifiers']['present']}/{report['semantic_classifiers']['total_expected']}")
        print(f"  Text model (FLAN-T5 baseline): {report['text_checkpoint']['present']}/{report['text_checkpoint']['total_expected']}")
        print(f"  Text model (FLAN-T5 controlled): {report['text_controlled_checkpoint']['present']}/{report['text_controlled_checkpoint']['total_expected']}")
        print(f"  InsightFace (SCRFD): {report['insightface_buffalo_l']['present']}/{report['insightface_buffalo_l']['total_expected']} (at {report['insightface_buffalo_l']['directory']})")
        print(f"  Facexlib (BiSeNet): {'OK' if report['facexlib_parser']['bisenet_present'] else 'MISSING'} (at {report['facexlib_parser']['path']})")

    if args.check:
        sys.exit(0 if report["all_weights_present"] else 1)


if __name__ == "__main__":
    main()
