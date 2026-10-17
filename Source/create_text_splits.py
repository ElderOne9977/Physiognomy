"""
Create group-aware train/validation/test split for the text generator.

Groups come from semantic_training_dataset.csv `group_id` (stem-based) so no
group crosses train/val/test. Deterministic given --seed.

Usage:
  python Source/create_text_splits.py --dataset semantic_training_dataset.csv \
      --output Source/text_generation_splits.json --seed 42
"""
import argparse
import csv
import json
import os
import random
import re
from pathlib import Path


def norm_name(value):
    s = os.path.basename(str(value).strip())
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\)\s+\(", ")(", s)
    return s.lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="semantic_training_dataset.csv")
    ap.add_argument("--output", default="Source/text_generation_splits.json")
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    groups_map = {}
    with open(args.dataset, encoding="utf-8-sig") as f:
        r = csv.reader(f)
        hdr = next(r)
        img_idx = hdr.index("image_id")
        grp_idx = hdr.index("group_id") if "group_id" in hdr else None
        for row in r:
            img_id = row[img_idx]
            norm = norm_name(img_id)
            g = row[grp_idx] if grp_idx is not None else norm
            groups_map.setdefault(g, []).append(norm)

    group_list = sorted(groups_map.keys())
    random.shuffle(group_list)
    total = len(group_list)
    train_end = int(total * args.train_ratio)
    val_end = train_end + int(total * args.val_ratio)
    train_groups = group_list[:train_end]
    val_groups = group_list[train_end:val_end]
    test_groups = group_list[val_end:]

    train_imgs = sum((groups_map[g] for g in train_groups), [])
    val_imgs = sum((groups_map[g] for g in val_groups), [])
    test_imgs = sum((groups_map[g] for g in test_groups), [])

    split = {
        "version": 1,
        "seed": args.seed,
        "group_key": "group_id (stem)",
        "split_ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": round(1.0 - args.train_ratio - args.val_ratio, 4),
        },
        "group_count": total,
        "train_group_count": len(train_groups),
        "val_group_count": len(val_groups),
        "test_group_count": len(test_groups),
        "train_image_count": len(train_imgs),
        "val_image_count": len(val_imgs),
        "test_image_count": len(test_imgs),
        "train_images": sorted(train_imgs),
        "val_images": sorted(val_imgs),
        "test_images": sorted(test_imgs),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(f"Wrote split: {len(train_imgs)} train / {len(val_imgs)} val / "
          f"{len(test_imgs)} test ({total} groups)")


if __name__ == "__main__":
    main()