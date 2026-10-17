# -*- coding: utf-8 -*-
"""Batch testing script for user test images.

Connects to running Gradio server (http://127.0.0.1:7860) to evaluate
face detection, QC compliance, semantic trait distribution, and
controlled text generation on user-supplied test sets.

Supports:
- --sample N: evaluate random subset of N images (default: None = all)
- --seed S: random seed for sampling (default: 42)
- --resume: skip images already processed in output JSONL
- --jsonl: output JSONL path
- --summary: output summary JSON path
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMG_DIR = ROOT / "test_img"
DEFAULT_JSONL = ROOT / "outputs" / "test_img_batch_results.jsonl"
DEFAULT_SUMMARY = ROOT / "outputs" / "test_img_batch_summary.json"


def extract_path(val):
    if isinstance(val, dict):
        return val.get("value") or val.get("path")
    return val


def parse_args():
    ap = argparse.ArgumentParser(description="Batch test images against Gradio server")
    ap.add_argument("--dir", type=str, default=str(DEFAULT_IMG_DIR), help="Image directory")
    ap.add_argument("--sample", type=int, default=None, help="Number of images to sample (default: all)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    ap.add_argument("--generator", type=str, default="Controlled FLAN · Có kiểm tra (experimental)", help="Gradio generator label")
    ap.add_argument("--jsonl", type=str, default=str(DEFAULT_JSONL), help="Output JSONL path")
    ap.add_argument("--summary", type=str, default=str(DEFAULT_SUMMARY), help="Output summary JSON path")
    ap.add_argument("--resume", action="store_true", help="Resume from existing JSONL")
    return ap.parse_args()


def main():
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    img_dir = Path(args.dir)
    jsonl_path = Path(args.jsonl)
    summary_path = Path(args.summary)

    if not img_dir.exists():
        print(f"Error: Directory not found: {img_dir}", file=sys.stderr)
        sys.exit(1)

    all_images = sorted([
        p for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
    ])
    total_found = len(all_images)
    print(f"Found {total_found} supported images in {img_dir}")

    # Check already processed if resume
    processed_names = set()
    if args.resume and jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        processed_names.add(record.get("image_name"))
                    except Exception:
                        pass
        print(f"Resuming: found {len(processed_names)} already processed images in {jsonl_path.name}")

    # Sampling
    if args.sample and args.sample < len(all_images):
        rng = random.Random(args.seed)
        selected_images = rng.sample(all_images, args.sample)
        selected_images.sort(key=lambda p: p.name)
        print(f"Sampled {len(selected_images)} images (seed={args.seed})")
    else:
        selected_images = all_images

    # Filter out already processed
    images_to_run = [p for p in selected_images if p.name not in processed_names]
    print(f"Queued {len(images_to_run)} images to process.")

    if not images_to_run:
        print("All target images are already processed.")
        compute_and_save_summary(jsonl_path, summary_path, total_found)
        return

    from gradio_client import Client, handle_file

    print("Connecting to Gradio server on http://127.0.0.1:7860...")
    try:
        client = Client("http://127.0.0.1:7860")
    except Exception as e:
        print(f"Failed to connect to Gradio server: {e}", file=sys.stderr)
        print("Please ensure 'python Source/demo_gradio.py' is running.", file=sys.stderr)
        sys.exit(1)

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if args.resume else "w"
    f_out = open(jsonl_path, mode, encoding="utf-8")

    t_start = time.time()
    n_total = len(images_to_run)

    try:
        for idx, img_path in enumerate(images_to_run, 1):
            t0 = time.time()
            img_name = img_path.name
            print(f"[{idx}/{n_total}] Processing {img_name}... ", end="", flush=True)

            entry = {
                "image_name": img_name,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }

            try:
                res = client.predict(
                    img=handle_file(str(img_path)),
                    gen=args.generator,
                    api_name="/_analyze"
                )
                elapsed = time.time() - t0
                report_html, status_html, raw_csv, raw_json, tech_details, gen_method, _ = res

                csv_p = extract_path(raw_csv)
                json_p = extract_path(raw_json)

                entry["elapsed_s"] = round(elapsed, 3)

                if json_p and os.path.exists(json_p):
                    with open(json_p, "r", encoding="utf-8") as jf:
                        data = json.load(jf)
                    entry["success"] = True
                    entry["face"] = data.get("face", {})
                    entry["predictions"] = data.get("predictions", {})
                    entry["text"] = data.get("text", "")
                    entry["validation"] = data.get("validation", {})
                    entry["timing"] = data.get("timing", {})
                    gen_data = data.get("generator") or {}
                    entry["provenance"] = gen_data.get("provenance")
                    entry["neural_used"] = gen_data.get("neural_used", False)
                    entry["fallback_used"] = gen_data.get("fallback_used", False)
                    
                    # Trait stats
                    probs = data.get("semantic", {}).get("probabilities", {})
                    entry["top_traits"] = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
                    
                    det_score = entry["face"].get("detection_score")
                    det_str = f"score={det_score:.3f}" if det_score else "score=N/A"
                    qc_ok = entry["face"].get("qc_pass", False)
                    qc_str = "QC:PASS" if qc_ok else f"QC:FAIL({entry['face'].get('qc_reasons')})"
                    neural_str = "Neural" if entry["neural_used"] else ("Fallback" if entry["fallback_used"] else "Deterministic")
                    print(f"OK ({elapsed:.2f}s, {det_str}, {qc_str}, {neural_str})")
                else:
                    entry["success"] = False
                    entry["error"] = "no_result_json"
                    entry["status_html"] = status_html
                    entry["tech_details"] = tech_details
                    print(f"FAILED (No result JSON in {elapsed:.2f}s)")

            except Exception as exc:
                elapsed = time.time() - t0
                entry["success"] = False
                entry["error"] = str(exc)
                entry["elapsed_s"] = round(elapsed, 3)
                print(f"ERROR ({elapsed:.2f}s): {exc}")

            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f_out.flush()

    finally:
        f_out.close()

    total_time = time.time() - t_start
    print(f"\nBatch processing finished in {total_time:.2f}s (avg {total_time/max(n_total, 1):.2f}s/image).")
    compute_and_save_summary(jsonl_path, summary_path, total_found)


def compute_and_save_summary(jsonl_path: Path, summary_path: Path, total_found_in_dir: int):
    if not jsonl_path.exists():
        return

    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass

    n = len(records)
    if n == 0:
        return

    successes = [r for r in records if r.get("success")]
    n_success = len(successes)
    
    # Face detection stats
    faces_detected = [r for r in successes if (r.get("face") or {}).get("face_detected")]
    qc_passed = [r for r in successes if (r.get("face") or {}).get("qc_pass")]
    
    # Text validation stats
    valid_5_sections = [r for r in successes if (r.get("validation") or {}).get("valid")]
    neural_gen_used = [r for r in successes if r.get("neural_used")]

    # Latency stats
    latencies = [r.get("elapsed_s", 0) for r in records if "elapsed_s" in r]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    # Top traits frequency
    trait_counts = {}
    for r in successes:
        for t_name, _ in r.get("top_traits", []):
            trait_counts[t_name] = trait_counts.get(t_name, 0) + 1

    top_dataset_traits = sorted(trait_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_images_in_dir": total_found_in_dir,
        "total_evaluated": n,
        "successful_analyses": n_success,
        "success_rate": round(n_success / n, 4),
        "face_detection_count": len(faces_detected),
        "face_detection_rate": round(len(faces_detected) / max(n_success, 1), 4),
        "qc_pass_count": len(qc_passed),
        "qc_pass_rate": round(len(qc_passed) / max(n_success, 1), 4),
        "valid_5_sections_count": len(valid_5_sections),
        "valid_5_sections_rate": round(len(valid_5_sections) / max(n_success, 1), 4),
        "neural_generation_used_count": len(neural_gen_used),
        "neural_generation_rate": round(len(neural_gen_used) / max(n_success, 1), 4),
        "average_latency_seconds": round(avg_latency, 2),
        "top_frequent_traits": top_dataset_traits,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "="*60)
    print("BATCH EVALUATION SUMMARY REPORT")
    print("="*60)
    print(f"Total Evaluated:          {n} / {total_found_in_dir} images")
    print(f"Successful Analyses:      {n_success} ({summary['success_rate']*100:.1f}%)")
    print(f"Face Detection Rate:      {len(faces_detected)}/{n_success} ({summary['face_detection_rate']*100:.1f}%)")
    print(f"QC Pass Rate:             {len(qc_passed)}/{n_success} ({summary['qc_pass_rate']*100:.1f}%)")
    print(f"5-Section Valid Text:     {len(valid_5_sections)}/{n_success} ({summary['valid_5_sections_rate']*100:.1f}%)")
    print(f"Neural Generation Used:   {len(neural_gen_used)}/{n_success} ({summary['neural_generation_rate']*100:.1f}%)")
    print(f"Average E2E Latency:      {avg_latency:.2f}s per image")
    print(f"Saved JSONL:              {jsonl_path}")
    print(f"Saved Summary:            {summary_path}")
    print("="*60)


if __name__ == "__main__":
    main()
