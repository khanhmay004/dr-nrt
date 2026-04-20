"""Messidor-2 preprocessing: Ben Graham normalize → 512×512 PNG.

Reads messidor-2/IMAGES/ and messidor-2/messidor_data.csv, preprocesses all
gradable images, writes data/messidor2_processed/ + data/messidor2_labels.csv.

Usage:
    python scripts/preprocess_messidor2.py
    python scripts/preprocess_messidor2.py --workers 8
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT_DIR, IMAGE_SIZE
from src.dataset import ben_graham_preprocess


# ── Messidor-2 raw paths ─────────────────────────────────────────────────────
MESSIDOR2_DIR = ROOT_DIR / "messidor-2"
MESSIDOR2_IMG_DIR = MESSIDOR2_DIR / "IMAGES"
MESSIDOR2_CSV = MESSIDOR2_DIR / "messidor_data.csv"

# ── Output paths ─────────────────────────────────────────────────────────────
OUTPUT_IMG_DIR = ROOT_DIR / "data" / "messidor2_processed"
OUTPUT_CSV = ROOT_DIR / "data" / "messidor2_labels.csv"


def load_messidor2_csv(csv_path: Path) -> list[dict]:
    """Load Messidor-2 label CSV and return list of gradable records.

    Columns: image_id, adjudicated_dr_grade, adjudicated_dme, adjudicated_gradable
    - adjudicated_dr_grade: 0-4 (5-class ICDR, same as APTOS)
    - adjudicated_gradable: 1 = gradable, 0 = ungradable (skip these)
    """
    records = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row["image_id"].strip()
            gradable = row["adjudicated_gradable"].strip()

            # Skip ungradable images
            if gradable != "1":
                continue

            grade_str = row["adjudicated_dr_grade"].strip()
            if not grade_str:
                continue

            grade = int(grade_str)
            img_path = MESSIDOR2_IMG_DIR / image_id

            # id_code without .png extension for consistency with APTOS
            id_code = image_id.replace(".png", "").replace(".jpg", "")

            records.append({
                "id_code": id_code,
                "diagnosis": grade,
                "img_path": img_path,
                "original_filename": image_id,
            })

    return records


def process_single(args_tuple):
    """Process a single image: read, Ben Graham preprocess, save as PNG."""
    record, output_dir, image_size = args_tuple
    img_path = record["img_path"]
    id_code = record["id_code"]

    if not Path(img_path).exists():
        return None

    image = cv2.imread(str(img_path))
    if image is None:
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = ben_graham_preprocess(image, image_size)

    out_path = output_dir / f"{id_code}.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    return {"id_code": id_code, "diagnosis": record["diagnosis"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Messidor-2 preprocessing")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_IMG_DIR),
                        help="Output directory for processed images")
    parser.add_argument("--output-csv", type=str, default=str(OUTPUT_CSV),
                        help="Output CSV path")
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1),
                        help="Number of parallel workers")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load labels ──────────────────────────────────────────────────────────
    print("Loading Messidor-2 labels...")
    records = load_messidor2_csv(MESSIDOR2_CSV)
    print(f"  Total gradable images: {len(records)}")

    # ── Grade distribution ───────────────────────────────────────────────────
    from collections import Counter
    grade_counts = Counter(r["diagnosis"] for r in records)
    print("\nGrade distribution:")
    for g in sorted(grade_counts):
        print(f"  Grade {g}: {grade_counts[g]} images")

    # ── Check file existence ─────────────────────────────────────────────────
    missing = [r for r in records if not Path(r["img_path"]).exists()]
    if missing:
        print(f"\n⚠ WARNING: {len(missing)} images not found on disk:")
        for m in missing[:10]:
            print(f"    {m['img_path']}")
        records = [r for r in records if Path(r["img_path"]).exists()]

    # ── Preprocess images ────────────────────────────────────────────────────
    print(f"\nPreprocessing {len(records)} images → {output_dir}")
    print(f"  Image size: {IMAGE_SIZE}×{IMAGE_SIZE}")
    print(f"  Workers: {args.workers}")

    task_args = [(r, output_dir, IMAGE_SIZE) for r in records]

    results = []
    if args.workers <= 1:
        for ta in tqdm(task_args, desc="Processing"):
            result = process_single(ta)
            if result:
                results.append(result)
    else:
        with multiprocessing.Pool(args.workers) as pool:
            for result in tqdm(pool.imap_unordered(process_single, task_args),
                               total=len(task_args), desc="Processing"):
                if result:
                    results.append(result)

    # ── Write output CSV ─────────────────────────────────────────────────────
    output_csv = Path(args.output_csv)
    results.sort(key=lambda r: r["id_code"])

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id_code", "diagnosis"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✓ Done. Processed {len(results)} images → {output_dir}")
    print(f"✓ Labels written to {output_csv}")

    # ── Summary ──────────────────────────────────────────────────────────────
    result_counts = Counter(r["diagnosis"] for r in results)
    print("\nFinal grade distribution:")
    for g in sorted(result_counts):
        print(f"  Grade {g}: {result_counts[g]} images")

    # Binary referable summary
    non_referable = result_counts.get(0, 0) + result_counts.get(1, 0)
    referable = sum(result_counts.get(g, 0) for g in [2, 3, 4])
    print(f"\nBinary (referable = grade ≥ 2):")
    print(f"  Non-referable (0-1): {non_referable}")
    print(f"  Referable (2-4):     {referable}")


if __name__ == "__main__":
    main()
