"""IDRiD Disease Grading preprocessing: Ben Graham normalize selected grades → 512×512 PNG.

Reads both IDRiD train and test partitions, filters by grade, preprocesses,
and writes data/idrid_processed/ + data/idrid_labels.csv.

Usage:
    python scripts/preprocess_idrid.py                     # default: grades 3,4
    python scripts/preprocess_idrid.py --grades 3 4        # explicit
    python scripts/preprocess_idrid.py --grades 0 1 2 3 4  # all grades
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT_DIR, IMAGE_SIZE
from src.dataset import ben_graham_preprocess


# ── IDRiD raw paths ─────────────────────────────────────────────────────────
IDRID_DIR = ROOT_DIR / "B_Disease_Grading"
IDRID_TRAIN_IMG_DIR = IDRID_DIR / "1. Original Images" / "a. Training Set"
IDRID_TEST_IMG_DIR  = IDRID_DIR / "1. Original Images" / "b. Testing Set"
IDRID_TRAIN_CSV     = IDRID_DIR / "2. Groundtruths" / "a. IDRiD_Disease Grading_Training Labels.csv"
IDRID_TEST_CSV      = IDRID_DIR / "2. Groundtruths" / "b. IDRiD_Disease Grading_Testing Labels.csv"

# ── Output paths ────────────────────────────────────────────────────────────
OUTPUT_IMG_DIR = ROOT_DIR / "data" / "idrid_processed"
OUTPUT_CSV     = ROOT_DIR / "data" / "idrid_labels.csv"


def load_idrid_csv(csv_path: Path, partition: str) -> list[dict]:
    """Load an IDRiD label CSV and return list of {id_code, diagnosis, img_path}."""
    records = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header

        for row in reader:
            if not row or not row[0].strip():
                continue
            image_name = row[0].strip()
            grade = int(row[1].strip())

            # Determine source image directory
            if partition == "train":
                img_dir = IDRID_TRAIN_IMG_DIR
                id_code = f"idrid_train_{image_name}"
            else:
                img_dir = IDRID_TEST_IMG_DIR
                id_code = f"idrid_test_{image_name}"

            img_path = img_dir / f"{image_name}.jpg"

            records.append({
                "id_code": id_code,
                "diagnosis": grade,
                "img_path": img_path,
                "partition": partition,
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
    parser = argparse.ArgumentParser(description="IDRiD Disease Grading preprocessing")
    parser.add_argument("--grades", type=int, nargs="+", default=[3, 4],
                        help="DR grades to include (default: 3 4)")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_IMG_DIR),
                        help="Output directory for processed images")
    parser.add_argument("--output-csv", type=str, default=str(OUTPUT_CSV),
                        help="Output CSV path")
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1),
                        help="Number of parallel workers")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load both train and test partitions ────────────────────────────────
    print("Loading IDRiD label CSVs...")
    train_records = load_idrid_csv(IDRID_TRAIN_CSV, "train")
    test_records  = load_idrid_csv(IDRID_TEST_CSV, "test")
    all_records = train_records + test_records

    print(f"  Train partition: {len(train_records)} records")
    print(f"  Test partition:  {len(test_records)} records")
    print(f"  Total:           {len(all_records)} records")

    # ── Filter by selected grades ──────────────────────────────────────────
    selected_grades = set(args.grades)
    filtered = [r for r in all_records if r["diagnosis"] in selected_grades]
    print(f"\nFiltering to grades {sorted(selected_grades)}:")

    from collections import Counter
    grade_counts = Counter(r["diagnosis"] for r in filtered)
    for g in sorted(grade_counts):
        print(f"  Grade {g}: {grade_counts[g]} images")
    print(f"  Total to preprocess: {len(filtered)}")

    # ── Check file existence ────────────────────────────────────────────────
    missing = [r for r in filtered if not Path(r["img_path"]).exists()]
    if missing:
        print(f"\n⚠ WARNING: {len(missing)} images not found on disk:")
        for m in missing[:10]:
            print(f"    {m['img_path']}")
        filtered = [r for r in filtered if Path(r["img_path"]).exists()]

    # ── Preprocess images ──────────────────────────────────────────────────
    print(f"\nPreprocessing {len(filtered)} images → {output_dir}")
    print(f"  Image size: {IMAGE_SIZE}×{IMAGE_SIZE}")
    print(f"  Workers: {args.workers}")

    task_args = [(r, output_dir, IMAGE_SIZE) for r in filtered]

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

    # ── Write output CSV ───────────────────────────────────────────────────
    output_csv = Path(args.output_csv)
    results.sort(key=lambda r: r["id_code"])

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id_code", "diagnosis"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✓ Done. Processed {len(results)} images → {output_dir}")
    print(f"✓ Labels written to {output_csv}")

    # ── Summary ────────────────────────────────────────────────────────────
    result_counts = Counter(r["diagnosis"] for r in results)
    for g in sorted(result_counts):
        print(f"  Grade {g}: {result_counts[g]} images")


if __name__ == "__main__":
    main()
