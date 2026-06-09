"""Offline oversampling: generate augmented copies of minority-class images to balance the dataset.

Usage:
    python scripts/offline_oversample.py --target 1000
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TRAIN_CSV, TRAIN_IMG_DIR, ROOT_DIR, IMAGE_SIZE
from src.dataset import ben_graham_preprocess
from src.transforms import get_offline_oversample_transform


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline oversampling for DR dataset")
    parser.add_argument("--target", type=int, default=1000, help="Target count per class")
    parser.add_argument("--output", type=str, default=str(ROOT_DIR / "data" / "train_oversampled"),
                        help="Output directory for oversampled images")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training labels (only codes that are in the train SPLIT — pre-val-split)
    # We use the full train CSV; the experiment code handles train/val split.
    labels: dict[str, int] = {}
    with open(TRAIN_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["id_code"]] = int(row["diagnosis"])

    class_codes: dict[int, list[str]] = {c: [] for c in range(5)}
    for code, label in labels.items():
        img_path = TRAIN_IMG_DIR / f"{code}.png"
        if img_path.exists():
            class_codes[label].append(code)

    counts = {c: len(codes) for c, codes in class_codes.items()}
    print(f"Original class counts: {counts}")
    print(f"Target per class: {args.target}")

    transform = get_offline_oversample_transform()
    total_generated = 0

    for cls in range(5):
        current = counts[cls]
        needed = max(0, args.target - current)
        if needed == 0:
            print(f"  Class {cls}: {current} images, no augmentation needed")
            continue

        codes = class_codes[cls]
        print(f"  Class {cls}: {current} images, generating {needed} augmented copies...")

        for i in tqdm(range(needed), desc=f"Class {cls}"):
            # Random original image
            idx = rng.randint(0, len(codes))
            code = codes[idx]
            img_path = TRAIN_IMG_DIR / f"{code}.png"

            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = ben_graham_preprocess(image, IMAGE_SIZE)

            if transform is not None:
                augmented = transform(image=image)
                image = augmented["image"]

            # Filename encodes class: {orig_code}_aug{N}_{class}
            out_name = f"{code}_aug{i}_{cls}.png"
            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            total_generated += 1

    print(f"\nDone. Generated {total_generated} augmented images in {output_dir}")


if __name__ == "__main__":
    main()
