"""Smoke-test the non-GPU cells of notebooks/eda_deep.ipynb on a small subsample.

Runs just enough of Section A/B/C to catch broken imports, API mismatches, and
Windows-path bugs before the user fires up the full notebook.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CLASS_NAMES, IMAGE_SIZE, TRAIN_CSV, TEST_CSV, TRAIN_IMG_DIR, TEST_IMG_DIR
from src.dataset import ben_graham_preprocess, load_labels
from src.analysis import fundus_cv, quality_metrics, eda_utils


def _resolve(p: Path, fb: Path) -> Path:
    return p if p.exists() else fb


TRAIN_CSV = _resolve(TRAIN_CSV, ROOT / "data" / "train_label.csv")
TEST_CSV = _resolve(TEST_CSV, ROOT / "data" / "test_label.csv")
TRAIN_IMG_DIR = _resolve(TRAIN_IMG_DIR, ROOT / "data" / "train_split")
TEST_IMG_DIR = _resolve(TEST_IMG_DIR, ROOT / "data" / "test_split")

rng = np.random.default_rng(42)
train_labels = load_labels(TRAIN_CSV)
test_labels = load_labels(TEST_CSV)
print(f"train={len(train_labels)} test={len(test_labels)}")


def load_rgb(code, img_dir):
    p = img_dir / f"{code}.png"
    if not p.exists():
        return None
    img = cv2.imread(str(p))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None


# --- A2 core: pHash on a handful of images ---
sample = rng.choice(list(train_labels.keys()), size=30, replace=False).tolist()
hashes = {}
for c in sample:
    img = load_rgb(c, TRAIN_IMG_DIR)
    if img is not None:
        hashes[c] = eda_utils.phash(img, hash_size=16)
dups = eda_utils.find_near_duplicates(hashes, threshold=10)
print(f"[A2] hashed {len(hashes)} -> {len(dups)} pairs within d<=10")

# --- A3 core: image sizes ---
rows = []
for c in sample:
    p = TRAIN_IMG_DIR / f"{c}.png"
    img = cv2.imread(str(p))
    if img is None:
        continue
    h, w = img.shape[:2]
    rows.append({"id": c, "label": train_labels[c], "height": h, "width": w, "aspect": w / h})
sizes_df = pd.DataFrame(rows)
print(f"[A3] sizes collected: {len(sizes_df)}")
print(sizes_df.head(3).to_string())

# --- A4 core: radial profile ---
profs = []
for c in sample[:10]:
    img = load_rgb(c, TRAIN_IMG_DIR)
    if img is not None:
        profs.append(quality_metrics.radial_intensity_profile(img, n_bins=20))
profs = np.stack(profs)
print(f"[A4] radial profiles: {profs.shape}")

# --- A5 core: quality metrics ---
for c in sample[:3]:
    img = load_rgb(c, TRAIN_IMG_DIR)
    if img is not None:
        print(f"[A5] {c}: {quality_metrics.compute_all(img)}")

# --- B1 core: preprocessing variants apply cleanly ---
img = load_rgb(sample[0], TRAIN_IMG_DIR)
assert img is not None
import importlib, types

# Replicate preproc functions inline
def preproc_raw(img, size=IMAGE_SIZE):
    h, w = img.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    img = img[y0 : y0 + s, x0 : x0 + s]
    return cv2.resize(img, (size, size))


def preproc_bg_nosub(img, size=IMAGE_SIZE):
    mask = fundus_cv.retinal_fov_mask(img)
    ys, xs = np.where(mask)
    if ys.size:
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        img = img[y0 : y1 + 1, x0 : x1 + 1]
    return cv2.resize(img, (size, size))


for name, fn in [("raw", preproc_raw), ("bg_nosub", preproc_bg_nosub)]:
    out = fn(img)
    print(f"[B1] {name} -> {out.shape} dtype={out.dtype}")

# --- B2 core: augmentation survival ---
import albumentations as A

img_bg = ben_graham_preprocess(img, IMAGE_SIZE)
base = fundus_cv.ma_candidates(img_bg)
base_cc = cv2.connectedComponents(base)[0] - 1 if base.sum() else 0
for aname, aug in [
    ("hflip", A.HorizontalFlip(p=1.0)),
    ("clahe", A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0)),
]:
    out = aug(image=img_bg)["image"]
    m = fundus_cv.ma_candidates(out)
    cc = cv2.connectedComponents(m)[0] - 1 if m.sum() else 0
    print(f"[B2] {aname}: base_ma={base_cc} aug_ma={cc}")

# --- B3 core: class mean + MSE sanity ---
means = {}
for k in (0, 2, 4):
    codes_k = [c for c in train_labels if train_labels[c] == k][:10]
    imgs = []
    for c in codes_k:
        img = load_rgb(c, TRAIN_IMG_DIR)
        if img is not None:
            imgs.append(cv2.resize(img, (128, 128)).astype(np.float32))
    means[k] = np.stack(imgs).mean(axis=0) if imgs else None
print(f"[B3] mean images computed for classes: {list(means.keys())}")
mse = np.mean((means[0] - means[4]) ** 2) if means[0] is not None and means[4] is not None else -1
print(f"[B3] MSE(class0, class4) = {mse:.1f}")

print("ALL SMOKE CHECKS PASSED")
