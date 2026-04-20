"""Rebuild per-pair + failure galleries with id_code in each thumbnail title.

Two variants per gallery:
  gallery_tagged_<name>.png     — raw RGB (what an ophthalmologist sees)
  gallery_tagged_<name>_bg.png  — Ben-Graham preprocessed (what the model sees)

Run: `.venv/Scripts/python scripts/retag_galleries.py`
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CLASS_NAMES, IMAGE_SIZE
from src.dataset import ben_graham_preprocess

TEST_DIR = ROOT / "data" / "test_split"
TRAIN_DIR = ROOT / "data" / "train_split"


def load_rgb(code: str, ben_graham: bool = False):
    for d in (TEST_DIR, TRAIN_DIR):
        p = d / f"{code}.png"
        if p.exists():
            img = cv2.imread(str(p))
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if ben_graham:
                rgb = ben_graham_preprocess(rgb, IMAGE_SIZE)
            return rgb
    return None


def render(df: pd.DataFrame, title: str, out: Path, n_cols: int = 6, ben_graham: bool = False):
    df = df.reset_index(drop=True)
    n = len(df)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3.3))
    axes = axes.flatten() if n > 1 else [axes]
    for i, row in df.iterrows():
        ax = axes[i]
        img = load_rgb(row["id_code"], ben_graham=ben_graham)
        if img is not None:
            ax.imshow(img)
        tl = int(row["true_label"])
        pl = int(row["pred_label"] if "pred_label" in row else row["rounded_prediction"])
        m = float(row["margin"])
        ax.set_title(f"{row['id_code']}\n{tl}->{pl}  m={m:.2f}", fontsize=8)
        ax.axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  ({n} samples)")


# --- per-pair: raw + Ben-Graham variants ---
pt = pd.read_csv(ROOT / "results" / "confusion_cache" / "per_pair_tags.csv")
for pair, grp in pt.groupby("pair"):
    safe = pair.replace("<->", "_vs_").replace(" ", "_")
    base = ROOT / "results" / "confusion_cache" / f"gallery_tagged_{safe}"
    g = grp.sort_values("margin", ascending=False)
    render(g, f"Pair {pair} (raw)", base.with_name(base.name + ".png"))
    render(g, f"Pair {pair} (Ben-Graham)", base.with_name(base.name + "_bg.png"), ben_graham=True)

# --- failures (top 50 ensemble errors) ---
ft = pd.read_csv(ROOT / "results" / "result_cache" / "failure_tags.csv")
base = ROOT / "results" / "result_cache" / "gallery_tagged_failures"
g = ft.sort_values("margin", ascending=False)
render(g, "Top-50 ensemble failures (raw)", base.with_name(base.name + ".png"))
render(g, "Top-50 ensemble failures (Ben-Graham)", base.with_name(base.name + "_bg.png"), ben_graham=True)

print("done")
