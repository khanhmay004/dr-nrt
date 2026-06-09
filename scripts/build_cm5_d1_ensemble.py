"""Build 5-class row-normalized confusion matrices for:
  * D1 single model        -> pic/cm5_d1.png
  * {D1, H1, H5} ensemble  -> pic/cm5_ensemble.png

D1 uses the threshold-optimized (cumulative_opt) predictions so the
F1 line up with the appendix table (Mild/Severe/PDR).
The ensemble uses the argmax predictions from ensemble_300_701_705.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "pic"

GRADES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

SOURCES = [
    {
        "tag": "D1 (exp300, threshold-opt)",
        "short": "D1",
        "preds": REPO_ROOT
        / "results"
        / "exp300_d1_dropout_cosine"
        / "threshold_opt"
        / "exp300_d1_dropout_cosine_thresh_opt_preds.csv",
        "stem": "cm5_d1",
    },
    {
        "tag": "Primary ensemble {D1, H1, H5}",
        "short": "Primary ensemble {D1, H1, H5}",
        "preds": REPO_ROOT
        / "results"
        / "ensemble_300_701_705"
        / "ensemble_300_701_705_argmax_preds.csv",
        "stem": "cm5_ensemble",
    },
]


def plot_cm(cm: np.ndarray, short: str, out_stem: str) -> None:
    row_sums = cm.sum(axis=1, keepdims=True)
    row_pct = np.divide(cm, np.where(row_sums == 0, 1, row_sums)) * 100.0

    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    im = ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=100)

    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(GRADES, fontsize=11, rotation=30, ha="right")
    ax.set_yticklabels(GRADES, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12, labelpad=10)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"{short}\n5-grade CM (row %)", fontsize=12)

    for i in range(5):
        for j in range(5):
            text_color = "white" if row_pct[i, j] > 50 else "black"
            ax.text(j, i - 0.12, f"{int(cm[i, j])}",
                    ha="center", va="center",
                    fontsize=12, color=text_color)
            ax.text(j, i + 0.18, f"({row_pct[i, j]:.1f}%)",
                    ha="center", va="center",
                    fontsize=10, color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("row %", fontsize=11)
    ax.tick_params(which="major", length=0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"{out_stem}.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print(f"wrote {out}")
    plt.close(fig)


def main() -> None:
    for src in SOURCES:
        df = pd.read_csv(src["preds"])
        yt = df["true_label"].to_numpy()
        yp = df["rounded_prediction"].to_numpy()
        cm = confusion_matrix(yt, yp, labels=list(range(5)))
        f1 = f1_score(yt, yp, labels=list(range(5)), average=None, zero_division=0)
        print(
            f"{src['tag']}: F1=[{', '.join(f'{x:.3f}' for x in f1)}]"
            f"  acc={(yt == yp).mean():.3f}  n={len(yt)}"
        )
        plot_cm(cm, short=src["short"], out_stem=src["stem"])


if __name__ == "__main__":
    main()
