"""Build the binary (referable DR, grade>=2) confusion matrix for the
ensemble (exp300+exp701+exp705), styled to match the per-model
Blues-colormap row-percentage CM figures.

Saves: figures/cmbin_ensemble_pct.{png,pdf}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PREDS_CSV = (
    REPO_ROOT
    / "results"
    / "ensemble_300_701_705"
    / "ensemble_300_701_705_argmax_preds.csv"
)
OUT_DIR = REPO_ROOT / "figures"
OUT_STEM = "cmbin_ensemble_pct"

LABELS = ["Non-referable\n(0,1)", "Referable\n(2,3,4)"]


def main() -> None:
    df = pd.read_csv(PREDS_CSV)
    y_true = (df["true_label"].to_numpy() >= 2).astype(int)
    y_pred = (df["rounded_prediction"].to_numpy() >= 2).astype(int)

    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    row_sums = cm.sum(axis=1, keepdims=True)
    row_pct = np.divide(cm, row_sums, where=row_sums > 0) * 100.0

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=100)

    ax.set_xticks(range(2))
    ax.set_yticks(range(2))
    ax.set_xticklabels(LABELS, fontsize=11, rotation=30, ha="right")
    ax.set_yticklabels(LABELS, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12, labelpad=10)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(
        "Binary CM (%) - referable \u2265 Moderate\n"
        "Ensemble (D1 + H1 + H5)",
        fontsize=12,
    )

    for i in range(2):
        for j in range(2):
            text_color = "white" if row_pct[i, j] > 50 else "black"
            ax.text(j, i - 0.08, f"{int(cm[i, j])}",
                    ha="center", va="center",
                    fontsize=14, color=text_color)
            ax.text(j, i + 0.18, f"({row_pct[i, j]:.1f}%)",
                    ha="center", va="center",
                    fontsize=12, color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("row %", fontsize=11)

    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"{OUT_STEM}.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
