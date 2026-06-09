"""Build the binary (referable DR, grade>=2) confusion matrix for D1
(exp300), styled to match the per-model Blues-colormap row-percentage
CM figures.

Cell counts are pinned to the headline metrics for D1:
    sens=0.951, spec=0.927, ppv=0.898, npv=0.965
which back-solves on the 550-image APTOS test split (327 non-ref / 223
ref) to TN=303, FP=24, FN=11, TP=212.

Saves: figures/cmbin_d1_pct.{png,pdf}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "figures"
OUT_STEM = "cmbin_d1_pct"

LABELS = ["Non-referable\n(0,1)", "Referable\n(2,3,4)"]

CM = np.array(
    [
        [303, 24],
        [11, 212],
    ],
    dtype=int,
)


def main() -> None:
    cm = CM
    row_sums = cm.sum(axis=1, keepdims=True)
    row_pct = cm / row_sums * 100.0

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=100)

    ax.set_xticks(range(2))
    ax.set_yticks(range(2))
    ax.set_xticklabels(LABELS, fontsize=11, rotation=30, ha="right")
    ax.set_yticklabels(LABELS, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12, labelpad=10)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Binary CM (%) - D1 (exp 300)", fontsize=12)

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

    ax.tick_params(which="major", length=0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"{OUT_STEM}.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
