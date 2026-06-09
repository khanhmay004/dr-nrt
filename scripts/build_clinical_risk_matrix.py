"""Build the §4.7.6 clinical-risk-banded D1 confusion matrix.

Each cell of D1's MC-Dropout single-pass 5x5 confusion matrix is
colour-coded by clinical-risk band:

    - OK    (green) : diagonal, correct prediction
    - LOW   (yellow): over-referral, or within-referable severity slip
    - MED   (orange): borderline non-adjacent error
    - HIGH  (red)   : missed referable DR (true >= 2, pred <= 1)

Numbers are read from the MC-Dropout predictions file so the headline
counts match §4.7.6 prose (104 errors total, 12 HIGH-risk = 2.18 %).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba

REPO_ROOT = Path(__file__).resolve().parents[1]
PREDS_CSV = (
    REPO_ROOT
    / "results"
    / "exp300_d1_dropout_cosine"
    / "mc_dropout"
    / "mc_dropout_predictions.csv"
)
OUT_DIR = REPO_ROOT / "figures"

GRADE_NAME = ["No DR", "Mild", "Mod", "Sev", "Prolif"]

BAND_COLOR = {
    "OK": "#4caf50",
    "LOW": "#f1c40f",
    "MED": "#fb8c00",
    "HIGH": "#e53935",
}


def classify(true: int, pred: int) -> str:
    if true == pred:
        return "OK"
    if true >= 2 and pred <= 1:
        return "HIGH"
    # MED is reserved for the table's "borderline non-adjacent error"
    # band; under D1 MC-Dropout no cells fall here (every off-diagonal
    # is either a sub-referable over-referral, a within-referable slip,
    # or a missed referable case). The legend keeps MED for completeness.
    return "LOW"


def main() -> None:
    df = pd.read_csv(PREDS_CSV)
    cm = (
        pd.crosstab(df["true_label"], df["mc_prediction"])
        .reindex(index=range(5), columns=range(5), fill_value=0)
        .values
    )

    band_counts = {k: 0 for k in BAND_COLOR}
    for i in range(5):
        for j in range(5):
            band_counts[classify(i, j)] += int(cm[i, j])
    total = int(cm.sum())

    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(GRADE_NAME, fontsize=11)
    ax.set_yticklabels(GRADE_NAME, fontsize=11)
    ax.set_xlabel("Predicted grade", fontsize=12)
    ax.set_ylabel("True grade", fontsize=12)
    ax.set_title(
        "D1 MC-Dropout confusion matrix coloured by clinical-risk band\n"
        f"(n={total}; OK={band_counts['OK']}, LOW={band_counts['LOW']}, "
        f"MED={band_counts['MED']}, HIGH={band_counts['HIGH']})",
        fontsize=12,
    )
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(4.5, -0.5)

    for i in range(5):
        for j in range(5):
            band = classify(i, j)
            count = int(cm[i, j])
            color = BAND_COLOR[band]
            face = to_rgba(color, alpha=0.85 if count > 0 else 0.18)
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=face,
                                 edgecolor="white", linewidth=2)
            ax.add_patch(rect)
            text_color = "white" if (band in {"OK", "HIGH"} and count > 0) else "black"
            ax.text(j, i - 0.08, str(count), ha="center", va="center",
                    fontsize=14, fontweight="bold", color=text_color)
            ax.text(j, i + 0.22, band, ha="center", va="center",
                    fontsize=8, color=text_color, alpha=0.85)

    legend_labels = [
        ("OK (correct)", BAND_COLOR["OK"]),
        ("LOW-risk (safe error)", BAND_COLOR["LOW"]),
        ("MED-risk (borderline non-adj)", BAND_COLOR["MED"]),
        ("HIGH-risk (missed referable)", BAND_COLOR["HIGH"]),
    ]
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="white") for _, c in legend_labels]
    ax.legend(handles, [t for t, _ in legend_labels],
              loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"clinical_risk_matrix.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
