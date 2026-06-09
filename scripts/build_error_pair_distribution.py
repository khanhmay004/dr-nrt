"""Error-distribution chart for D1: under- vs over-grading per confusion pair.

Groups: four adjacent pairs (NoDR-Mild, Mild-Mod, Mod-Sev, Sev-Prolif) and a
fifth aggregate bucket for all non-adjacent errors (|true - pred| >= 2).
Two bars per group: under-grading (pred < true) and over-grading (pred > true).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
D1 = REPO / "results" / "exp300_d1_dropout_cosine" / "exp300_d1_dropout_cosine_preds.csv"
OUT_PDF = REPO / "figures" / "error_pair_distribution.pdf"
OUT_PNG = REPO / "figures" / "results" / "error_pair_distribution.png"
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

ADJACENT = [(0, 1, "NoDR-Mild"),
            (1, 2, "Mild-Mod"),
            (2, 3, "Mod-Sev"),
            (3, 4, "Sev-Prolif")]


def main() -> None:
    df = pd.read_csv(D1)
    y = df["true_label"].to_numpy().astype(int)
    yp = df["rounded_prediction"].to_numpy().astype(int)
    err = y != yp

    under = np.zeros(5, dtype=int)  # pred < true (missed severity)
    over = np.zeros(5, dtype=int)   # pred > true (over-called)

    for idx, (lo, hi, _) in enumerate(ADJACENT):
        under[idx] = int(np.sum(err & (y == hi) & (yp == lo)))
        over[idx]  = int(np.sum(err & (y == lo) & (yp == hi)))

    non_adj = err & (np.abs(y - yp) >= 2)
    under[4] = int(np.sum(non_adj & (yp < y)))
    over[4]  = int(np.sum(non_adj & (yp > y)))

    labels = [lbl for _, _, lbl in ADJACENT] + ["non-adjacent\n(|Δ|≥2)"]
    x = np.arange(len(labels))
    w = 0.38

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    b1 = ax.bar(x - w/2, under, w, label="Under-grading (pred < true)",
                color="#d62728", edgecolor="white")
    b2 = ax.bar(x + w/2, over, w, label="Over-grading (pred > true)",
                color="#1f77b4", edgecolor="white")
    for rect, v in list(zip(b1, under)) + list(zip(b2, over)):
        if v > 0:
            ax.text(rect.get_x() + rect.get_width()/2,
                    rect.get_height() + 0.5, str(v),
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x, labels)
    ax.set_ylabel("Error count")
    ax.set_xlabel("Confusion pair")
    total_err = int(err.sum())
    ax.set_title(f"D1 (exp300) error distribution by pair and direction  "
                 f"(n={total_err} errors / {len(df)} test samples)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    top = max(under.max(), over.max())
    ax.set_ylim(0, top * 1.18)
    plt.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {OUT_PDF.relative_to(REPO)} and {OUT_PNG.relative_to(REPO)}")
    print("\nCounts:")
    print(f"{'pair':<20s} {'under':>6s} {'over':>6s} {'total':>6s}")
    for lbl, u, o in zip(labels, under, over):
        print(f"{lbl.replace(chr(10),' '):<20s} {u:>6d} {o:>6d} {u+o:>6d}")
    print(f"{'TOTAL':<20s} {under.sum():>6d} {over.sum():>6d} {under.sum()+over.sum():>6d}")


if __name__ == "__main__":
    main()
