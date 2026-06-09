"""Build the §4.7 MISCLASSIFIED CASES overview figure.

Picks one mid-margin error per true grade (NoDR, Mild, Moderate, Severe,
Proliferative) from D1's cum-opt predictions and arranges them as a
1 x 5 strip. Each panel shows the preprocessed fundus image with a
caption reporting image ID, true grade, D1 rounded prediction, D1 raw
prediction, and signed margin.

Selections are "mid-margin" — deliberately not the largest |margin| in
each class, so the strip shows typical rather than pathological
failures. The most extreme Proliferative->Mild cases are kept for
`tab:results-top-errors` and `high_margin_error_gallery`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
ERRORS_CSV = REPO_ROOT / "results" / "result_cache" / "errors_to_tag.csv"
IMG_DIR = REPO_ROOT / "data" / "test_split"
OUT_DIR = REPO_ROOT / "figures"
OUT_PDF = OUT_DIR / "misclassified_cases_overview.pdf"
OUT_PNG = OUT_DIR / "misclassified_cases_overview.png"

GRADE_NAME = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Prolif"}

# Mid-margin picks, one per direction. Rationale captured inline.
PICKS = [
    "2d07162a13b1",  # NoDR->Mild,  margin +1.36, underexposed, label_correct
    "f6f7dba7104d",  # Mild->Mod,   margin +0.98, lens_dirt MA_only, label_correct
    "e34fa07bd64d",  # Mod->Sev,    margin +0.78, ok cotton_wool, ambiguous
    "24b87f744598",  # Sev->Mod,    margin -0.87, ok exudates, label_correct
    "63b4d030b016",  # Prolif->Mod, margin -1.64, ok vitreous_hemo, label_correct
]


def main() -> None:
    df = pd.read_csv(ERRORS_CSV).set_index("id_code")
    missing = [i for i in PICKS if i not in df.index]
    if missing:
        raise SystemExit(f"picks missing from errors CSV: {missing}")

    fig, axes = plt.subplots(1, 5, figsize=(17.5, 4.2))
    for ax, id_code in zip(axes, PICKS):
        row = df.loc[id_code]
        img_path = IMG_DIR / f"{id_code}.png"
        if not img_path.exists():
            raise SystemExit(f"image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        true_grade = int(row["true_label"])
        pred_grade = int(row["rounded_prediction"])
        raw = float(row["raw_prediction"])
        margin = float(row["margin"])

        title = f"{GRADE_NAME[true_grade]} \u2192 {GRADE_NAME[pred_grade]}"
        ax.set_title(title, fontsize=12, fontweight="bold")

        caption = (
            f"id: {id_code}\n"
            f"true={true_grade} ({GRADE_NAME[true_grade]})  "
            f"pred={pred_grade} ({GRADE_NAME[pred_grade]})\n"
            f"raw={raw:.2f}  margin={margin:+.2f}"
        )
        ax.set_xlabel(caption, fontsize=9, family="monospace")

    fig.suptitle(
        "Representative misclassified cases (D1 cum-opt), one per true grade",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight", dpi=200)
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=200)
    plt.close(fig)

    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
