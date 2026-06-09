"""Build results/result_cache/errors_to_tag.csv from D1 cum-opt predictions.

Reads the D1 cum-opt (threshold-optimised cumulative decoder) predictions and
emits the 85-row error CSV the grader will hand-tag on three axes
(image quality, lesion pattern, label plausibility).

Rows are sorted by |margin| descending so the most confident errors — the
same ones surfaced in the top-margin table in Section 4.7.2 — sit at the top
of the tagging queue.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PREDS_CSV = (
    REPO_ROOT
    / "results"
    / "exp300_d1_dropout_cosine"
    / "threshold_opt"
    / "exp300_d1_dropout_cosine_thresh_opt_preds.csv"
)
OUT_CSV = REPO_ROOT / "results" / "result_cache" / "errors_to_tag.csv"

TAG_COLUMNS = ["quality_tag", "lesion_tag", "label_tag", "notes"]


def main() -> None:
    df = pd.read_csv(PREDS_CSV)
    errors = df.loc[df["rounded_prediction"] != df["true_label"]].copy()
    errors["margin"] = errors["raw_prediction"] - errors["true_label"]
    errors["abs_margin"] = errors["margin"].abs()
    errors = errors.sort_values("abs_margin", ascending=False).drop(columns="abs_margin")

    for col in TAG_COLUMNS:
        errors[col] = ""

    ordered_cols = [
        "id_code",
        "true_label",
        "rounded_prediction",
        "raw_prediction",
        "margin",
        *TAG_COLUMNS,
    ]
    errors = errors[ordered_cols]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(OUT_CSV, index=False)

    assert len(errors) == 85, f"expected 85 errors, got {len(errors)}"
    print(f"wrote {len(errors)} error rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
