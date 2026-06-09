"""Aggregate the user-filled errors_to_tag.csv into a tag-taxonomy summary.

Run after the grader fills the quality_tag / lesion_tag / label_tag columns
in `results/result_cache/errors_to_tag.csv`. Produces:

    results/result_cache/errors_tag_summary.json  (machine-readable digest)

and prints counts-per-axis plus a label_tag x (true->pred) cross-tab to
stdout. The printed tables are what should be folded into the
"Error-image tag taxonomy" paragraph at the end of Section 4.7.2.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TAG_CSV = REPO_ROOT / "results" / "result_cache" / "errors_to_tag.csv"
OUT_JSON = REPO_ROOT / "results" / "result_cache" / "errors_tag_summary.json"

GRADE_NAME = {0: "NoDR", 1: "Mild", 2: "Mod", 3: "Sev", 4: "Prolif"}


def _counts(series: pd.Series) -> dict[str, int]:
    filled = series.fillna("").astype(str).str.strip()
    filled = filled[filled != ""]
    return filled.value_counts().to_dict()


def main() -> None:
    df = pd.read_csv(TAG_CSV)
    total = len(df)
    tagged = df[["quality_tag", "lesion_tag", "label_tag"]].fillna("").apply(
        lambda s: (s.astype(str).str.strip() != "").any(), axis=1
    ).sum()

    quality = _counts(df["quality_tag"])
    lesion = _counts(df["lesion_tag"])
    label = _counts(df["label_tag"])

    df["pair"] = df.apply(
        lambda r: f"{GRADE_NAME[int(r['true_label'])]}->{GRADE_NAME[int(r['rounded_prediction'])]}",
        axis=1,
    )
    label_by_pair: dict[str, dict[str, int]] = {}
    for lbl, sub in df.groupby(df["label_tag"].fillna("").astype(str).str.strip()):
        if not lbl:
            continue
        label_by_pair[lbl] = sub["pair"].value_counts().to_dict()

    likely_mask = df["label_tag"].fillna("").astype(str).str.strip().isin(
        {"likely_mislabel", "ambiguous_boundary"}
    )
    likely_rows = (
        df.loc[likely_mask, ["id_code", "pair", "margin", "label_tag", "notes"]]
        .to_dict(orient="records")
    )

    digest = {
        "total_errors": int(total),
        "tagged_rows": int(tagged),
        "quality_tag_counts": quality,
        "lesion_tag_counts": lesion,
        "label_tag_counts": label,
        "label_tag_by_pair": label_by_pair,
        "likely_mislabel_or_ambiguous_rows": likely_rows,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as fh:
        json.dump(digest, fh, indent=2)

    print(f"Tagged {tagged}/{total} errors\n")
    print("quality_tag:", quality)
    print("lesion_tag :", lesion)
    print("label_tag  :", label)
    print("\nlabel_tag x (true -> pred):")
    for lbl, pairs in label_by_pair.items():
        print(f"  {lbl}:")
        for pair, n in pairs.items():
            print(f"    {pair}: {n}")
    if likely_rows:
        print(f"\n{len(likely_rows)} likely_mislabel / ambiguous_boundary rows:")
        for r in likely_rows:
            print(
                f"  {r['id_code']}  {r['pair']:<14}  margin={r['margin']:+.3f}  "
                f"[{r['label_tag']}]  {r.get('notes','') or ''}"
            )
    print(f"\nDigest written to {OUT_JSON}")


if __name__ == "__main__":
    main()
