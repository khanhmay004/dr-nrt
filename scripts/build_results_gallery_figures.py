"""Build the §4.7 image-gallery figures (grids of fundus photos).

Emits (all saved under figures/):
    - high_margin_error_gallery.pdf   (§4.7.2; 6 high-margin errors, 2x3)
    - sharpness_q1_q4_samples.pdf     (§4.7.3; Q1 vs Q4, 2x4)
    - mislabel_gallery.pdf            (§4.7.5; 6 candidate mislabels, 2x3)

All selections are made from on-disk CSVs (errors_to_tag, aptos_quality,
candidate_mislabels). Images are read from data/test_split/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
PREDS_CSV = (
    REPO_ROOT
    / "results"
    / "exp300_d1_dropout_cosine"
    / "threshold_opt"
    / "exp300_d1_dropout_cosine_thresh_opt_preds.csv"
)
ERRORS_CSV = REPO_ROOT / "results" / "result_cache" / "errors_to_tag.csv"
QUALITY_CSV = REPO_ROOT / "results" / "eda_cache" / "aptos_quality.csv"
CAND_CSV = REPO_ROOT / "results" / "confusion_cache" / "candidate_mislabels.csv"
IMG_DIR = REPO_ROOT / "data" / "test_split"
OUT_DIR = REPO_ROOT / "figures"

GRADE_NAME = {0: "No DR", 1: "Mild", 2: "Mod", 3: "Sev", 4: "Prolif"}


def _save(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print(f"wrote {out}")
    plt.close(fig)


def _imshow(ax, id_code: str, title: str, caption: str) -> None:
    img_path = IMG_DIR / f"{id_code}.png"
    img = Image.open(img_path).convert("RGB")
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(caption, fontsize=8, family="monospace")


def plot_high_margin_gallery() -> None:
    """Pick 6 rows from top-10: 3 Prolif->Mild + 2 Prolif->Mod + NoDR->Mod and Mod->Prolif."""
    picks = [
        "b37aae3c8fe1",  # Prolif -> Mild
        "8bed09514c3b",  # Prolif -> Mild
        "eaa0dfbd5024",  # Prolif -> Mild
        "8fd7ad26e691",  # Prolif -> Mod
        "735836b1ffa6",  # NoDR -> Mod
        "b9127e38d9b9",  # Mod -> Prolif
    ]
    df = pd.read_csv(ERRORS_CSV).set_index("id_code")
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.5))
    for ax, pid in zip(axes.flat, picks):
        row = df.loc[pid]
        t = int(row["true_label"])
        p = int(row["rounded_prediction"])
        raw = float(row["raw_prediction"])
        margin = float(row["margin"])
        title = f"{GRADE_NAME[t]} \u2192 {GRADE_NAME[p]}"
        cap = (
            f"id: {pid}\n"
            f"true={t}  pred={p}  raw={raw:.2f}\n"
            f"margin={margin:+.2f}"
        )
        _imshow(ax, pid, title, cap)
    fig.suptitle("High-margin D1 cum-opt errors (top-10 subset)", fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, "high_margin_error_gallery")


def plot_sharpness_q1_q4() -> None:
    """2x4: 4 Q1 (blurriest), 4 Q4 (sharpest), balanced by grade across rows."""
    preds = pd.read_csv(PREDS_CSV)
    q = pd.read_csv(QUALITY_CSV)
    q = q[q["split"] == "test"].merge(
        preds[["id_code", "true_label", "rounded_prediction"]],
        left_on="code",
        right_on="id_code",
        how="inner",
    )
    q = q.sort_values("laplacian_var").reset_index(drop=True)
    n = len(q)
    q1 = q.iloc[: n // 4].copy()
    q4 = q.iloc[3 * n // 4 :].copy()

    def pick_one_per_grade(sub: pd.DataFrame, grades: Iterable[int]) -> list[pd.Series]:
        out: list[pd.Series] = []
        used: set[str] = set()
        for g in grades:
            cands = sub[(sub["true_label"] == g) & (~sub["id_code"].isin(used))]
            if cands.empty:
                cands = sub[~sub["id_code"].isin(used)]
            pick = cands.iloc[len(cands) // 2]  # median-sharpness within that grade
            out.append(pick)
            used.add(pick["id_code"])
        return out

    grades = [0, 1, 2, 3]
    q1_picks = pick_one_per_grade(q1, grades)
    q4_picks = pick_one_per_grade(q4, grades)

    fig, axes = plt.subplots(2, 4, figsize=(15, 8.5))
    for row_axes, row_picks, row_label in (
        (axes[0], q1_picks, "Q1 (blurriest)"),
        (axes[1], q4_picks, "Q4 (sharpest)"),
    ):
        for ax, pick in zip(row_axes, row_picks):
            t = int(pick["true_label"])
            p = int(pick["rounded_prediction"])
            correct = "correct" if t == p else "wrong"
            title = f"{row_label} | {GRADE_NAME[t]} \u2192 {GRADE_NAME[p]} ({correct})"
            cap = (
                f"id: {pick['id_code']}\n"
                f"laplacian_var={pick['laplacian_var']:.1f}\n"
                f"true={t}  pred={p}"
            )
            _imshow(ax, pick["id_code"], title, cap)

    fig.suptitle(
        f"Sharpness Q1 vs Q4 exemplars (1 per grade; overall Q1={q1['laplacian_var'].max():.1f}-max, "
        f"Q4={q4['laplacian_var'].min():.1f}-min Laplacian var)",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    _save(fig, "sharpness_q1_q4_samples")


def plot_mislabel_gallery() -> None:
    """Pick 6 candidate mislabels spanning multiple directions."""
    cand = pd.read_csv(CAND_CSV)
    cand["pair"] = list(zip(cand["true_label"].astype(int), cand["both_pred"].astype(int)))

    target_directions = [
        (1, 2),  # Mild -> Mod
        (1, 2),
        (2, 1),  # Mod -> Mild
        (3, 2),  # Sev -> Mod
        (4, 2),  # Prolif -> Mod
        (4, 1),  # Prolif -> Mild
    ]
    picks: list[str] = []
    used: set[str] = set()
    for pair in target_directions:
        cands = cand[(cand["pair"] == pair) & (~cand["id_code"].isin(used))]
        if cands.empty:
            continue
        pick = cands.iloc[0]
        picks.append(pick["id_code"])
        used.add(pick["id_code"])

    cand_idx = cand.set_index("id_code")
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.5))
    for ax, pid in zip(axes.flat, picks):
        row = cand_idx.loc[pid]
        t = int(row["true_label"])
        p = int(row["both_pred"])
        r701 = float(row["r701"]) if not pd.isna(row["r701"]) else None
        r300 = float(row["r300"]) if not pd.isna(row["r300"]) else None
        title = f"APTOS {GRADE_NAME[t]} \u2192 consensus {GRADE_NAME[p]}"
        cap = f"id: {pid}\nlabel={t}  both_pred={p}"
        if r701 is not None and r300 is not None:
            cap += f"\nr300={r300:.2f}  r701={r701:.2f}"
        _imshow(ax, pid, title, cap)

    fig.suptitle(
        f"Candidate mislabels: D1+H1 high-confidence consensus against public label "
        f"(6 of {len(cand)})",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    _save(fig, "mislabel_gallery")


def main() -> None:
    plot_high_margin_gallery()
    plot_sharpness_q1_q4()
    plot_mislabel_gallery()


if __name__ == "__main__":
    main()
