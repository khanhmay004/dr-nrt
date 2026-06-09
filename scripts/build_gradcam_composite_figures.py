"""Compose the §4.7.4 / §4.7.6 Grad-CAM figures from existing per-image strips.

Each per-image strip in results/explainability_cache/exp300/ already has
(original | background | Grad-CAM@pred | Grad-CAM@true | occlusion | IG |
 softmax bar). This script stacks selected strips into four composite
figures:

    - gradcam_correct_exemplars_5grade.pdf  (5 rows, one per grade)
    - gradcam_correct_vs_wrong_prolif.pdf   (correct vs wrong Prolif)
    - gradcam_per_pair_gallery.pdf          (4 confusion pairs, 2 rows each)
    - high_risk_prolif_to_mild.pdf          (3 Prolif->Mild rows)

No model inference is performed — this is a pure layout script.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
CORRECT_CSV = REPO_ROOT / "results" / "explainability_cache" / "exp300" / "panel_index.csv"
ERRORS_CSV = REPO_ROOT / "results" / "explainability_cache" / "exp300" / "errors_panel_index.csv"
CORRECT_DIR = REPO_ROOT / "results" / "explainability_cache" / "exp300" / "galleries"
ERRORS_DIR = REPO_ROOT / "results" / "explainability_cache" / "exp300" / "errors_gallery"
OUT_DIR = REPO_ROOT / "figures"

GRADE_NAME = {0: "No DR", 1: "Mild", 2: "Mod", 3: "Sev", 4: "Prolif"}


def _save(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=180)
        print(f"wrote {out}")
    plt.close(fig)


def _stack_strips(
    strips: Sequence[tuple[str, Path]],
    figsize: tuple[float, float],
    suptitle: str,
    section_dividers: Sequence[tuple[int, str]] = (),
) -> plt.Figure:
    """strips = list of (row_label, png_path). section_dividers =
    [(row_index, section_label), ...] for banner rows inserted before that row."""
    n_rows = len(strips) + len(section_dividers)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(n_rows, 1, hspace=0.15)

    divider_map = {idx: label for idx, label in section_dividers}
    insert = 0
    for row_i, (label, path) in enumerate(strips):
        while (row_i + insert) in divider_map:
            banner_ax = fig.add_subplot(gs[row_i + insert, 0])
            banner_ax.text(0.5, 0.5, divider_map[row_i + insert], ha="center", va="center",
                           fontsize=13, fontweight="bold",
                           bbox=dict(facecolor="#4a7fb5", edgecolor="none", pad=6),
                           color="white")
            banner_ax.set_axis_off()
            insert += 1
        ax = fig.add_subplot(gs[row_i + insert, 0])
        img = Image.open(path).convert("RGB")
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=10, labelpad=40)

    fig.suptitle(suptitle, fontsize=13, y=0.995)
    return fig


def plot_correct_exemplars_5grade() -> None:
    df = pd.read_csv(CORRECT_CSV)
    strips: list[tuple[str, Path]] = []
    for grade in range(5):
        cand = df[df["true"] == grade]
        if cand.empty:
            continue
        pick = cand.iloc[0]
        path = CORRECT_DIR / f"correct_hiconf_{pick['id_code']}.png"
        strips.append((f"{GRADE_NAME[grade]}\n(correct)", path))
    fig = _stack_strips(
        strips,
        figsize=(15, 2.0 * len(strips) + 0.5),
        suptitle="D1 correct-prediction Grad-CAM exemplars, one per grade",
    )
    _save(fig, "gradcam_correct_exemplars_5grade")


def plot_correct_vs_wrong_prolif() -> None:
    correct_df = pd.read_csv(CORRECT_CSV)
    err_df = pd.read_csv(ERRORS_CSV)
    correct_prolif = correct_df[correct_df["true"] == 4].head(5)
    wrong_prolif = err_df[err_df["true"] == 4].sort_values("margin").head(5)

    strips: list[tuple[str, Path]] = []
    for _, row in correct_prolif.iterrows():
        strips.append((f"correct\n{row['id_code']}",
                       CORRECT_DIR / f"correct_hiconf_{row['id_code']}.png"))
    for _, row in wrong_prolif.iterrows():
        pred = int(row["csv_pred"])
        strips.append((f"wrong -> {GRADE_NAME[pred]}\n{row['id_code']}",
                       ERRORS_DIR / f"err_true4_pred{pred}_{row['id_code']}.png"))

    dividers = [(0, "Correctly-classified Proliferative (5)"),
                (6, "Misclassified Proliferative (5, sorted by margin)")]
    fig = _stack_strips(
        strips,
        figsize=(15, 1.9 * len(strips) + 2.5),
        suptitle="Proliferative class: correct vs misclassified Grad-CAM attention",
        section_dividers=dividers,
    )
    _save(fig, "gradcam_correct_vs_wrong_prolif")


def plot_per_pair_gallery() -> None:
    err_df = pd.read_csv(ERRORS_CSV)
    pairs = [
        ("NoDR <-> Mild", [(0, 1), (1, 0)]),
        ("Mild <-> Mod", [(1, 2), (2, 1)]),
        ("Mod <-> Sev", [(2, 3), (3, 2)]),
        ("Sev <-> Prolif", [(3, 4), (4, 3)]),
    ]
    strips: list[tuple[str, Path]] = []
    dividers: list[tuple[int, str]] = []
    row_cursor = 0
    for pair_label, pair_list in pairs:
        dividers.append((row_cursor, pair_label))
        row_cursor += 1
        for t, p in pair_list:
            cand = err_df[(err_df["true"] == t) & (err_df["csv_pred"] == p)]
            if cand.empty:
                continue
            row = cand.iloc[0]
            strips.append(
                (f"{GRADE_NAME[t]} -> {GRADE_NAME[p]}\n{row['id_code']}",
                 ERRORS_DIR / f"err_true{t}_pred{p}_{row['id_code']}.png")
            )
            row_cursor += 1
    fig = _stack_strips(
        strips,
        figsize=(15, 1.9 * len(strips) + 3.0),
        suptitle="Per-pair Grad-CAM gallery (one exemplar per direction)",
        section_dividers=dividers,
    )
    _save(fig, "gradcam_per_pair_gallery")


def plot_high_risk_prolif_to_mild() -> None:
    err_df = pd.read_csv(ERRORS_CSV)
    rows = err_df[(err_df["true"] == 4) & (err_df["csv_pred"] == 1)].sort_values("margin")
    strips: list[tuple[str, Path]] = []
    for _, row in rows.iterrows():
        strips.append(
            (f"{row['id_code']}\nraw={row['margin'] + 4:.2f}\nmargin={row['margin']:+.2f}",
             ERRORS_DIR / f"err_true4_pred1_{row['id_code']}.png")
        )
    fig = _stack_strips(
        strips,
        figsize=(15, 2.0 * len(strips) + 0.5),
        suptitle="HIGH-risk Proliferative -> Mild case studies (3 of 3)",
    )
    _save(fig, "high_risk_prolif_to_mild")


def main() -> None:
    plot_correct_exemplars_5grade()
    plot_correct_vs_wrong_prolif()
    plot_per_pair_gallery()
    plot_high_risk_prolif_to_mild()


if __name__ == "__main__":
    main()
