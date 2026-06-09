"""Build the §4.7 chart-style figures (bars/histograms/curves/Sankey).

Emits (all saved under figures/):
    - error_pair_distribution.pdf   (§4.7.1)
    - margin_distribution.pdf       (§4.7.2)
    - error_stratification.pdf      (§4.7.3; 3 panels: sharpness, illum, size)
    - insertion_deletion_curves.pdf (§4.7.4; per-image insertion/deletion AUC)
    - sankey_error_flow.pdf         (§4.7.6)

All numbers are derived from the D1 cum-opt predictions
(results/exp300_d1_dropout_cosine/threshold_opt/*.csv) to stay consistent
with the §4.7.1/§4.7.2 narrative. Stratification panels use the APTOS-test
subset of results/eda_cache/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[1]
PREDS_CSV = (
    REPO_ROOT
    / "results"
    / "exp300_d1_dropout_cosine"
    / "threshold_opt"
    / "exp300_d1_dropout_cosine_thresh_opt_preds.csv"
)
ERRORS_CSV = REPO_ROOT / "results" / "result_cache" / "errors_to_tag.csv"
CAND_CSV = REPO_ROOT / "results" / "confusion_cache" / "candidate_mislabels.csv"
QUALITY_CSV = REPO_ROOT / "results" / "eda_cache" / "aptos_quality.csv"
ILLUM_CSV = REPO_ROOT / "results" / "eda_cache" / "illumination_regime.csv"
SIZES_CSV = REPO_ROOT / "results" / "eda_cache" / "aptos_sizes.csv"
INSDEL_CSV = REPO_ROOT / "results" / "explainability_cache" / "insertion_deletion.csv"

OUT_DIR = REPO_ROOT / "figures"
GRADE_NAME = {0: "No DR", 1: "Mild", 2: "Mod", 3: "Sev", 4: "Prolif"}


def _save(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print(f"wrote {out}")
    plt.close(fig)


def _load_errors_for_pairs() -> pd.DataFrame:
    df = pd.read_csv(PREDS_CSV)
    err = df.loc[df["rounded_prediction"] != df["true_label"]].copy()
    err["true_label"] = err["true_label"].astype(int)
    err["rounded_prediction"] = err["rounded_prediction"].astype(int)
    err["gap"] = err["rounded_prediction"] - err["true_label"]
    return err


def plot_error_pair_distribution() -> None:
    err = _load_errors_for_pairs()
    groups = [
        ("NoDR-Mild", {(0, 1), (1, 0)}),
        ("Mild-Mod", {(1, 2), (2, 1)}),
        ("Mod-Sev", {(2, 3), (3, 2)}),
        ("Sev-Prolif", {(3, 4), (4, 3)}),
    ]

    def pair_count(pairs: set[tuple[int, int]], *, under: bool) -> int:
        """under = model picked a LOWER grade than truth (true > pred)."""
        total = 0
        for t, p in pairs:
            sub = err[(err["true_label"] == t) & (err["rounded_prediction"] == p)]
            if under and t > p:
                total += len(sub)
            if (not under) and p > t:
                total += len(sub)
        return total

    over_counts = [pair_count(g[1], under=False) for g in groups]
    under_counts = [pair_count(g[1], under=True) for g in groups]

    non_adj = err[err["gap"].abs() >= 2]
    non_adj_under = int((non_adj["gap"] < 0).sum())
    non_adj_over = int((non_adj["gap"] > 0).sum())

    labels = [g[0] for g in groups] + ["non-adj"]
    over_counts.append(non_adj_over)
    under_counts.append(non_adj_under)

    x = np.arange(len(labels))
    width = 0.62
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.bar(x, over_counts, width, label="over-grading (pred > true)", color="#d9534f")
    ax.bar(x, under_counts, width, bottom=over_counts, label="under-grading (pred < true)", color="#4a7fb5")
    for i, (ov, un) in enumerate(zip(over_counts, under_counts)):
        if ov:
            ax.text(i, ov / 2, str(ov), ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        if un:
            ax.text(i, ov + un / 2, str(un), ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        total = ov + un
        ax.text(i, total + 0.6, str(total), ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Error count")
    ax.set_title(f"Confusion-pair error distribution (D1 cum-opt, n={len(err)})")
    ax.legend(loc="upper right", frameon=False)
    ax.set_ylim(0, max(a + b for a, b in zip(over_counts, under_counts)) * 1.18)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    _save(fig, "error_pair_distribution")


def plot_margin_distribution() -> None:
    errs = pd.read_csv(ERRORS_CSV)
    errs["abs_m_true"] = (errs["raw_prediction"] - errs["true_label"]).abs()

    cand = pd.read_csv(CAND_CSV)
    preds = pd.read_csv(PREDS_CSV).set_index("id_code")
    cand = cand.join(preds[["raw_prediction"]], on="id_code", how="inner")
    cand["abs_m_true"] = (cand["raw_prediction"] - cand["true_label"]).abs()

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    bins = np.linspace(0, 3, 31)
    ax.hist(errs["abs_m_true"], bins=bins, color="#4a7fb5", alpha=0.85, label=f"All errors (n={len(errs)})")
    ax.hist(cand["abs_m_true"], bins=bins, color="#d9534f", alpha=0.6, label=f"Candidate mislabels (n={len(cand)})")
    ax.axvline(1.0, color="#555", linestyle="--", linewidth=1)
    ax.axvline(2.0, color="#222", linestyle="--", linewidth=1)
    ax.text(1.02, ax.get_ylim()[1] * 0.92, "adj / non-adj", fontsize=9, color="#555")
    ax.text(2.02, ax.get_ylim()[1] * 0.92, "serious (>=2 grades)", fontsize=9, color="#222")
    ax.set_xlabel(r"$|\mathrm{raw\ prediction}\ -\ \mathrm{true\ grade}|$")
    ax.set_ylabel("Count")
    ax.set_title("Margin distribution: D1 cum-opt errors vs candidate mislabels")
    ax.legend(frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    _save(fig, "margin_distribution")


def plot_error_stratification() -> None:
    preds = pd.read_csv(PREDS_CSV)
    preds["err"] = (preds["rounded_prediction"] != preds["true_label"]).astype(int)

    q = pd.read_csv(QUALITY_CSV)
    q = q[q["split"] == "test"].copy()
    q = q.merge(preds[["id_code", "err"]], left_on="code", right_on="id_code", how="inner")
    q["sharp_q"] = pd.qcut(q["laplacian_var"], 4, labels=["Q1 (blurriest)", "Q2", "Q3", "Q4 (sharpest)"])
    sharp = q.groupby("sharp_q", observed=True)["err"].agg(["mean", "count"]).reset_index()

    illum = pd.read_csv(ILLUM_CSV)
    illum = illum[illum["split"] == "test"].merge(preds[["id_code", "err"]], left_on="code", right_on="id_code", how="inner")
    illum_agg = illum.groupby("regime")["err"].agg(["mean", "count"]).reset_index()
    regime_order = ["dim", "uneven", "well-lit"]
    illum_agg["regime"] = pd.Categorical(illum_agg["regime"], categories=regime_order, ordered=True)
    illum_agg = illum_agg.sort_values("regime")

    sizes = pd.read_csv(SIZES_CSV)
    sizes = sizes[sizes["split"] == "test"].merge(preds[["id_code", "err"]], left_on="id", right_on="id_code", how="inner")
    bins = [0, 1024, 2048, sizes["height"].max() + 1]
    labels = ["<=1024", "1025-2048", ">2048"]
    sizes["bucket"] = pd.cut(sizes["height"], bins=bins, labels=labels, include_lowest=True)
    size_agg = sizes.groupby("bucket", observed=True)["err"].agg(["mean", "count"]).reset_index()

    overall = preds["err"].mean()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    specs = [
        ("Sharpness (Laplacian var)", sharp, "sharp_q", axes[0]),
        ("Illumination regime", illum_agg, "regime", axes[1]),
        ("Native height (px)", size_agg, "bucket", axes[2]),
    ]
    for title, df, xcol, ax in specs:
        xs = df[xcol].astype(str).tolist()
        rates = (df["mean"] * 100).tolist()
        ns = df["count"].tolist()
        bars = ax.bar(xs, rates, color="#4a7fb5")
        for bar, rate, n in zip(bars, rates, ns):
            ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.6,
                    f"{rate:.1f}%\nn={n}", ha="center", va="bottom", fontsize=9)
        ax.axhline(overall * 100, color="#d9534f", linestyle="--", linewidth=1,
                   label=f"overall {overall*100:.1f}%")
        ax.set_title(title)
        ax.set_ylabel("Error rate (%)")
        ax.set_ylim(0, max(rates) * 1.35)
        ax.legend(frameon=False, loc="upper right", fontsize=9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("D1 cum-opt error rate stratified by image-acquisition quality", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, "error_stratification")


def plot_insertion_deletion() -> None:
    df = pd.read_csv(INSDEL_CSV)
    ins = df["insertion_auc"].values
    dele = df["deletion_auc"].values
    gap = df["faithfulness"].values

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    ax = axes[0]
    ax.boxplot([ins, dele], labels=["insertion AUC", "deletion AUC"], patch_artist=True,
               boxprops=dict(facecolor="#cfe1f2"), medianprops=dict(color="#d9534f"))
    for i, arr in enumerate([ins, dele], start=1):
        ax.scatter(np.full_like(arr, i) + np.random.uniform(-0.08, 0.08, len(arr)),
                   arr, alpha=0.5, color="#555", s=14)
    ax.axhline(ins.mean(), color="#4a7fb5", linestyle=":", linewidth=1)
    ax.axhline(dele.mean(), color="#d9534f", linestyle=":", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUC")
    ax.set_title(f"Per-image insertion vs deletion AUC (n={len(df)})")
    ax.text(0.02, 0.97,
            f"mean insertion = {ins.mean():.3f}\nmean deletion  = {dele.mean():.3f}\n"
            f"mean gap       = {gap.mean():.3f}",
            transform=ax.transAxes, va="top", family="monospace", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="#ccc"))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax = axes[1]
    ax.scatter(ins, dele, color="#4a7fb5", alpha=0.8)
    lims = (0, 1)
    ax.plot(lims, lims, color="#888", linestyle="--", linewidth=1, label="y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("insertion AUC")
    ax.set_ylabel("deletion AUC")
    ax.set_title("Per-image insertion vs deletion (points below diagonal = faithful)")
    ax.legend(frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    _save(fig, "insertion_deletion_curves")


def plot_sankey() -> None:
    preds = pd.read_csv(PREDS_CSV)
    cm = pd.crosstab(preds["true_label"].astype(int), preds["rounded_prediction"].astype(int))
    cm = cm.reindex(index=[0, 1, 2, 3, 4], columns=[0, 1, 2, 3, 4], fill_value=0).values

    fig, ax = plt.subplots(figsize=(10, 6))
    left_x, right_x = 0.1, 0.9
    true_totals = cm.sum(axis=1)
    pred_totals = cm.sum(axis=0)
    n = 5
    left_heights = true_totals / true_totals.sum()
    right_heights = pred_totals / pred_totals.sum()

    gap = 0.02
    left_ys = []
    y = 1.0
    for h in left_heights:
        left_ys.append((y - h, y))
        y -= h + gap
    right_ys = []
    y = 1.0
    for h in right_heights:
        right_ys.append((y - h, y))
        y -= h + gap

    for i, (lo, hi) in enumerate(left_ys):
        ax.fill_betweenx([lo, hi], left_x - 0.03, left_x, color="#4a7fb5")
        ax.text(left_x - 0.04, (lo + hi) / 2, f"{GRADE_NAME[i]} (n={int(true_totals[i])})",
                ha="right", va="center", fontsize=10)
    for j, (lo, hi) in enumerate(right_ys):
        ax.fill_betweenx([lo, hi], right_x, right_x + 0.03, color="#4a7fb5")
        ax.text(right_x + 0.04, (lo + hi) / 2, f"{GRADE_NAME[j]} (n={int(pred_totals[j])})",
                ha="left", va="center", fontsize=10)

    left_cursor = [ys[1] for ys in left_ys]
    right_cursor = [ys[1] for ys in right_ys]

    def color_for(t: int, p: int) -> str:
        if t == p:
            return "#4caf50"
        if (t >= 2) and (p <= 1):
            return "#e53935"
        return "#f1c40f"

    flow_total = cm.sum()
    for i in range(n):
        for j in range(n):
            flow = cm[i, j]
            if flow == 0:
                continue
            h = flow / flow_total
            left_top = left_cursor[i]
            left_bot = left_top - h
            right_top = right_cursor[j]
            right_bot = right_top - h
            left_cursor[i] = left_bot
            right_cursor[j] = right_bot
            color = color_for(i, j)
            verts_x = np.linspace(left_x, right_x, 40)
            t = (verts_x - left_x) / (right_x - left_x)
            smooth = 0.5 - 0.5 * np.cos(np.pi * t)
            top = left_top + (right_top - left_top) * smooth
            bot = left_bot + (right_bot - left_bot) * smooth
            ax.fill_between(verts_x, bot, top, color=color, alpha=0.55, linewidth=0)

    legend_handles = [
        Patch(color="#4caf50", label="OK (diagonal)"),
        Patch(color="#f1c40f", label="LOW-risk (safe error)"),
        Patch(color="#e53935", label="HIGH-risk (missed referable)"),
    ]
    ax.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False,
              bbox_to_anchor=(0.5, -0.08))

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.1)
    ax.set_axis_off()
    ax.set_title("APTOS-2019 test flow: true grade \u2192 D1 cum-opt prediction", fontsize=12)
    _save(fig, "sankey_error_flow")


def main() -> None:
    plot_error_pair_distribution()
    plot_margin_distribution()
    plot_error_stratification()
    plot_insertion_deletion()
    plot_sankey()


if __name__ == "__main__":
    main()
