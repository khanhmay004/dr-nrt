"""Calibration metrics & reliability diagrams.

Most experiments in this repo only saved per-sample integer predictions,
so probability-based ECE is restricted to whatever models do save softmax
outputs. For the regression-style ensemble we substitute an *ordinal-rank
calibration* view (predicted-class confidence proxied from rounding margin).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar


# ---------------------------------------------------------------------------
# Bin-based calibration metrics
# ---------------------------------------------------------------------------


def _binned_calibration(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int,
) -> dict[str, np.ndarray]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_acc = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)
    bin_count = np.zeros(n_bins, dtype=np.int64)
    for i in range(n_bins):
        m = (confidences > edges[i]) & (confidences <= edges[i + 1])
        if i == 0:
            m = (confidences >= edges[i]) & (confidences <= edges[i + 1])
        if not m.any():
            continue
        bin_acc[i] = correct[m].mean()
        bin_conf[i] = confidences[m].mean()
        bin_count[i] = m.sum()
    return {"edges": edges, "acc": bin_acc, "conf": bin_conf, "count": bin_count}


def expected_calibration_error(
    probs: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Standard ECE (Naeini 2015), equal-width confidence bins."""
    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == targets).astype(float)
    binned = _binned_calibration(confidences, correct, n_bins)
    n = len(targets)
    return float(np.sum(binned["count"] * np.abs(binned["acc"] - binned["conf"])) / n)


def adaptive_calibration_error(
    probs: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 15,
) -> float:
    """ACE — equal-mass quantile bins. More stable on imbalanced confidence dists."""
    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == targets).astype(float)
    order = np.argsort(confidences)
    chunks = np.array_split(order, n_bins)
    n = len(targets)
    ace = 0.0
    for ch in chunks:
        if len(ch) == 0:
            continue
        ace += len(ch) * abs(correct[ch].mean() - confidences[ch].mean())
    return float(ace / n)


def maximum_calibration_error(
    probs: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 15,
) -> float:
    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == targets).astype(float)
    binned = _binned_calibration(confidences, correct, n_bins)
    nonempty = binned["count"] > 0
    if not nonempty.any():
        return 0.0
    return float(np.max(np.abs(binned["acc"][nonempty] - binned["conf"][nonempty])))


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------


def reliability_diagram(
    probs: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 15,
    title: str = "Reliability Diagram",
    ax=None,
):
    """Draw reliability diagram. Returns the matplotlib Axes."""
    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == targets).astype(float)
    binned = _binned_calibration(confidences, correct, n_bins)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    centers = 0.5 * (binned["edges"][:-1] + binned["edges"][1:])
    nonempty = binned["count"] > 0
    ax.bar(
        centers[nonempty], binned["acc"][nonempty],
        width=1.0 / n_bins, alpha=0.7, edgecolor="black", label="Accuracy",
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    return ax


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    """Find scalar T > 0 that minimises NLL of softmax(logits / T).

    Pure numpy implementation so no torch dependency leaks into analysis code.
    """
    targets = targets.astype(np.int64)
    n, k = logits.shape

    def nll(T: float) -> float:
        if T <= 1e-3:
            return 1e9
        z = logits / T
        z = z - z.max(axis=1, keepdims=True)
        log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return float(-log_probs[np.arange(n), targets].mean())

    res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
    return float(res.x)


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Ordinal-rank calibration for regression-style ensemble
# ---------------------------------------------------------------------------


def regression_margin_confidence(raw: np.ndarray, thresholds: list[float]) -> np.ndarray:
    """Confidence proxy for a regression head with OptimizedRounder thresholds.

    Distance from the *nearest* class boundary, normalised by the gap between
    boundaries. 1.0 means the regression score sits in the middle of a class
    interval; 0.0 means it is exactly on a boundary.
    """
    raw = np.asarray(raw, dtype=np.float64)
    t = sorted(thresholds)
    intervals = [(-np.inf, t[0])]
    for i in range(len(t) - 1):
        intervals.append((t[i], t[i + 1]))
    intervals.append((t[-1], np.inf))

    conf = np.zeros_like(raw)
    for i, (lo, hi) in enumerate(intervals):
        m = (raw >= lo) & (raw < hi)
        if not m.any():
            continue
        if np.isfinite(lo) and np.isfinite(hi):
            half = (hi - lo) / 2.0
            mid = (hi + lo) / 2.0
            conf[m] = 1.0 - np.abs(raw[m] - mid) / half
        elif not np.isfinite(lo):
            ref = hi
            d = ref - raw[m]
            conf[m] = np.clip(d, 0, 1.0)
        else:
            ref = lo
            d = raw[m] - ref
            conf[m] = np.clip(d, 0, 1.0)
    return np.clip(conf, 0.0, 1.0)


def regression_calibration_curve(
    raw: np.ndarray,
    rounded: np.ndarray,
    targets: np.ndarray,
    thresholds: list[float],
    n_bins: int = 10,
):
    """Reliability-style curve for regression head, using margin as confidence proxy."""
    conf = regression_margin_confidence(raw, thresholds)
    correct = (rounded == targets).astype(float)
    binned = _binned_calibration(conf, correct, n_bins)
    return binned, conf
