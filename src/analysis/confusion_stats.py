"""Statistical tools for confusion-matrix / error analysis.

All functions operate on 1-D integer-class arrays (``y_true`` / ``y_pred``)
so that they work for both single-model and ensemble predictions.
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for a metric. Returns (point, lo, hi)."""
    if rng is None:
        rng = np.random.default_rng(42)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    point = float(metric_fn(y_true, y_pred))
    stats_arr = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats_arr[i] = metric_fn(y_true[idx], y_pred[idx])
    lo = float(np.quantile(stats_arr, alpha / 2))
    hi = float(np.quantile(stats_arr, 1 - alpha / 2))
    return point, lo, hi


def paired_bootstrap_diff(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Paired bootstrap on (A - B). Returns delta, CI, and two-sided p-value."""
    if rng is None:
        rng = np.random.default_rng(42)
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)
    n = len(y_true)
    delta = float(metric_fn(y_true, y_pred_a) - metric_fn(y_true, y_pred_b))
    diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = metric_fn(y_true[idx], y_pred_a[idx]) - metric_fn(y_true[idx], y_pred_b[idx])
    lo = float(np.quantile(diffs, alpha / 2))
    hi = float(np.quantile(diffs, 1 - alpha / 2))
    # two-sided p via fraction of bootstrap diffs with opposite sign of point delta
    if delta >= 0:
        p = 2 * float((diffs <= 0).mean())
    else:
        p = 2 * float((diffs >= 0).mean())
    p = min(p, 1.0)
    return {"delta": delta, "ci_lo": lo, "ci_hi": hi, "p_value": p}


# ---------------------------------------------------------------------------
# Confusion-matrix decomposition with CIs
# ---------------------------------------------------------------------------


def confusion_with_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 5,
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Return raw, row-normalized, col-normalized CMs plus per-cell bootstrap CIs.

    The per-cell CIs are for the row-normalized (recall) version.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(np.float64)
    row_sum = cm.sum(axis=1, keepdims=True)
    col_sum = cm.sum(axis=0, keepdims=True)
    cm_row = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum > 0)
    cm_col = np.divide(cm, col_sum, out=np.zeros_like(cm), where=col_sum > 0)

    boot = np.zeros((n_boot, num_classes, num_classes), dtype=np.float64)
    n = len(y_true)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m = confusion_matrix(y_true[idx], y_pred[idx], labels=labels).astype(np.float64)
        rs = m.sum(axis=1, keepdims=True)
        boot[i] = np.divide(m, rs, out=np.zeros_like(m), where=rs > 0)
    lo = np.quantile(boot, alpha / 2, axis=0)
    hi = np.quantile(boot, 1 - alpha / 2, axis=0)
    return {
        "cm_raw": cm.astype(np.int64),
        "cm_row": cm_row,
        "cm_col": cm_col,
        "cm_row_ci_lo": lo,
        "cm_row_ci_hi": hi,
    }


# ---------------------------------------------------------------------------
# Adjacent-vs-non-adjacent error analysis
# ---------------------------------------------------------------------------


def adjacent_error_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 5,
) -> dict[str, float | np.ndarray]:
    """Fraction of errors with |pred - true| = 1 (adjacent) vs >= 2."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    err_mask = y_true != y_pred
    if err_mask.sum() == 0:
        return {
            "adjacent_frac": 1.0,
            "nonadjacent_frac": 0.0,
            "per_class_adjacent": np.ones(num_classes),
        }
    diff = np.abs(y_true[err_mask] - y_pred[err_mask])
    adj = float((diff == 1).mean())
    non = float((diff >= 2).mean())

    per_class = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        class_err = err_mask & (y_true == c)
        if class_err.sum() == 0:
            per_class[c] = np.nan
            continue
        d = np.abs(y_true[class_err] - y_pred[class_err])
        per_class[c] = float((d == 1).mean())
    return {
        "adjacent_frac": adj,
        "nonadjacent_frac": non,
        "per_class_adjacent": per_class,
    }


def off_by_n_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 5,
) -> np.ndarray:
    """Per-class distribution of (pred - true). Returns (num_classes, 2*num_classes - 1)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    offsets = np.arange(-(num_classes - 1), num_classes)
    out = np.zeros((num_classes, len(offsets)), dtype=np.float64)
    for c in range(num_classes):
        mask = y_true == c
        if mask.sum() == 0:
            continue
        diffs = y_pred[mask] - y_true[mask]
        for j, k in enumerate(offsets):
            out[c, j] = float((diffs == k).mean())
    return out


# ---------------------------------------------------------------------------
# Ordinal vs nominal kappa decomposition
# ---------------------------------------------------------------------------


def kappa_split(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return unweighted, linear, and quadratic kappa.

    QWK - nominal κ quantifies the "forgivable" ordinal component of agreement.
    """
    nom = float(cohen_kappa_score(y_true, y_pred))
    lin = float(cohen_kappa_score(y_true, y_pred, weights="linear"))
    qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    return {
        "nominal_kappa": nom,
        "linear_kappa": lin,
        "quadratic_kappa": qwk,
        "ordinal_gap": qwk - nom,
    }


# ---------------------------------------------------------------------------
# McNemar test for paired binary (correct / incorrect) outcomes
# ---------------------------------------------------------------------------


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    sample_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Exact McNemar test on A-correct vs B-correct.

    ``sample_mask`` can restrict to a subset (e.g., samples where true class is 2 or 3).
    """
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)
    if sample_mask is not None:
        y_true = y_true[sample_mask]
        y_pred_a = y_pred_a[sample_mask]
        y_pred_b = y_pred_b[sample_mask]
    a_correct = y_pred_a == y_true
    b_correct = y_pred_b == y_true
    b01 = int(np.sum(a_correct & ~b_correct))  # A right, B wrong
    b10 = int(np.sum(~a_correct & b_correct))  # A wrong, B right
    n = b01 + b10
    if n == 0:
        return {"b01": b01, "b10": b10, "n_discordant": 0, "p_value": 1.0}
    # exact binomial: P(X <= min(b01, b10)) * 2, clipped
    k = min(b01, b10)
    p = 2 * stats.binom.cdf(k, n, 0.5)
    p = float(min(p, 1.0))
    return {"b01": b01, "b10": b10, "n_discordant": n, "p_value": p}


def mcnemar_per_pair(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    pairs: Iterable[tuple[int, int]] = ((0, 1), (1, 2), (2, 3), (3, 4)),
) -> dict[tuple[int, int], dict[str, float]]:
    """McNemar on samples restricted to each (true, pred) near-boundary pair."""
    y_true = np.asarray(y_true)
    out: dict[tuple[int, int], dict[str, float]] = {}
    for (i, j) in pairs:
        mask = (y_true == i) | (y_true == j)
        out[(i, j)] = mcnemar_test(y_true, y_pred_a, y_pred_b, sample_mask=mask)
    return out


# ---------------------------------------------------------------------------
# Per-class metric helpers (for bootstrap CI consumers)
# ---------------------------------------------------------------------------


def metric_qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def metric_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def metric_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def metric_per_class_f1(c: int) -> Callable[[np.ndarray, np.ndarray], float]:
    """Return a metric function that computes F1 for a single class."""
    def _fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        scores = f1_score(
            y_true, y_pred,
            labels=list(range(5)),
            average=None,
            zero_division=0,
        )
        return float(scores[c]) if c < len(scores) else 0.0
    return _fn


# ---------------------------------------------------------------------------
# Feature-space analyses (prototype distances, JS-divergence)
# ---------------------------------------------------------------------------


def class_prototypes(features: np.ndarray, labels: np.ndarray, num_classes: int = 5) -> np.ndarray:
    """Mean feature vector per class. NaN row for absent classes."""
    dim = features.shape[1]
    protos = np.full((num_classes, dim), np.nan, dtype=np.float64)
    for c in range(num_classes):
        m = labels == c
        if m.any():
            protos[c] = features[m].mean(axis=0)
    return protos


def prototype_distance_delta(
    features: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 5,
) -> np.ndarray:
    """For misclassified samples, d(x, proto_true) - d(x, proto_pred).

    Positive ⇒ sample sits closer to the pred prototype than to the true one
    (confusion is at the representation level). Negative ⇒ representation is
    on the true side but decision boundary failed it. Returns one value per
    sample (NaN for correctly classified).
    """
    protos = class_prototypes(features, y_true, num_classes)
    out = np.full(len(features), np.nan, dtype=np.float64)
    miscls = y_true != y_pred
    for idx in np.where(miscls)[0]:
        pt = protos[y_true[idx]]
        pp = protos[y_pred[idx]]
        if np.isnan(pt).any() or np.isnan(pp).any():
            continue
        d_true = np.linalg.norm(features[idx] - pt)
        d_pred = np.linalg.norm(features[idx] - pp)
        out[idx] = float(d_true - d_pred)
    return out


def _hist_density(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    h, _ = np.histogram(x, bins=edges)
    h = h.astype(np.float64) + 1e-12
    return h / h.sum()


def js_divergence_1d(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen–Shannon divergence between two discrete distributions."""
    m = 0.5 * (p + q)
    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(a * np.log(a / b)))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def class_embedding_js_matrix(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 5,
    n_bins: int = 30,
    n_projections: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Pairwise JS-divergence of per-class feature distributions.

    For high-dim features we project onto a random 1-D direction and compare
    the resulting marginals (optionally averaged over ``n_projections`` draws,
    which reduces variance at the cost of compute). Interpretable proxy for
    how distinguishable two classes are in feature space.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    dim = features.shape[1]
    out = np.zeros((num_classes, num_classes), dtype=np.float64)
    for _ in range(n_projections):
        w = rng.normal(size=dim)
        w /= np.linalg.norm(w) + 1e-12
        proj = features @ w
        lo, hi = float(proj.min()), float(proj.max())
        edges = np.linspace(lo, hi, n_bins + 1)
        densities = {}
        for c in range(num_classes):
            m = labels == c
            if m.any():
                densities[c] = _hist_density(proj[m], edges)
        for i in range(num_classes):
            for j in range(num_classes):
                if i == j or i not in densities or j not in densities:
                    continue
                out[i, j] += js_divergence_1d(densities[i], densities[j])
    out /= n_projections
    return out
