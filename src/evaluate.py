from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
)

from src.config import CLASS_NAMES, NUM_CLASSES


def compute_ece(
    probs: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error with equal-width confidence bins."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == targets).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return float(ece / len(targets))


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def regression_to_class(
    preds: np.ndarray,
    thresholds: list[float] | None = None,
) -> np.ndarray:
    if thresholds is None:
        thresholds = [0.5, 1.5, 2.5, 3.5]
    result = np.zeros_like(preds, dtype=np.int64)
    for i, t in enumerate(thresholds):
        result += (preds >= t).astype(np.int64)
    return result


class OptimizedRounder:
    def __init__(self) -> None:
        self.thresholds: list[float] = [0.5, 1.5, 2.5, 3.5]

    def _qwk_loss(self, thresholds: np.ndarray, preds: np.ndarray, targets: np.ndarray) -> float:
        classes = regression_to_class(preds, thresholds.tolist())
        return -quadratic_weighted_kappa(targets, classes)

    def fit(self, preds: np.ndarray, targets: np.ndarray) -> list[float]:
        result = minimize(
            self._qwk_loss,
            x0=np.array([0.5, 1.5, 2.5, 3.5]),
            args=(preds, targets),
            method="Nelder-Mead",
            options={"maxiter": 1000, "xatol": 1e-6},
        )
        self.thresholds = sorted(result.x.tolist())
        return self.thresholds

    def predict(self, preds: np.ndarray) -> np.ndarray:
        return regression_to_class(preds, self.thresholds)


def compute_metrics(
    y_true: np.ndarray,
    y_pred_classes: np.ndarray,
    y_pred_probs: np.ndarray | None = None,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    metrics["qwk"] = quadratic_weighted_kappa(y_true, y_pred_classes)
    metrics["accuracy"] = accuracy_score(y_true, y_pred_classes)
    metrics["macro_f1"] = f1_score(y_true, y_pred_classes, average="macro", zero_division=0)
    metrics["sensitivity"] = recall_score(y_true, y_pred_classes, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred_classes, labels=list(range(NUM_CLASSES)))
    specificity_per_class = []
    for c in range(NUM_CLASSES):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp
        specificity_per_class.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    metrics["specificity"] = float(np.mean(specificity_per_class))

    per_class_f1 = f1_score(y_true, y_pred_classes, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
    for i, name in enumerate(CLASS_NAMES):
        metrics[f"f1_{name}"] = float(per_class_f1[i]) if i < len(per_class_f1) else 0.0

    if y_pred_probs is not None and y_pred_probs.ndim == 2:
        try:
            metrics["auc_roc"] = roc_auc_score(
                y_true, y_pred_probs, multi_class="ovr", average="macro",
            )
        except ValueError:
            metrics["auc_roc"] = 0.0
        metrics["ece"] = compute_ece(y_pred_probs, y_true)

    return metrics


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(NUM_CLASSES),
        yticks=range(NUM_CLASSES),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ylabel="True",
        xlabel="Predicted",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
) -> None:
    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    save_path.write_text(report, encoding="utf-8")


def save_training_curves(
    log_df: pd.DataFrame,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(log_df["epoch"], log_df["train_loss"], label="Train Loss")
    axes[0].plot(log_df["epoch"], log_df["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(log_df["epoch"], log_df["val_qwk"], label="Val QWK")
    axes[1].plot(log_df["epoch"], log_df["val_macro_f1"], label="Val Macro F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation Metrics")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def save_predictions(
    codes: list[str],
    raw_preds: np.ndarray,
    rounded_preds: np.ndarray,
    true_labels: np.ndarray,
    save_path: Path,
) -> None:
    df = pd.DataFrame({
        "id_code": codes,
        "raw_prediction": raw_preds,
        "rounded_prediction": rounded_preds.astype(int),
        "true_label": true_labels.astype(int),
    })
    df.to_csv(save_path, index=False)
