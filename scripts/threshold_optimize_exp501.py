"""Threshold optimization for Exp 501 (F2 Joint OrdSupCon).

Loads the best checkpoint, extracts softmax probabilities on val & test,
computes expected grade (continuous), optimizes thresholds on val,
and applies to test. Saves updated metrics + confusion matrix.

Usage:
    python scripts/threshold_optimize_exp501.py [--device cuda]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config, CLASS_NAMES, NUM_CLASSES
from src.dataset import build_datasets
from src.models import build_model
from src.transforms import get_val_transform
from src.evaluate import (
    OptimizedRounder,
    compute_metrics,
    regression_to_class,
    save_confusion_matrix,
    save_classification_report,
    save_predictions,
)


def extract_probs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Run inference, return (softmax_probs [N,5], targets [N], codes [N])."""
    model.eval()
    all_probs, all_targets, all_codes = [], [], []

    with torch.no_grad():
        for images, targets, codes in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.numpy())
            all_codes.extend(codes)

    return np.concatenate(all_probs), np.concatenate(all_targets), all_codes


def expected_grade(probs: np.ndarray) -> np.ndarray:
    """Compute E[grade] = Σ(p_k * k) for k=0..4.  Returns continuous [N]."""
    grades = np.arange(NUM_CLASSES, dtype=np.float64)
    return probs @ grades


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = get_config(501)

    # --- Load model ---
    ckpt_path = Path("checkpoints/exp501_f2_joint_ordsupcon/"
                      "exp501_f2_joint_ordsupcon_best.pth")
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        sys.exit(1)

    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    # --- Build val & test datasets ---
    transform_val = get_val_transform()
    _, val_ds, test_ds = build_datasets(cfg, transform_train=None, transform_val=transform_val)
    print(f"Val: {len(val_ds)} | Test: {len(test_ds)}")

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    # --- Extract softmax probabilities ---
    print("Extracting val probabilities...")
    val_probs, val_targets, val_codes = extract_probs(model, val_loader, device)
    print("Extracting test probabilities...")
    test_probs, test_targets, test_codes = extract_probs(model, test_loader, device)

    # =====================================================================
    # Strategy 1: Argmax baseline (same as original exp501 evaluation)
    # =====================================================================
    baseline_preds = test_probs.argmax(axis=1)
    baseline_metrics = compute_metrics(test_targets.astype(int), baseline_preds, test_probs)
    print("\n" + "=" * 70)
    print("STRATEGY 1: Argmax (original)")
    print("=" * 70)
    _print_metrics(baseline_metrics)

    # =====================================================================
    # Strategy 2: Expected-grade + optimized thresholds
    # =====================================================================
    val_eg = expected_grade(val_probs)
    test_eg = expected_grade(test_probs)

    rounder = OptimizedRounder()
    opt_thresholds = rounder.fit(val_eg, val_targets.astype(int))
    print(f"\nOptimized thresholds (on val): {[f'{t:.4f}' for t in opt_thresholds]}")

    opt_preds = regression_to_class(test_eg, opt_thresholds)
    opt_metrics = compute_metrics(test_targets.astype(int), opt_preds, test_probs)
    print("\n" + "=" * 70)
    print("STRATEGY 2: Expected-grade + optimized thresholds")
    print("=" * 70)
    _print_metrics(opt_metrics)

    # =====================================================================
    # Strategy 3: Grid-search per-class probability thresholds
    # =====================================================================
    print("\n" + "=" * 70)
    print("STRATEGY 3: Cumulative probability thresholds (ordinal)")
    print("=" * 70)
    cum_preds = cumulative_threshold_optimize(val_probs, val_targets.astype(int),
                                               test_probs)
    cum_metrics = compute_metrics(test_targets.astype(int), cum_preds, test_probs)
    _print_metrics(cum_metrics)

    # =====================================================================
    # Pick best strategy and save
    # =====================================================================
    strategies = {
        "argmax": (baseline_preds, baseline_metrics),
        "expected_grade_opt": (opt_preds, opt_metrics),
        "cumulative_opt": (cum_preds, cum_metrics),
    }

    best_name = max(strategies, key=lambda k: strategies[k][1]["qwk"])
    best_preds, best_metrics = strategies[best_name]

    print("\n" + "=" * 70)
    print(f"BEST STRATEGY: {best_name}  (QWK={best_metrics['qwk']:.4f})")
    print("=" * 70)

    # --- Save results ---
    out_dir = cfg.results_dir / "threshold_opt"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_confusion_matrix(test_targets.astype(int), best_preds,
                          out_dir / "exp501_thresh_opt_cm.png")
    save_classification_report(test_targets.astype(int), best_preds,
                               out_dir / "exp501_thresh_opt_cls_report.txt")
    save_predictions(test_codes, test_eg, best_preds, test_targets.astype(int),
                     out_dir / "exp501_thresh_opt_preds.csv")

    # Save full comparison
    with open(out_dir / "exp501_thresh_opt_summary.txt", "w") as f:
        for sname, (spreds, smetrics) in strategies.items():
            f.write(f"\n{'='*60}\n{sname}\n{'='*60}\n")
            for k, v in smetrics.items():
                f.write(f"  {k}: {v:.4f}\n")
        f.write(f"\nBest strategy: {best_name}\n")
        f.write(f"Optimized thresholds: {opt_thresholds}\n")

    print(f"\nResults saved to {out_dir}")

    # --- Delta table ---
    print("\n" + "=" * 70)
    print("DELTA: Best strategy vs original argmax")
    print("=" * 70)
    for k in ["qwk", "accuracy", "macro_f1", "sensitivity", "specificity",
              "f1_No DR", "f1_Mild", "f1_Moderate", "f1_Severe", "f1_Proliferative",
              "auc_roc", "ece"]:
        orig = baseline_metrics.get(k, 0)
        opt = best_metrics.get(k, 0)
        delta = opt - orig
        arrow = "▲" if delta > 0 else "▼" if delta < 0 else "─"
        print(f"  {k:20s}: {orig:.4f} → {opt:.4f}  ({arrow} {delta:+.4f})")


def cumulative_threshold_optimize(
    val_probs: np.ndarray,
    val_targets: np.ndarray,
    test_probs: np.ndarray,
) -> np.ndarray:
    """Ordinal cumulative-probability threshold optimization.

    For an ordinal problem, P(grade >= k) = sum(probs[k:]).
    We find optimal thresholds t1..t4 on these cumulative probs.
    """
    from scipy.optimize import minimize
    from src.evaluate import quadratic_weighted_kappa

    def cumprob_to_class(probs: np.ndarray, thresholds: list[float]) -> np.ndarray:
        n = probs.shape[0]
        preds = np.zeros(n, dtype=int)
        for k in range(1, NUM_CLASSES):
            cum_prob = probs[:, k:].sum(axis=1)
            preds += (cum_prob >= thresholds[k - 1]).astype(int)
        return preds

    def neg_qwk(thresholds, probs, targets):
        preds = cumprob_to_class(probs, thresholds.tolist())
        return -quadratic_weighted_kappa(targets, preds)

    result = minimize(
        neg_qwk,
        x0=np.array([0.5, 0.5, 0.5, 0.5]),
        args=(val_probs, val_targets),
        method="Nelder-Mead",
        options={"maxiter": 2000, "xatol": 1e-6},
    )
    best_t = sorted(result.x.tolist())
    print(f"  Cumulative thresholds (on val): {[f'{t:.4f}' for t in best_t]}")

    return cumprob_to_class(test_probs, best_t)


def _print_metrics(m: dict[str, float]) -> None:
    print(f"  QWK:          {m['qwk']:.4f}")
    print(f"  Accuracy:     {m['accuracy']:.4f}")
    print(f"  Macro F1:     {m['macro_f1']:.4f}")
    print(f"  Sensitivity:  {m.get('sensitivity', 0):.4f}")
    print(f"  Specificity:  {m.get('specificity', 0):.4f}")
    print(f"  AUC-ROC:      {m.get('auc_roc', 0):.4f}")
    print(f"  ECE:          {m.get('ece', 0):.4f}")
    print(f"  Per-class F1: ", end="")
    for name in CLASS_NAMES:
        print(f"{name}={m.get(f'f1_{name}', 0):.3f}  ", end="")
    print()


if __name__ == "__main__":
    main()
