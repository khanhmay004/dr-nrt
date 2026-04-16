"""Threshold optimization for any experiment checkpoint.

Usage:
    python scripts/threshold_optimize.py --exp 300 --device cuda
    python scripts/threshold_optimize.py --exp 300 --ckpt path/to/best.pth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

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


def extract_probs(model, loader, device):
    model.eval()
    all_probs, all_targets, all_codes = [], [], []
    with torch.no_grad():
        for images, targets, codes in loader:
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.numpy())
            all_codes.extend(codes)
    return np.concatenate(all_probs), np.concatenate(all_targets), all_codes


def expected_grade(probs):
    return probs @ np.arange(NUM_CLASSES, dtype=np.float64)


def cumulative_threshold_optimize(val_probs, val_targets, test_probs):
    from scipy.optimize import minimize
    from src.evaluate import quadratic_weighted_kappa

    def cumprob_to_class(probs, thresholds):
        preds = np.zeros(probs.shape[0], dtype=int)
        for k in range(1, NUM_CLASSES):
            preds += (probs[:, k:].sum(axis=1) >= thresholds[k - 1]).astype(int)
        return preds

    def neg_qwk(thresholds, probs, targets):
        return -quadratic_weighted_kappa(targets, cumprob_to_class(probs, thresholds.tolist()))

    result = minimize(neg_qwk, x0=np.array([0.5, 0.5, 0.5, 0.5]),
                      args=(val_probs, val_targets), method="Nelder-Mead",
                      options={"maxiter": 2000, "xatol": 1e-6})
    best_t = sorted(result.x.tolist())
    print(f"  Cumulative thresholds (on val): {[f'{t:.4f}' for t in best_t]}")
    return cumprob_to_class(test_probs, best_t)


def _print_metrics(m):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ckpt", type=str, default="")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = get_config(args.exp)

    ckpt_path = Path(args.ckpt) if args.ckpt else cfg.ckpt_dir / f"{cfg.exp_name}_best.pth"
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        sys.exit(1)

    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location=device))
    model.eval()
    print(f"Loaded: {ckpt_path}")

    transform_val = get_val_transform()
    _, val_ds, test_ds = build_datasets(cfg, transform_train=None, transform_val=transform_val)
    print(f"Val: {len(val_ds)} | Test: {len(test_ds)}")

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    val_probs, val_targets, _ = extract_probs(model, val_loader, device)
    test_probs, test_targets, test_codes = extract_probs(model, test_loader, device)

    # Strategy 1: Argmax
    baseline_preds = test_probs.argmax(axis=1)
    baseline_metrics = compute_metrics(test_targets.astype(int), baseline_preds, test_probs)
    print(f"\n{'='*60}\nSTRATEGY 1: Argmax (original)\n{'='*60}")
    _print_metrics(baseline_metrics)

    # Strategy 2: Expected-grade + optimized thresholds
    val_eg, test_eg = expected_grade(val_probs), expected_grade(test_probs)
    rounder = OptimizedRounder()
    opt_thresholds = rounder.fit(val_eg, val_targets.astype(int))
    print(f"\nOptimized thresholds (on val): {[f'{t:.4f}' for t in opt_thresholds]}")
    opt_preds = regression_to_class(test_eg, opt_thresholds)
    opt_metrics = compute_metrics(test_targets.astype(int), opt_preds, test_probs)
    print(f"\n{'='*60}\nSTRATEGY 2: Expected-grade + optimized thresholds\n{'='*60}")
    _print_metrics(opt_metrics)

    # Strategy 3: Cumulative probability
    print(f"\n{'='*60}\nSTRATEGY 3: Cumulative probability thresholds\n{'='*60}")
    cum_preds = cumulative_threshold_optimize(val_probs, val_targets.astype(int), test_probs)
    cum_metrics = compute_metrics(test_targets.astype(int), cum_preds, test_probs)
    _print_metrics(cum_metrics)

    # Pick best
    strategies = {
        "argmax": (baseline_preds, baseline_metrics),
        "expected_grade_opt": (opt_preds, opt_metrics),
        "cumulative_opt": (cum_preds, cum_metrics),
    }
    best_name = max(strategies, key=lambda k: strategies[k][1]["qwk"])
    best_preds, best_metrics = strategies[best_name]

    print(f"\n{'='*60}\nBEST: {best_name}  (QWK={best_metrics['qwk']:.4f})\n{'='*60}")

    # Save
    out_dir = cfg.results_dir / "threshold_opt"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_confusion_matrix(test_targets.astype(int), best_preds,
                          out_dir / f"{cfg.exp_name}_thresh_opt_cm.png")
    save_classification_report(test_targets.astype(int), best_preds,
                               out_dir / f"{cfg.exp_name}_thresh_opt_cls_report.txt")
    save_predictions(test_codes, test_eg, best_preds, test_targets.astype(int),
                     out_dir / f"{cfg.exp_name}_thresh_opt_preds.csv")

    with open(out_dir / f"{cfg.exp_name}_thresh_opt_summary.txt", "w") as f:
        for sname, (_, smetrics) in strategies.items():
            f.write(f"\n{'='*60}\n{sname}\n{'='*60}\n")
            for k, v in smetrics.items():
                f.write(f"  {k}: {v:.4f}\n")
        f.write(f"\nBest strategy: {best_name}\n")
        f.write(f"Optimized thresholds: {opt_thresholds}\n")

    # Delta table
    print(f"\n{'='*60}\nDELTA: Best vs argmax\n{'='*60}")
    for k in ["qwk", "accuracy", "macro_f1", "sensitivity", "specificity",
              "f1_No DR", "f1_Mild", "f1_Moderate", "f1_Severe", "f1_Proliferative",
              "auc_roc", "ece"]:
        orig = baseline_metrics.get(k, 0)
        opt = best_metrics.get(k, 0)
        delta = opt - orig
        arrow = "▲" if delta > 0 else "▼" if delta < 0 else "─"
        print(f"  {k:20s}: {orig:.4f} → {opt:.4f}  ({arrow} {delta:+.4f})")

    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
