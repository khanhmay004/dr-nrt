"""MC Dropout uncertainty evaluation for any experiment checkpoint.

Loads best checkpoint, runs T stochastic forward passes with dropout enabled,
computes predictive entropy, and evaluates uncertainty quality.

Usage:
    python scripts/mc_dropout_eval.py --exp 300 --T 20 --device cuda
    python scripts/mc_dropout_eval.py --exp 100 --T 20 --ckpt checkpoints/exp08_gem/exp08_gem_best.pth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config, CLASS_NAMES, NUM_CLASSES
from src.dataset import build_datasets
from src.models import build_model
from src.transforms import get_val_transform
from src.evaluate import (
    compute_metrics,
    save_confusion_matrix,
    save_classification_report,
    save_predictions,
)


def mc_forward(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    T: int = 20,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Run T stochastic forward passes with dropout enabled.

    Returns:
        mc_probs: [N, T, K] softmax probabilities from each pass
        targets:  [N] ground truth labels
        codes:    [N] image IDs
    """
    model.train()  # Enable dropout at inference

    all_probs, all_targets, all_codes = [], [], []

    for images, targets, codes in loader:
        images = images.to(device)
        batch_probs = []

        for t in range(T):
            with torch.no_grad():
                logits = model(images)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                batch_probs.append(probs)

        # Stack: [T, B, K] -> [B, T, K]
        batch_probs = np.stack(batch_probs, axis=0).transpose(1, 0, 2)
        all_probs.append(batch_probs)
        all_targets.append(targets.numpy())
        all_codes.extend(codes)

    return np.concatenate(all_probs), np.concatenate(all_targets), all_codes


def predictive_entropy(mc_probs: np.ndarray) -> np.ndarray:
    """H[y|x] = -Σ p̄_c log p̄_c, where p̄ = mean over T passes.  [N]"""
    mean_probs = mc_probs.mean(axis=1)  # [N, K]
    # Clamp to avoid log(0)
    mean_probs = np.clip(mean_probs, 1e-10, 1.0)
    return -(mean_probs * np.log(mean_probs)).sum(axis=1)


def mutual_information(mc_probs: np.ndarray) -> np.ndarray:
    """Epistemic uncertainty = H[y|x] - E_θ[H[y|x,θ]]  (BALD).  [N]"""
    # Total entropy
    pe = predictive_entropy(mc_probs)

    # Expected entropy of individual passes
    mc_probs_clamp = np.clip(mc_probs, 1e-10, 1.0)
    per_pass_entropy = -(mc_probs_clamp * np.log(mc_probs_clamp)).sum(axis=2)  # [N, T]
    expected_entropy = per_pass_entropy.mean(axis=1)  # [N]

    return pe - expected_entropy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, required=True, help="Experiment ID")
    parser.add_argument("--T", type=int, default=20, help="Number of MC forward passes")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ckpt", type=str, default="", help="Override checkpoint path")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = get_config(args.exp)

    # --- Load model ---
    ckpt_path = Path(args.ckpt) if args.ckpt else cfg.ckpt_dir / f"{cfg.exp_name}_best.pth"
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        sys.exit(1)

    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location=device))
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Dropout: {cfg.head_dropout}")

    if cfg.head_dropout == 0.0:
        print("WARNING: This model has no dropout (head_dropout=0.0). "
              "MC Dropout will have no stochasticity. Results will be deterministic.")

    # --- Datasets ---
    transform_val = get_val_transform()
    _, val_ds, test_ds = build_datasets(cfg, transform_train=None, transform_val=transform_val)
    print(f"Test set: {len(test_ds)} samples")

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    # --- MC Dropout inference ---
    print(f"\nRunning {args.T} stochastic forward passes...")
    mc_probs, targets, codes = mc_forward(model, test_loader, device, T=args.T)
    print(f"MC probs shape: {mc_probs.shape}")  # [N, T, K]

    # --- Compute uncertainty metrics ---
    mean_probs = mc_probs.mean(axis=1)  # [N, K]
    mc_preds = mean_probs.argmax(axis=1)  # [N]
    entropy = predictive_entropy(mc_probs)
    mi = mutual_information(mc_probs)

    targets_int = targets.astype(int)
    is_correct = (mc_preds == targets_int)
    is_wrong = ~is_correct

    # --- Standard metrics (MC averaged predictions) ---
    metrics = compute_metrics(targets_int, mc_preds, mean_probs)
    print("\n" + "=" * 60)
    print(f"MC Dropout Results (T={args.T})")
    print("=" * 60)
    print(f"  QWK:          {metrics['qwk']:.4f}")
    print(f"  Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  Macro F1:     {metrics['macro_f1']:.4f}")
    print(f"  AUC-ROC:      {metrics.get('auc_roc', 0):.4f}")
    print(f"  ECE:          {metrics.get('ece', 0):.4f}")
    for name in CLASS_NAMES:
        print(f"  F1 {name}: {metrics.get(f'f1_{name}', 0):.4f}")

    # --- Single-pass comparison ---
    model.eval()
    with torch.no_grad():
        single_probs_list, single_targets_list = [], []
        for images, tgt, _ in test_loader:
            logits = model(images.to(device))
            single_probs_list.append(torch.softmax(logits, dim=1).cpu().numpy())
            single_targets_list.append(tgt.numpy())
    single_probs = np.concatenate(single_probs_list)
    single_preds = single_probs.argmax(axis=1)
    single_targets = np.concatenate(single_targets_list)
    single_metrics = compute_metrics(single_targets.astype(int), single_preds, single_probs)
    print(f"\n  (Single-pass baseline: QWK={single_metrics['qwk']:.4f}, "
          f"Acc={single_metrics['accuracy']:.4f}, F1={single_metrics['macro_f1']:.4f})")

    # --- Uncertainty quality metrics ---
    print("\n" + "=" * 60)
    print("Uncertainty Quality Metrics")
    print("=" * 60)

    # 1. AUROC(entropy, error): does high entropy predict mistakes?
    if is_wrong.sum() > 0 and is_correct.sum() > 0:
        auroc_entropy = roc_auc_score(is_wrong.astype(int), entropy)
        print(f"  AUROC(entropy → error):  {auroc_entropy:.4f}")
        auroc_mi = roc_auc_score(is_wrong.astype(int), mi)
        print(f"  AUROC(MI → error):       {auroc_mi:.4f}")
    else:
        auroc_entropy = 0.0
        print("  AUROC: N/A (all correct or all wrong)")

    # 2. Per-class mean entropy
    print(f"\n  Per-class mean entropy:")
    for c, name in enumerate(CLASS_NAMES):
        mask = targets_int == c
        if mask.sum() > 0:
            mean_e = entropy[mask].mean()
            std_e = entropy[mask].std()
            print(f"    {name:15s}: {mean_e:.4f} ± {std_e:.4f}  (n={mask.sum()})")

    # 3. Mean entropy for correct vs wrong predictions
    print(f"\n  Entropy statistics:")
    print(f"    Correct predictions: {entropy[is_correct].mean():.4f} ± {entropy[is_correct].std():.4f} (n={is_correct.sum()})")
    if is_wrong.sum() > 0:
        print(f"    Wrong predictions:   {entropy[is_wrong].mean():.4f} ± {entropy[is_wrong].std():.4f} (n={is_wrong.sum()})")

    # 4. Coverage-accuracy curve: remove high-entropy samples, measure accuracy
    print(f"\n  Coverage-Accuracy Curve:")
    print(f"    {'Coverage':>10s}  {'Accuracy':>10s}  {'QWK':>10s}  {'Samples':>10s}")
    sorted_idx = np.argsort(entropy)  # lowest entropy first
    coverages = [1.0, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50]
    coverage_data = []
    for cov in coverages:
        n_keep = int(len(targets_int) * cov)
        if n_keep < 10:
            continue
        keep_idx = sorted_idx[:n_keep]
        sub_preds = mc_preds[keep_idx]
        sub_targets = targets_int[keep_idx]
        acc = (sub_preds == sub_targets).mean()
        if len(np.unique(sub_targets)) >= 2:
            qwk_val = compute_metrics(sub_targets, sub_preds)["qwk"]
        else:
            qwk_val = float("nan")
        print(f"    {cov:10.0%}  {acc:10.4f}  {qwk_val:10.4f}  {n_keep:10d}")
        coverage_data.append((cov, acc, qwk_val, n_keep))

    # --- Save outputs ---
    out_dir = cfg.results_dir / "mc_dropout"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_confusion_matrix(targets_int, mc_preds, out_dir / "mc_dropout_cm.png")
    save_classification_report(targets_int, mc_preds, out_dir / "mc_dropout_cls_report.txt")

    # Save per-sample predictions + uncertainty
    import pandas as pd
    df = pd.DataFrame({
        "id_code": codes,
        "true_label": targets_int,
        "mc_prediction": mc_preds,
        "single_prediction": single_preds,
        "entropy": np.round(entropy, 6),
        "mutual_info": np.round(mi, 6),
        "correct": is_correct.astype(int),
    })
    # Add per-class mean probs
    for c, name in enumerate(CLASS_NAMES):
        df[f"p_{name}"] = np.round(mean_probs[:, c], 6)
    df.to_csv(out_dir / "mc_dropout_predictions.csv", index=False)

    # --- Plots ---
    _plot_entropy_histogram(entropy, is_correct, out_dir / "entropy_histogram.png")
    _plot_entropy_by_class(entropy, targets_int, out_dir / "entropy_by_class.png")
    _plot_coverage_accuracy(coverage_data, out_dir / "coverage_accuracy.png")

    # Save summary
    with open(out_dir / "mc_dropout_summary.txt", "w") as f:
        f.write(f"Experiment: {cfg.exp_name}\n")
        f.write(f"Checkpoint: {ckpt_path}\n")
        f.write(f"MC passes (T): {args.T}\n")
        f.write(f"Dropout rate: {cfg.head_dropout}\n\n")
        f.write("=== MC Averaged Metrics ===\n")
        for k, v in metrics.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write(f"\n=== Single-Pass Metrics ===\n")
        for k, v in single_metrics.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write(f"\n=== Uncertainty Quality ===\n")
        f.write(f"  AUROC(entropy → error): {auroc_entropy:.4f}\n")
        f.write(f"  Mean entropy (correct): {entropy[is_correct].mean():.4f}\n")
        if is_wrong.sum() > 0:
            f.write(f"  Mean entropy (wrong):   {entropy[is_wrong].mean():.4f}\n")

    print(f"\nAll outputs saved to {out_dir}")


def _plot_entropy_histogram(entropy, is_correct, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(entropy[is_correct], bins=30, alpha=0.6, label="Correct", color="#2ecc71", density=True)
    ax.hist(entropy[~is_correct], bins=30, alpha=0.6, label="Wrong", color="#e74c3c", density=True)
    ax.set_xlabel("Predictive Entropy")
    ax.set_ylabel("Density")
    ax.set_title("Entropy Distribution: Correct vs Wrong Predictions")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_entropy_by_class(entropy, targets, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    data = [entropy[targets == c] for c in range(NUM_CLASSES)]
    bp = ax.boxplot(data, labels=CLASS_NAMES, patch_artist=True)
    colors = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c", "#9b59b6"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xlabel("DR Grade")
    ax.set_ylabel("Predictive Entropy")
    ax.set_title("Uncertainty by DR Severity Grade")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_coverage_accuracy(coverage_data, save_path):
    if not coverage_data:
        return
    covs, accs, qwks, _ = zip(*coverage_data)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(covs, accs, "o-", color="#2ecc71", linewidth=2, label="Accuracy")
    ax1.set_xlabel("Coverage (fraction of samples retained)")
    ax1.set_ylabel("Accuracy", color="#2ecc71")
    ax1.tick_params(axis="y", labelcolor="#2ecc71")

    ax2 = ax1.twinx()
    qwks_clean = [q if not np.isnan(q) else 0 for q in qwks]
    ax2.plot(covs, qwks_clean, "s--", color="#3498db", linewidth=2, label="QWK")
    ax2.set_ylabel("QWK", color="#3498db")
    ax2.tick_params(axis="y", labelcolor="#3498db")

    ax1.set_title("Coverage-Accuracy Curve (lower coverage = more confident)")
    ax1.invert_xaxis()
    ax1.grid(True, alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
