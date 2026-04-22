"""MC-Dropout uncertainty for an ensemble of classification members.

For each member, runs T stochastic forward passes with dropout enabled on the
APTOS test set, then concatenates all M*T softmax samples into one bag per
image.  The bag mean is the ensemble prediction; predictive entropy and BALD
mutual information are computed over the full bag, so both sources of
uncertainty (model disagreement + dropout disagreement) feed the AUROC.

Outputs mirror the per-model `scripts/mc_dropout_eval.py` layout under
    results/ensemble_<tag>/mc_dropout/

Usage:
    python scripts/mc_dropout_ensemble.py --exps 300 701 --T 20 --device cuda
    python scripts/mc_dropout_ensemble.py --exps 300 701 --T 20 --batch-size 16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Repo root + scripts dir on sys.path so we can reuse helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import get_config, CLASS_NAMES, NUM_CLASSES
from src.dataset import build_datasets
from src.models import build_model
from src.transforms import get_val_transform
from src.evaluate import (
    compute_metrics,
    save_confusion_matrix,
    save_classification_report,
)
from mc_dropout_eval import (
    mc_forward,
    predictive_entropy,
    mutual_information,
    _plot_entropy_histogram,
    _plot_entropy_by_class,
    _plot_coverage_accuracy,
)


def _find_checkpoint(cfg) -> Path:
    """Prefer best_composite, then best."""
    candidates = [
        cfg.ckpt_dir / f"{cfg.exp_name}_best_composite.pth",
        cfg.ckpt_dir / f"{cfg.exp_name}_best.pth",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"No checkpoint for {cfg.exp_name} in {cfg.ckpt_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exps", type=int, nargs="+", required=True,
                        help="Experiment IDs to ensemble, e.g. 300 701")
    parser.add_argument("--T", type=int, default=20, help="MC passes per member")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    configs = [get_config(eid) for eid in args.exps]
    print(f"Device  : {device}")
    print(f"Members : {[c.exp_name for c in configs]}")
    print(f"T/member: {args.T}   (total MC samples per image = {args.T * len(configs)})")

    # All APTOS members share the same val/test split.
    base = configs[0]
    transform_val = get_val_transform()
    _, _, test_ds = build_datasets(base, transform_train=None,
                                    transform_val=transform_val)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)
    print(f"Test set: {len(test_ds)} samples\n")

    # --- Run MC for each member ---
    member_mc = []                     # list of [N, T, K]
    targets_ref, codes_ref = None, None

    for cfg in configs:
        if cfg.head_dropout == 0.0:
            print(f"  WARN  {cfg.exp_name}: head_dropout=0 — MC pass is deterministic.")
        ckpt = _find_checkpoint(cfg)
        model = build_model(cfg).to(device)
        model.load_state_dict(torch.load(ckpt, weights_only=True, map_location=device))
        print(f"[{cfg.exp_name}] Loaded {ckpt.name}, dropout={cfg.head_dropout}")
        print(f"  Running T={args.T} stochastic forward passes...")
        mc_probs, targets, codes = mc_forward(model, test_loader, device, T=args.T)
        print(f"  → mc_probs shape {mc_probs.shape}")
        member_mc.append(mc_probs)

        if targets_ref is None:
            targets_ref, codes_ref = targets, codes
        else:
            assert np.array_equal(targets_ref, targets), \
                f"targets mismatch: {cfg.exp_name} disagrees with first member"

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # --- Concatenate into [N, M*T, K] sample bag ---
    ens_mc = np.concatenate(member_mc, axis=1)
    print(f"\nEnsemble MC bag: {ens_mc.shape}  "
          f"({len(configs)} members × {args.T} passes)")

    mean_probs = ens_mc.mean(axis=1)             # [N, K]
    ens_preds  = mean_probs.argmax(axis=1)
    entropy    = predictive_entropy(ens_mc)
    mi         = mutual_information(ens_mc)

    targets_int = targets_ref.astype(int)
    is_correct  = (ens_preds == targets_int)
    is_wrong    = ~is_correct

    metrics = compute_metrics(targets_int, ens_preds, mean_probs)
    print("\n" + "=" * 60)
    print("Ensemble MC Dropout Results")
    print("=" * 60)
    for k in ("qwk", "accuracy", "macro_f1", "auc_roc", "ece"):
        print(f"  {k:14s}: {metrics.get(k, 0):.4f}")
    for name in CLASS_NAMES:
        print(f"  F1 {name:14s}: {metrics.get(f'f1_{name}', 0):.4f}")

    # --- Uncertainty quality ---
    if is_wrong.sum() and is_correct.sum():
        auroc_entropy = roc_auc_score(is_wrong.astype(int), entropy)
        auroc_mi      = roc_auc_score(is_wrong.astype(int), mi)
    else:
        auroc_entropy = auroc_mi = float("nan")

    print(f"\n  AUROC(entropy → error): {auroc_entropy:.4f}")
    print(f"  AUROC(MI → error):      {auroc_mi:.4f}")
    print(f"  Mean entropy (correct): {entropy[is_correct].mean():.4f}")
    if is_wrong.sum():
        print(f"  Mean entropy (wrong):   {entropy[is_wrong].mean():.4f}")

    # Per-class entropy
    print(f"\n  Per-class mean entropy:")
    for c, name in enumerate(CLASS_NAMES):
        mask = targets_int == c
        if mask.sum():
            print(f"    {name:15s}: {entropy[mask].mean():.4f} ± "
                  f"{entropy[mask].std():.4f}  (n={mask.sum()})")

    # Coverage curve
    print(f"\n  Coverage-Accuracy:")
    print(f"    {'Coverage':>10s}  {'Accuracy':>10s}  {'QWK':>10s}  {'Samples':>10s}")
    sorted_idx = np.argsort(entropy)
    coverage_data = []
    for cov in [1.0, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50]:
        n_keep = int(len(targets_int) * cov)
        if n_keep < 10:
            continue
        keep = sorted_idx[:n_keep]
        sp, st = ens_preds[keep], targets_int[keep]
        acc = (sp == st).mean()
        qwk_v = compute_metrics(st, sp)["qwk"] if len(np.unique(st)) >= 2 else float("nan")
        print(f"    {cov:10.0%}  {acc:10.4f}  {qwk_v:10.4f}  {n_keep:10d}")
        coverage_data.append((cov, acc, qwk_v, n_keep))

    # --- Save artefacts ---
    tag = "_".join(str(eid) for eid in args.exps)
    out_dir = base.results_dir.parent / f"ensemble_{tag}" / "mc_dropout"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_confusion_matrix(targets_int, ens_preds, out_dir / "mc_dropout_cm.png")
    save_classification_report(targets_int, ens_preds,
                                out_dir / "mc_dropout_cls_report.txt")

    df = pd.DataFrame({
        "id_code": codes_ref,
        "true_label": targets_int,
        "ens_prediction": ens_preds,
        "entropy": np.round(entropy, 6),
        "mutual_info": np.round(mi, 6),
        "correct": is_correct.astype(int),
    })
    for c, name in enumerate(CLASS_NAMES):
        df[f"p_{name}"] = np.round(mean_probs[:, c], 6)
    df.to_csv(out_dir / "mc_dropout_predictions.csv", index=False)

    np.savez(out_dir / "mc_dropout_probs.npz",
             id_code=np.array(codes_ref),
             true_label=targets_int,
             mean_probs=mean_probs,
             entropy=entropy,
             mutual_info=mi)

    _plot_entropy_histogram(entropy, is_correct, out_dir / "entropy_histogram.png")
    _plot_entropy_by_class(entropy, targets_int, out_dir / "entropy_by_class.png")
    _plot_coverage_accuracy(coverage_data, out_dir / "coverage_accuracy.png")

    with open(out_dir / "mc_dropout_summary.txt", "w") as f:
        f.write(f"Ensemble members: {[c.exp_name for c in configs]}\n")
        f.write(f"MC passes per member (T): {args.T}\n")
        f.write(f"Total MC samples per image: {args.T * len(configs)}\n")
        f.write(f"Member dropout rates: {[c.head_dropout for c in configs]}\n\n")
        f.write("=== MC Averaged Metrics ===\n")
        for k, v in metrics.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write(f"\n=== Uncertainty Quality ===\n")
        f.write(f"  AUROC(entropy -> error): {auroc_entropy:.4f}\n")
        f.write(f"  AUROC(MI -> error):      {auroc_mi:.4f}\n")
        f.write(f"  Mean entropy (correct):  {entropy[is_correct].mean():.4f}\n")
        if is_wrong.sum():
            f.write(f"  Mean entropy (wrong):    {entropy[is_wrong].mean():.4f}\n")

    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
