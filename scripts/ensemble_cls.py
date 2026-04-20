"""Classification-ensemble inference with TTA and post-hoc threshold opt.

Averages softmax probabilities across members on val+test, converts to an
expected-grade scalar, then fits OptimizedRounder on the pooled val expected
grade. Applies those thresholds to the pooled test expected grade.

Usage:
    python scripts/ensemble_cls.py --exps 900 300 701 --device cuda
    python scripts/ensemble_cls.py --exps 900 300 701 --device cuda --no-tta
    python scripts/ensemble_cls.py --exps 900 300 701 --weights 1.0 0.8 0.8

Each member is loaded from its best_composite checkpoint if present, else
best.pth, else pseudo/swa fallback.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax as sp_softmax
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import NUM_CLASSES, get_config
from src.dataset import build_datasets
from src.evaluate import (
    OptimizedRounder,
    compute_metrics,
    regression_to_class,
    save_classification_report,
    save_confusion_matrix,
    save_predictions,
)
from src.models import build_model
from src.tta import predict_no_tta, predict_with_tta
from src.transforms import get_val_transform


def _find_checkpoint(cfg) -> Path:
    """Prefer best_composite, then best, then pseudo, then swa."""
    candidates = [
        cfg.ckpt_dir / f"{cfg.exp_name}_best_composite.pth",
        cfg.ckpt_dir / f"{cfg.exp_name}_best.pth",
        cfg.ckpt_dir / f"{cfg.exp_name}_pseudo.pth",
        cfg.ckpt_dir / f"{cfg.exp_name}_swa.pth",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"No checkpoint for {cfg.exp_name} in {cfg.ckpt_dir}")


def _member_probs(cfg, val_ds, test_ds, val_loader, test_loader, device, use_tta):
    """Return (val_probs [Nv,C], test_probs [Nt,C]) for one classification member."""
    ckpt = _find_checkpoint(cfg)
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(ckpt, weights_only=True, map_location=device))
    model.eval()
    print(f"  Loaded {cfg.exp_name}: {ckpt.name}")

    if use_tta:
        v_logits, _ = predict_with_tta(model, val_ds, device, is_regression=False)
        t_logits, _ = predict_with_tta(model, test_ds, device, is_regression=False)
    else:
        v_logits, _, _ = predict_no_tta(model, val_loader, device, is_regression=False)
        t_logits, _, _ = predict_no_tta(model, test_loader, device, is_regression=False)

    # Free GPU memory before next member loads
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return sp_softmax(v_logits, axis=1), sp_softmax(t_logits, axis=1)


def _print_metrics(tag, m):
    print(f"\n{'='*64}\n{tag}\n{'='*64}")
    print(f"  QWK:         {m['qwk']:.4f}")
    print(f"  Accuracy:    {m['accuracy']:.4f}")
    print(f"  Macro F1:    {m['macro_f1']:.4f}")
    print(f"  Sensitivity: {m.get('sensitivity', 0):.4f}")
    print(f"  Specificity: {m.get('specificity', 0):.4f}")
    for cname in ("No DR", "Mild", "Moderate", "Severe", "Proliferative"):
        print(f"  F1 {cname:14s}: {m.get(f'f1_{cname}', 0):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exps", type=int, nargs="+", required=True,
                        help="Experiment IDs to ensemble, e.g. 900 300 701")
    parser.add_argument("--weights", type=float, nargs="*", default=None,
                        help="Optional per-member weights (same count as --exps)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-tta", action="store_true", help="Disable TTA")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_tta = not args.no_tta
    print(f"Device: {device} | TTA: {use_tta}")

    configs = [get_config(eid) for eid in args.exps]
    if args.weights:
        assert len(args.weights) == len(configs), "weights count must match exps"
        weights = np.array(args.weights, dtype=np.float64)
    else:
        weights = np.ones(len(configs), dtype=np.float64)
    weights = weights / weights.sum()
    print(f"Members: {[c.exp_name for c in configs]}")
    print(f"Weights: {weights.tolist()}")

    # Build val+test datasets once (use base cfg for oversampling/etc.; all
    # APTOS members share the same val/test split by construction).
    base = configs[0]
    transform_val = get_val_transform()
    _, val_ds, test_ds = build_datasets(base, transform_train=None,
                                         transform_val=transform_val)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)
    val_targets = np.array([val_ds.samples[i][1] for i in range(len(val_ds))],
                           dtype=np.int64)
    test_targets = np.array([test_ds.samples[i][1] for i in range(len(test_ds))],
                            dtype=np.int64)
    test_codes = [test_ds.samples[i][0] for i in range(len(test_ds))]
    print(f"Val: {len(val_ds)} | Test: {len(test_ds)}")

    # Collect TTA softmax probs for every member
    val_probs_list, test_probs_list = [], []
    for cfg in configs:
        assert not cfg.is_regression, f"{cfg.exp_name} is regression — not supported by this cls ensemble"
        print(f"\nMember: {cfg.exp_name}")
        vp, tp = _member_probs(cfg, val_ds, test_ds, val_loader, test_loader,
                               device, use_tta)
        val_probs_list.append(vp)
        test_probs_list.append(tp)

        # Per-member sanity: report that member's solo test QWK at argmax
        m_argmax = compute_metrics(test_targets, tp.argmax(axis=1))
        print(f"  solo test QWK (argmax): {m_argmax['qwk']:.4f}")

    # Weighted average of probabilities
    val_probs = np.zeros_like(val_probs_list[0])
    test_probs = np.zeros_like(test_probs_list[0])
    for w, vp, tp in zip(weights, val_probs_list, test_probs_list):
        val_probs += w * vp
        test_probs += w * tp

    # Strategy A: argmax of averaged probs
    argmax_preds = test_probs.argmax(axis=1)
    m_argmax = compute_metrics(test_targets, argmax_preds, test_probs)
    _print_metrics("ENSEMBLE — argmax of avg probs", m_argmax)

    # Strategy B: expected-grade + OptimizedRounder fit on val (pure QWK obj)
    class_axis = np.arange(NUM_CLASSES, dtype=np.float64)
    val_eg = val_probs @ class_axis
    test_eg = test_probs @ class_axis
    rounder = OptimizedRounder()
    thresholds = rounder.fit(val_eg, val_targets)
    print(f"\nExpected-grade thresholds (QWK-obj): {[f'{t:.4f}' for t in thresholds]}")
    eg_preds = regression_to_class(test_eg, thresholds)
    m_eg = compute_metrics(test_targets, eg_preds, test_probs)
    _print_metrics("ENSEMBLE — expected-grade + OptimizedRounder (QWK obj)", m_eg)

    # Strategy C: cumulative threshold opt on val
    from scipy.optimize import minimize
    from src.evaluate import quadratic_weighted_kappa

    def cum_to_class(probs, thr):
        preds = np.zeros(probs.shape[0], dtype=int)
        for k in range(1, NUM_CLASSES):
            preds += (probs[:, k:].sum(axis=1) >= thr[k - 1]).astype(int)
        return preds

    def neg_qwk_cum(thr, probs, targets):
        return -quadratic_weighted_kappa(targets, cum_to_class(probs, thr.tolist()))

    res = minimize(neg_qwk_cum, x0=np.array([0.5, 0.5, 0.5, 0.5]),
                   args=(val_probs, val_targets),
                   method="Nelder-Mead",
                   options={"maxiter": 2000, "xatol": 1e-6})
    cum_thr = sorted(res.x.tolist())
    print(f"\nCumulative thresholds (on val): {[f'{t:.4f}' for t in cum_thr]}")
    cum_preds = cum_to_class(test_probs, cum_thr)
    m_cum = compute_metrics(test_targets, cum_preds, test_probs)
    _print_metrics("ENSEMBLE — cumulative probability thresholds", m_cum)

    # Strategy D: expected-grade thresholds that maximise COMPOSITE
    # (0.6·QWK + 0.4·MacroF1) on val. The pure-QWK objective collapses
    # minority classes; composite preserves minority F1 while keeping QWK.
    from sklearn.metrics import f1_score

    def _composite_val(thr_arr, probs_eg, targets):
        preds = regression_to_class(probs_eg, sorted(thr_arr.tolist()))
        q = quadratic_weighted_kappa(targets, preds)
        mf1 = f1_score(targets, preds, average="macro", zero_division=0)
        return -(0.6 * q + 0.4 * mf1)

    res_c = minimize(_composite_val, x0=np.array([0.5, 1.5, 2.5, 3.5]),
                     args=(val_eg, val_targets),
                     method="Nelder-Mead",
                     options={"maxiter": 2000, "xatol": 1e-6})
    comp_thr = sorted(res_c.x.tolist())
    print(f"\nExpected-grade thresholds (composite-obj): {[f'{t:.4f}' for t in comp_thr]}")
    comp_preds = regression_to_class(test_eg, comp_thr)
    m_comp = compute_metrics(test_targets, comp_preds, test_probs)
    _print_metrics("ENSEMBLE — expected-grade + composite-opt thresholds", m_comp)

    # Selection: composite score on val (not QWK alone — avoids minority collapse).
    def _val_composite_of(preds_val):
        q = quadratic_weighted_kappa(val_targets, preds_val)
        mf1 = f1_score(val_targets, preds_val, average="macro", zero_division=0)
        return q, mf1, 0.6 * q + 0.4 * mf1

    def _strategy_val(preds_on_val):
        q, mf1, comp = _val_composite_of(preds_on_val)
        return q, mf1, comp

    vq_am, vmf1_am, vcomp_am = _strategy_val(val_probs.argmax(axis=1))
    vq_eg, vmf1_eg, vcomp_eg = _strategy_val(regression_to_class(val_eg, thresholds))
    vq_cum, vmf1_cum, vcomp_cum = _strategy_val(cum_to_class(val_probs, cum_thr))
    vq_c, vmf1_c, vcomp_c = _strategy_val(regression_to_class(val_eg, comp_thr))

    print(f"\n{'='*72}\nVAL selection table (composite = 0.6·QWK + 0.4·MacroF1):")
    print(f"  {'strategy':30s}  {'QWK':>7s}  {'MacroF1':>8s}  {'Composite':>10s}")
    print(f"  {'argmax':30s}  {vq_am:7.4f}  {vmf1_am:8.4f}  {vcomp_am:10.4f}")
    print(f"  {'expected_grade_qwk_opt':30s}  {vq_eg:7.4f}  {vmf1_eg:8.4f}  {vcomp_eg:10.4f}")
    print(f"  {'cumulative_qwk_opt':30s}  {vq_cum:7.4f}  {vmf1_cum:8.4f}  {vcomp_cum:10.4f}")
    print(f"  {'expected_grade_composite_opt':30s}  {vq_c:7.4f}  {vmf1_c:8.4f}  {vcomp_c:10.4f}")
    print(f"{'='*72}")

    strategies = {
        "argmax": (argmax_preds, m_argmax, vcomp_am),
        "expected_grade_qwk_opt": (eg_preds, m_eg, vcomp_eg),
        "cumulative_qwk_opt": (cum_preds, m_cum, vcomp_cum),
        "expected_grade_composite_opt": (comp_preds, m_comp, vcomp_c),
    }
    best_name = max(strategies, key=lambda k: strategies[k][2])
    best_preds, best_metrics, _ = strategies[best_name]
    print(f"\nBEST (by val composite): {best_name}")
    print(f"  TEST  QWK={best_metrics['qwk']:.4f}  "
          f"MacroF1={best_metrics['macro_f1']:.4f}  "
          f"F1-Severe={best_metrics.get('f1_Severe', 0):.4f}")

    # Save artefacts under a dedicated ensemble directory
    tag = "_".join(str(eid) for eid in args.exps)
    out_dir = base.results_dir.parent / f"ensemble_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"ensemble_{tag}_{best_name}"

    save_confusion_matrix(test_targets, best_preds, out_dir / f"{stem}_cm.png")
    save_classification_report(test_targets, best_preds,
                                out_dir / f"{stem}_cls_report.txt")
    save_predictions(test_codes, test_eg, best_preds, test_targets,
                     out_dir / f"{stem}_preds.csv")

    with open(out_dir / f"ensemble_{tag}_summary.txt", "w") as f:
        f.write(f"Members: {[c.exp_name for c in configs]}\n")
        f.write(f"Weights: {weights.tolist()}\n")
        f.write(f"TTA: {use_tta}\n\n")
        for sname, (_, sm, s_vcomp) in strategies.items():
            f.write(f"--- {sname} (val composite {s_vcomp:.4f}) ---\n")
            for k in ("qwk", "accuracy", "macro_f1", "sensitivity", "specificity",
                      "f1_No DR", "f1_Mild", "f1_Moderate", "f1_Severe",
                      "f1_Proliferative"):
                f.write(f"  {k}: {sm.get(k, 0):.4f}\n")
            f.write("\n")
        f.write(
            f"Best (by val composite): {best_name}\n"
            f"  test QWK      = {best_metrics['qwk']:.4f}\n"
            f"  test MacroF1  = {best_metrics['macro_f1']:.4f}\n"
            f"  test F1-Severe= {best_metrics.get('f1_Severe', 0):.4f}\n"
        )

    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
