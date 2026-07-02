"""Canonical metrics for docs/result-full.md.

Reads every experiment's *_preds.csv and ensemble *_test_probs.npz, computes
QWK / Acc / Macro-F1 / per-class F1 / binary collapse / bootstrap 95% CIs
and saves results to scripts/_canonical_metrics.json.

Bootstrap: 5000 samples, np.random.default_rng(42), paired index across configs.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    cohen_kappa_score, f1_score, accuracy_score, confusion_matrix,
    roc_auc_score, classification_report,
)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
LABELS = pd.read_csv(ROOT / "data" / "test_label.csv")
LABELS = LABELS.rename(columns={"diagnosis": "true_label_gt"})
N_TEST = len(LABELS)
assert N_TEST == 550, N_TEST

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

# Registry: id -> (display_name, preds_csv_path, decoder, role)
REGISTRY: dict[str, dict] = {
    # Phase-0 + Phase-1
    "B0":  {"dir": "exp00_baseline",          "preds": "exp00_baseline_preds.csv",          "decoder": "argmax", "role": "phase1"},
    "B1":  {"dir": "exp01_std_aug",           "preds": "exp01_std_aug_preds.csv",           "decoder": "argmax", "role": "phase1"},
    "B2":  {"dir": "exp02_adv_aug",           "preds": "exp02_adv_aug_preds.csv",           "decoder": "argmax", "role": "phase1"},
    "B3":  {"dir": "exp03_focal_loss",        "preds": "exp03_focal_loss_preds.csv",        "decoder": "argmax", "role": "phase1"},
    "B4":  {"dir": "exp04_label_smooth",      "preds": "exp04_label_smooth_preds.csv",      "decoder": "argmax", "role": "phase1_neg"},
    "B5":  {"dir": "exp05_mixup_branch3",     "preds": "exp05_mixup_preds.csv",             "decoder": "argmax", "role": "phase1_neg"},
    "B6":  {"dir": "exp06_cutmix",            "preds": "exp06_cutmix_preds.csv",            "decoder": "argmax", "role": "phase1_neg"},
    "B7":  {"dir": "exp07_regression",        "preds": "exp07_regression_preds.csv",        "decoder": "regression-round", "role": "phase1_neg"},
    "B8":  {"dir": "exp08_gem",               "preds": "exp08_gem_preds.csv",               "decoder": "argmax", "role": "phase1"},
    "B9":  {"dir": "exp09_cosine_lr",         "preds": "exp09_cosine_lr_preds.csv",         "decoder": "argmax", "role": "phase1_neg"},
    "B12": {"dir": "exp12_opt_thresh_opA",    "preds": "exp12_opt_thresh_opA_preds.csv",    "decoder": "cum-opt-opA", "role": "phase1_dep"},
    "B13": {"dir": "exp13_pseudo_label",      "preds": "exp13_pseudo_label_preds.csv",      "decoder": "argmax", "role": "phase1_dep"},
    # Phase-2 A-line
    "A0":  {"dir": "exp100_a0_baseline_ece",  "preds": "exp100_a0_baseline_ece_preds.csv",  "decoder": "argmax", "role": "phase2_a"},
    "A0b": {"dir": "exp101_a0b_weighted_sampler", "preds": "exp101_a0b_weighted_sampler_preds.csv", "decoder": "argmax", "role": "phase2_a"},
    "A0c": {"dir": "exp102_a0c_offline_oversample-v2", "preds": "exp102_a0c_offline_oversample_preds.csv", "decoder": "argmax", "role": "phase2_a"},
    "A1":  {"dir": "exp103_a1_ordsupcon_aptos-v2", "preds": "exp103_a1_ordsupcon_aptos_preds.csv", "decoder": "argmax", "role": "phase2_a"},
    "A1v3":{"dir": "exp605_a1v3_ordsupcon_40ep", "preds": "exp605_a1v3_ordsupcon_40ep_preds.csv", "decoder": "argmax", "role": "phase2_a_util"},
    "A2":  {"dir": "exp200_a2_ordsupcon_eyepacs", "preds": "exp200_a2_ordsupcon_eyepacs_preds.csv", "decoder": "argmax", "role": "phase2_a"},
    "A2v2":{"dir": "exp201_a2v2_freeze5_eyepacs", "preds": "exp201_a2v2_freeze5_eyepacs_preds.csv", "decoder": "argmax", "role": "phase2_a"},
    # D-line
    "D1":  {"dir": "exp300_d1_dropout_cosine", "preds": "exp300_d1_dropout_cosine_preds.csv", "decoder": "argmax", "role": "headline"},
    "D1_cum_opt": {"dir": "exp300_d1_dropout_cosine/threshold_opt", "preds": "exp300_d1_dropout_cosine_thresh_opt_preds.csv", "decoder": "cum-opt", "role": "headline"},
    # F-line
    "F2":  {"dir": "exp501_f2_joint_ordsupcon", "preds": "exp501_f2_joint_ordsupcon_preds.csv", "decoder": "argmax", "role": "phase2_f"},
    "F3":  {"dir": "exp502_f3_joint_fixed",   "preds": "exp502_f3_joint_fixed_preds.csv",   "decoder": "argmax", "role": "phase2_f"},
    # G-line
    "G1":  {"dir": "exp600_g1_corn_imagenet", "preds": "exp600_g1_corn_imagenet_preds.csv", "decoder": "argmax", "role": "phase2_g"},
    # H-line
    "H0":  {"dir": "exp700_h0_linear_probe_a2", "preds": "exp700_h0_linear_probe_a2_preds.csv", "decoder": "argmax", "role": "phase2_h"},
    "H1":  {"dir": "exp701_h1_ordsupcon_d1recipe", "preds": "exp701_h1_ordsupcon_d1recipe_preds.csv", "decoder": "argmax", "role": "headline"},
    "H1_cum_opt": {"dir": "exp701_h1_ordsupcon_d1recipe/threshold_opt", "preds": "exp701_h1_ordsupcon_d1recipe_thresh_opt_preds.csv", "decoder": "cum-opt", "role": "headline"},
    "H2":  {"dir": "exp702_h2_lpft_a2",       "preds": "exp702_h2_lpft_a2_preds.csv",       "decoder": "argmax", "role": "phase2_h"},
    "H5":  {"dir": "exp705_h5_a1_d1recipe",   "preds": "exp705_h5_a1_d1recipe_preds.csv",   "decoder": "argmax", "role": "headline"},
    # I-line
    "I1":  {"dir": "exp802_i1_emd_on_d1_recipe", "preds": "exp802_i1_emd_on_d1_recipe_preds.csv", "decoder": "argmax", "role": "phase2_i"},
    "I3":  {"dir": "exp804_i3_swad_on_d1",    "preds": "exp804_i3_swad_on_d1_preds.csv",    "decoder": "argmax", "role": "phase2_i"},
    "I4":  {"dir": "exp805_i4_l2sp_a2_d1recipe", "preds": "exp805_i4_l2sp_a2_d1recipe_preds.csv", "decoder": "argmax", "role": "phase2_i"},
    "I5":  {"dir": "exp806_i5_prototype_head_a2_emd", "preds": "exp806_i5_prototype_head_a2_emd_preds.csv", "decoder": "argmax", "role": "phase2_i"},
    # Champion + ensembles
    "C9":  {"dir": "exp900_ordsupcon_champion", "preds": "exp900_ordsupcon_champion_preds.csv", "decoder": "argmax", "role": "ensemble_member"},
    "C9_cum_opt": {"dir": "exp900_ordsupcon_champion/threshold_opt", "preds": "exp900_ordsupcon_champion_thresh_opt_preds.csv", "decoder": "cum-opt", "role": "ensemble_member"},
    "E-DH_argmax":  {"dir": "ensemble_300_701", "preds": "ensemble_300_701_argmax_preds.csv", "decoder": "argmax", "role": "ensemble"},
    "E-DH_egopt":   {"dir": "ensemble_300_701", "preds": "ensemble_300_701_expected_grade_opt_preds.csv", "decoder": "expected-grade-opt", "role": "ensemble"},
    "E-DHH_argmax": {"dir": "ensemble_300_701_705", "preds": "ensemble_300_701_705_argmax_preds.csv", "decoder": "argmax", "role": "ensemble"},
    "E-DC_egopt":   {"dir": "ensemble_300_900", "preds": "ensemble_300_900_expected_grade_opt_preds.csv", "decoder": "expected-grade-opt", "role": "ensemble"},
    "E-DCH_egopt":  {"dir": "ensemble_300_900_701", "preds": "ensemble_300_900_701_expected_grade_opt_preds.csv", "decoder": "expected-grade-opt", "role": "ensemble"},
    "E-DCH-eq_argmax": {"dir": "ensemble_900_300_701", "preds": "ensemble_900_300_701_argmax_preds.csv", "decoder": "argmax", "role": "ensemble"},
    "E-DCH-eq_egopt":  {"dir": "ensemble_900_300_701", "preds": "ensemble_900_300_701_expected_grade_opt_preds.csv", "decoder": "expected-grade-opt", "role": "ensemble"},
}

RNG = np.random.default_rng(42)
N_BOOT = 5000
BOOT_IDX = RNG.integers(0, N_TEST, size=(N_BOOT, N_TEST))  # shared paired idx


def _qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def _macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2, 3, 4], zero_division=0)


def _per_class_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3, 4], zero_division=0)


def _boot_ci(y_true, y_pred, fn):
    """Paired bootstrap CI using shared BOOT_IDX. Returns (point, lo, hi)."""
    point = fn(y_true, y_pred)
    vals = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = BOOT_IDX[b]
        vals[b] = fn(y_true[idx], y_pred[idx])
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(point), float(lo), float(hi), vals


def _per_class_f1_ci(y_true, y_pred):
    point = _per_class_f1(y_true, y_pred)
    vals = np.empty((N_BOOT, 5))
    for b in range(N_BOOT):
        idx = BOOT_IDX[b]
        vals[b] = _per_class_f1(y_true[idx], y_pred[idx])
    lo = np.quantile(vals, 0.025, axis=0)
    hi = np.quantile(vals, 0.975, axis=0)
    return point.tolist(), lo.tolist(), hi.tolist()


def _binary_metrics(y_true, y_pred, threshold):
    """Binary collapse: positive iff grade >= threshold."""
    yt = (y_true >= threshold).astype(int)
    yp = (y_pred >= threshold).astype(int)
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    tn, fp, fn_, tp = cm.ravel()
    sens = tp / (tp + fn_) if (tp + fn_) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn_) if (tn + fn_) else 0.0
    bal = 0.5 * (sens + spec)
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn_), "tp": int(tp),
            "sens": float(sens), "spec": float(spec), "ppv": float(ppv),
            "npv": float(npv), "balanced_acc": float(bal)}


def _binary_sens_spec_ci(y_true, y_pred, threshold):
    yt = (y_true >= threshold).astype(int)
    yp = (y_pred >= threshold).astype(int)
    def _sens(yt_, yp_):
        tp = int(((yt_ == 1) & (yp_ == 1)).sum())
        p = int((yt_ == 1).sum())
        return tp / p if p else 0.0
    def _spec(yt_, yp_):
        tn = int(((yt_ == 0) & (yp_ == 0)).sum())
        n = int((yt_ == 0).sum())
        return tn / n if n else 0.0
    sens_vals = np.empty(N_BOOT)
    spec_vals = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = BOOT_IDX[b]
        sens_vals[b] = _sens(yt[idx], yp[idx])
        spec_vals[b] = _spec(yt[idx], yp[idx])
    return {
        "sens_ci": [float(np.quantile(sens_vals, 0.025)), float(np.quantile(sens_vals, 0.975))],
        "spec_ci": [float(np.quantile(spec_vals, 0.025)), float(np.quantile(spec_vals, 0.975))],
    }


def _load_preds(cfg):
    path = RES / cfg["dir"] / cfg["preds"]
    if not path.exists():
        return None
    df = pd.read_csv(path)
    # Join with ground truth on id_code, sort by ground-truth ordering for stable bootstrap pairing
    merged = LABELS.merge(df, on="id_code", how="left")
    if merged["rounded_prediction"].isna().any():
        n_missing = int(merged["rounded_prediction"].isna().sum())
        return {"error": f"missing predictions: {n_missing}"}
    y_true = merged["true_label_gt"].to_numpy(dtype=int)
    y_pred = merged["rounded_prediction"].to_numpy(dtype=int)
    return {"y_true": y_true, "y_pred": y_pred, "path": str(path.relative_to(ROOT)).replace("\\", "/")}


def _compute_one(cfg, label):
    loaded = _load_preds(cfg)
    if loaded is None or "error" in (loaded or {}):
        return {"id": label, "error": loaded["error"] if loaded else "missing file"}
    y_true = loaded["y_true"]
    y_pred = loaded["y_pred"]
    qwk_pt, qwk_lo, qwk_hi, qwk_boot = _boot_ci(y_true, y_pred, _qwk)
    mf1_pt, mf1_lo, mf1_hi, mf1_boot = _boot_ci(y_true, y_pred, _macro_f1)
    acc_pt, acc_lo, acc_hi, _ = _boot_ci(y_true, y_pred, accuracy_score)
    pcf1_pt, pcf1_lo, pcf1_hi = _per_class_f1_ci(y_true, y_pred)
    cm5 = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    bin_ref = _binary_metrics(y_true, y_pred, 2)
    bin_st = _binary_metrics(y_true, y_pred, 3)
    bin_ref_ci = _binary_sens_spec_ci(y_true, y_pred, 2)
    bin_st_ci = _binary_sens_spec_ci(y_true, y_pred, 3)
    return {
        "id": label,
        "dir": cfg["dir"], "preds": cfg["preds"], "decoder": cfg["decoder"], "role": cfg["role"],
        "source": loaded["path"],
        "n": int(len(y_true)),
        "qwk": qwk_pt, "qwk_ci": [qwk_lo, qwk_hi],
        "macro_f1": mf1_pt, "macro_f1_ci": [mf1_lo, mf1_hi],
        "accuracy": acc_pt, "accuracy_ci": [acc_lo, acc_hi],
        "per_class_f1": pcf1_pt, "per_class_f1_ci_lo": pcf1_lo, "per_class_f1_ci_hi": pcf1_hi,
        "confusion_matrix_5": cm5.tolist(),
        "binary_referable": {**bin_ref, **bin_ref_ci},
        "binary_sight_threatening": {**bin_st, **bin_st_ci},
        "_y_true_idx": "shared",
        "_qwk_boot_path": None,
        "_mf1_boot_path": None,
    }


def _paired_pvalue(y_true, y_a, y_b, fn):
    """Two-sided paired bootstrap p-value for fn(B) - fn(A) > 0 vs < 0.
    Uses shared BOOT_IDX so the same resample is used for both methods (paired).
    Returns (delta_point, p_two_sided, ci_lo, ci_hi)."""
    a_pt = fn(y_true, y_a)
    b_pt = fn(y_true, y_b)
    delta_pt = b_pt - a_pt
    deltas = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = BOOT_IDX[i]
        deltas[i] = fn(y_true[idx], y_b[idx]) - fn(y_true[idx], y_a[idx])
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    # Two-sided p: 2 * min(P(delta<=0), P(delta>=0))
    p_pos = float((deltas >= 0).mean())
    p_neg = float((deltas <= 0).mean())
    p = float(min(1.0, 2 * min(p_pos, p_neg)))
    return float(delta_pt), float(p), float(lo), float(hi)


def main():
    out: dict = {"settings": {
        "n_test": N_TEST, "n_boot": N_BOOT, "boot_seed": 42,
        "class_names": CLASS_NAMES,
    }, "configs": {}}
    preds_cache: dict[str, np.ndarray] = {}  # id -> y_pred
    y_true_ref = None

    for label, cfg in REGISTRY.items():
        print(f"computing {label} ({cfg['dir']})...")
        rec = _compute_one(cfg, label)
        out["configs"][label] = rec
        if "error" not in rec:
            loaded = _load_preds(cfg)
            preds_cache[label] = loaded["y_pred"]
            if y_true_ref is None:
                y_true_ref = loaded["y_true"]
            else:
                if not np.array_equal(y_true_ref, loaded["y_true"]):
                    rec["_id_align_warning"] = "y_true mismatch with reference"

    # Paired bootstrap pairs for Section 2 (negatives + progressive chain) and Section 10
    pairs = [
        ("B0", "B1"), ("B1", "B2"), ("B2", "B3"), ("B3", "B8"), ("B8", "D1"),
        ("B0", "D1"),
        ("B2", "B4"), ("B2", "B5"), ("B2", "B6"),
        ("B8", "B9"), ("B8", "B12"), ("B8", "B13"), ("B8", "B7"),
        ("A0", "A0b"), ("A0", "A0c"), ("A0c", "A1"), ("A0c", "A2"),
        ("A2", "A2v2"), ("A0", "D1"),
        ("D1", "E-DH_argmax"), ("D1", "E-DHH_argmax"), ("E-DH_argmax", "E-DHH_argmax"),
        ("D1", "H1"), ("D1", "H5"), ("H1", "H5"),
        ("A2", "H0"), ("A2", "H1"), ("A2", "H2"), ("A2", "I4"),
        ("D1", "I1"), ("D1", "I3"),
    ]
    out["pairs"] = []
    for a, b in pairs:
        if a not in preds_cache or b not in preds_cache:
            out["pairs"].append({"a": a, "b": b, "error": "missing preds"})
            continue
        d_qwk, p_qwk, lo_qwk, hi_qwk = _paired_pvalue(y_true_ref, preds_cache[a], preds_cache[b], _qwk)
        d_mf1, p_mf1, lo_mf1, hi_mf1 = _paired_pvalue(y_true_ref, preds_cache[a], preds_cache[b], _macro_f1)
        d_acc, p_acc, lo_acc, hi_acc = _paired_pvalue(y_true_ref, preds_cache[a], preds_cache[b], accuracy_score)
        out["pairs"].append({
            "a": a, "b": b,
            "delta_qwk": d_qwk, "p_qwk": p_qwk, "delta_qwk_ci": [lo_qwk, hi_qwk],
            "delta_macro_f1": d_mf1, "p_macro_f1": p_mf1, "delta_macro_f1_ci": [lo_mf1, hi_mf1],
            "delta_accuracy": d_acc, "p_accuracy": p_acc, "delta_accuracy_ci": [lo_acc, hi_acc],
        })

    out_path = ROOT / "scripts" / "_canonical_metrics.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
    print(f"wrote {out_path}")
    return out


if __name__ == "__main__":
    main()
