"""Build Appendix A supplementary tables.

Emits, under results/result_cache/:
  - appendix_a_full_metrics.csv        one row per experiment, all 5-grade metrics
  - appendix_a_per_class_f1.csv        per-class F1 for every experiment
  - appendix_a_kappa_split.csv         nominal / linear / quadratic kappa per experiment
  - appendix_a_referable_binary.csv    referable-DR (>=2) clinical metrics per experiment
  - appendix_a_ablation_bootstrap.csv  paired-bootstrap deltas & p-values between
                                       consecutive ablation-ladder steps (and vs baseline)
  - appendix_a_external_messidor2.csv  zero-shot Messidor-2 metrics (from summary files)
"""
from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

RESULTS_DIR = Path("results")
OUT_DIR = RESULTS_DIR / "result_cache"
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
RNG_SEED = 42
N_BOOT = 2000


# --- experiment registry ------------------------------------------------------

EXPERIMENTS: list[tuple[str, str, str]] = [
    # (tag, display name, preds.csv path)
    ("exp00", "Baseline (exp00)", "exp00_baseline/exp00_baseline_preds.csv"),
    ("exp01_std", "+ Std aug (exp01)", "exp01_std_aug/exp01_std_aug_preds.csv"),
    ("exp02_adv", "+ Adv aug (exp02)", "exp02_adv_aug/exp02_adv_aug_preds.csv"),
    ("exp03_foc", "+ Focal loss (exp03)", "exp03_focal_loss/exp03_focal_loss_preds.csv"),
    ("exp04_ls", "+ Label smooth (exp04)", "exp04_label_smooth/exp04_label_smooth_preds.csv"),
    ("exp05_mix", "+ Mixup (exp05)", "exp05_mixup_branch3/exp05_mixup_preds.csv"),
    ("exp06_cut", "+ CutMix (exp06)", "exp06_cutmix/exp06_cutmix_preds.csv"),
    ("exp07_reg", "Regression head (exp07)", "exp07_regression/exp07_regression_preds.csv"),
    ("exp08_gem", "GeM pooling (exp08)", "exp08_gem/exp08_gem_preds.csv"),
    ("exp09_cos", "Cosine LR (exp09)", "exp09_cosine_lr/exp09_cosine_lr_preds.csv"),
    ("exp12_thr", "Threshold opt (exp12)", "exp12_opt_thresh_opA/exp12_opt_thresh_opA_preds.csv"),
    ("exp13_pl", "Pseudo-label (exp13)", "exp13_pseudo_label/exp13_pseudo_label_preds.csv"),
    ("exp100_a0", "A0 baseline+ECE (exp100)", "exp100_a0_baseline_ece/exp100_a0_baseline_ece_preds.csv"),
    ("exp101_a0b", "A0b weighted sampler (exp101)", "exp101_a0b_weighted_sampler/exp101_a0b_weighted_sampler_preds.csv"),
    ("exp102_a0c", "A0c offline oversample (exp102)", "exp102_a0c_offline_oversample/exp102_a0c_offline_oversample_preds.csv"),
    ("exp102_a0c_v2", "A0c offline oversample v2", "exp102_a0c_offline_oversample-v2/exp102_a0c_offline_oversample_preds.csv"),
    ("exp103_a1", "A1 OrdSupCon APTOS (exp103)", "exp103_a1_ordsupcon_aptos/exp103_a1_ordsupcon_aptos_preds.csv"),
    ("exp103_a1_v2", "A1 OrdSupCon APTOS v2", "exp103_a1_ordsupcon_aptos-v2/exp103_a1_ordsupcon_aptos_preds.csv"),
    ("exp200_a2", "A2 OrdSupCon EyePACS (exp200)", "exp200_a2_ordsupcon_eyepacs/exp200_a2_ordsupcon_eyepacs_preds.csv"),
    ("exp201_a2v2", "A2v2 freeze5 (exp201)", "exp201_a2v2_freeze5_eyepacs/exp201_a2v2_freeze5_eyepacs_preds.csv"),
    ("exp300", "D1 dropout+cosine (exp300)", "exp300_d1_dropout_cosine/exp300_d1_dropout_cosine_preds.csv"),
    ("exp501_f2", "F2 joint OrdSupCon (exp501)", "exp501_f2_joint_ordsupcon/exp501_f2_joint_ordsupcon_preds.csv"),
    ("exp502_f3", "F3 joint fixed (exp502)", "exp502_f3_joint_fixed/exp502_f3_joint_fixed_preds.csv"),
    ("exp600_g1", "G1 CORN ImageNet (exp600)", "exp600_g1_corn_imagenet/exp600_g1_corn_imagenet_preds.csv"),
    ("exp605_a1v3", "A1v3 OrdSupCon 40ep (exp605)", "exp605_a1v3_ordsupcon_40ep/exp605_a1v3_ordsupcon_40ep_preds.csv"),
    ("exp700_lp", "H0 linear probe A2 (exp700)", "exp700_h0_linear_probe_a2/exp700_h0_linear_probe_a2_preds.csv"),
    ("exp701", "H1 OrdSupCon D1 (exp701)", "exp701_h1_ordsupcon_d1recipe/exp701_h1_ordsupcon_d1recipe_preds.csv"),
    ("exp702_lpft", "H2 LP-FT A2 (exp702)", "exp702_h2_lpft_a2/exp702_h2_lpft_a2_preds.csv"),
    ("exp705", "H5 A1+D1 (exp705)", "exp705_h5_a1_d1recipe/exp705_h5_a1_d1recipe_preds.csv"),
    ("exp802_emd", "I1 EMD on D1 (exp802)", "exp802_i1_emd_on_d1_recipe/exp802_i1_emd_on_d1_recipe_preds.csv"),
    ("exp804_swad", "I3 SWAD on D1 (exp804)", "exp804_i3_swad_on_d1/exp804_i3_swad_on_d1_preds.csv"),
    ("exp805_l2sp", "I4 L2-SP A2+D1 (exp805)", "exp805_i4_l2sp_a2_d1recipe/exp805_i4_l2sp_a2_d1recipe_preds.csv"),
    ("exp806_proto", "I5 prototype head EMD (exp806)", "exp806_i5_prototype_head_a2_emd/exp806_i5_prototype_head_a2_emd_preds.csv"),
    ("exp900", "Champion OrdSupCon (exp900)", "exp900_ordsupcon_champion/exp900_ordsupcon_champion_preds.csv"),
    ("ens_300_701", "Ensemble 300+701 argmax", "ensemble_300_701/ensemble_300_701_argmax_preds.csv"),
    ("ens_300_701_eg", "Ensemble 300+701 exp-grade", "ensemble_300_701/ensemble_300_701_expected_grade_opt_preds.csv"),
    ("ens_300_900", "Ensemble 300+900 exp-grade", "ensemble_300_900/ensemble_300_900_expected_grade_opt_preds.csv"),
    ("ens_300_900_701", "Ensemble 300+900+701 exp-grade", "ensemble_300_900_701/ensemble_300_900_701_expected_grade_opt_preds.csv"),
    ("ens_900_300_701_arg", "Ensemble 900+300+701 argmax", "ensemble_900_300_701/ensemble_900_300_701_argmax_preds.csv"),
    ("ens_900_300_701_eg", "Ensemble 900+300+701 exp-grade", "ensemble_900_300_701/ensemble_900_300_701_expected_grade_opt_preds.csv"),
    ("ens_300_701_705", "Ensemble 300+701+705 argmax", "ensemble_300_701_705/ensemble_300_701_705_argmax_preds.csv"),
]


# --- metric helpers -----------------------------------------------------------

def _load(preds_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(preds_csv)
    y_true = df["true_label"].to_numpy(dtype=int)
    y_pred = df["rounded_prediction"].to_numpy(dtype=int)
    return y_true, y_pred


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    f1_per = f1_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average=None, zero_division=0)
    # referable binary (>=2)
    yt_bin = (y_true >= 2).astype(int)
    yp_bin = (y_pred >= 2).astype(int)
    tp = int(((yt_bin == 1) & (yp_bin == 1)).sum())
    fp = int(((yt_bin == 0) & (yp_bin == 1)).sum())
    fn = int(((yt_bin == 1) & (yp_bin == 0)).sum())
    tn = int(((yt_bin == 0) & (yp_bin == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "linear_kappa": float(cohen_kappa_score(y_true, y_pred, weights="linear")),
        "nominal_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "mae": mae,
        "referable_sensitivity": sens,
        "referable_specificity": spec,
        "referable_ppv": ppv,
        "referable_npv": npv,
        "ref_tp": tp, "ref_fp": fp, "ref_fn": fn, "ref_tn": tn,
        **{f"f1_{c}": float(f1_per[i]) for i, c in enumerate(CLASS_NAMES)},
        **{
            f"support_{c}": int((y_true == i).sum())
            for i, c in enumerate(CLASS_NAMES)
        },
    }


def _bootstrap_qwk_ci(y_true, y_pred, n_boot=N_BOOT, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[i] = cohen_kappa_score(y_true[idx], y_pred[idx], weights="quadratic")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


# --- ablation paired bootstrap ------------------------------------------------

# Align with the existing ablation_ladder.csv semantics.
ABLATION_LADDER = [
    ("baseline", "exp00"),
    ("+ std aug", "exp01_std"),
    ("+ adv aug", "exp02_adv"),
    ("+ focal loss", "exp03_foc"),
    ("+ D1 (cosine + dropout)", "exp300"),
    ("+ OrdSupCon (h1 / D1 recipe)", "exp701"),
    ("+ ensemble (900 + 300 + 701)", "ens_900_300_701_eg"),
]


def _paired_bootstrap(y_true, yA, yB, n_boot=N_BOOT, seed=RNG_SEED):
    """Paired bootstrap over a shared test set.

    Returns deltas (A minus B) and p-values (two-sided) for QWK, macro-F1, accuracy.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = {"qwk": [], "macro_f1": [], "accuracy": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        ya = yA[idx]
        yb = yB[idx]
        deltas["qwk"].append(
            cohen_kappa_score(yt, ya, weights="quadratic")
            - cohen_kappa_score(yt, yb, weights="quadratic")
        )
        deltas["macro_f1"].append(
            f1_score(yt, ya, average="macro", zero_division=0)
            - f1_score(yt, yb, average="macro", zero_division=0)
        )
        deltas["accuracy"].append(accuracy_score(yt, ya) - accuracy_score(yt, yb))

    out = []
    obs = {
        "qwk": cohen_kappa_score(y_true, yA, weights="quadratic")
        - cohen_kappa_score(y_true, yB, weights="quadratic"),
        "macro_f1": f1_score(y_true, yA, average="macro", zero_division=0)
        - f1_score(y_true, yB, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, yA) - accuracy_score(y_true, yB),
    }
    for m, vals in deltas.items():
        arr = np.asarray(vals)
        lo, hi = np.percentile(arr, [2.5, 97.5])
        # two-sided p via proportion of centred samples at least as extreme as obs
        centred = arr - obs[m]
        p = float(np.mean(np.abs(centred) >= abs(obs[m])))
        out.append(
            {
                "metric": m,
                "delta": float(obs[m]),
                "ci_lo": float(lo),
                "ci_hi": float(hi),
                "p_value": p,
            }
        )
    return out


# --- Messidor-2 external eval -------------------------------------------------

def _parse_messidor_summary(path: Path) -> dict[str, float | int | str]:
    out: dict[str, float | int | str] = {"source": path.parent.name}
    text = path.read_text()
    for line in text.splitlines():
        m = re.match(r"\s*([a-zA-Z_][\w\s]*?):\s+([-0-9.]+)\s*$", line)
        if m:
            key = m.group(1).strip().replace(" ", "_")
            try:
                out[key] = float(m.group(2))
            except ValueError:
                out[key] = m.group(2)
        m2 = re.match(r"\s*TP:\s*(\d+),\s*TN:\s*(\d+),\s*FP:\s*(\d+),\s*FN:\s*(\d+)", line)
        if m2:
            out["tp"] = int(m2.group(1))
            out["tn"] = int(m2.group(2))
            out["fp"] = int(m2.group(3))
            out["fn"] = int(m2.group(4))
    return out


# --- main ---------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    full_rows: list[dict] = []
    per_class_rows: list[dict] = []
    kappa_rows: list[dict] = []
    ref_rows: list[dict] = []
    preds_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for tag, name, rel in EXPERIMENTS:
        p = RESULTS_DIR / rel
        if not p.exists():
            print(f"  [skip] {tag}: {p} missing")
            continue
        y_true, y_pred = _load(p)
        preds_cache[tag] = (y_true, y_pred)
        m = _metrics(y_true, y_pred)
        qwk_lo, qwk_hi = _bootstrap_qwk_ci(y_true, y_pred)
        row = {"experiment_tag": tag, "experiment_name": name, **m,
               "qwk_ci_lo": qwk_lo, "qwk_ci_hi": qwk_hi,
               "source": rel}
        full_rows.append(row)
        for c in CLASS_NAMES:
            per_class_rows.append(
                {
                    "experiment_tag": tag,
                    "experiment_name": name,
                    "class": c,
                    "f1": m[f"f1_{c}"],
                    "support": m[f"support_{c}"],
                }
            )
        kappa_rows.append(
            {
                "experiment_tag": tag,
                "experiment_name": name,
                "nominal_kappa": m["nominal_kappa"],
                "linear_kappa": m["linear_kappa"],
                "quadratic_kappa": m["qwk"],
                "ordinal_gap": m["qwk"] - m["nominal_kappa"],
                "mae": m["mae"],
            }
        )
        ref_rows.append(
            {
                "experiment_tag": tag,
                "experiment_name": name,
                "tp": m["ref_tp"], "fp": m["ref_fp"],
                "fn": m["ref_fn"], "tn": m["ref_tn"],
                "sensitivity": m["referable_sensitivity"],
                "specificity": m["referable_specificity"],
                "ppv": m["referable_ppv"],
                "npv": m["referable_npv"],
                "balanced_accuracy": 0.5 * (m["referable_sensitivity"] + m["referable_specificity"]),
            }
        )
        print(f"  [ok] {tag:22s} qwk={m['qwk']:.4f} acc={m['accuracy']:.4f} mF1={m['macro_f1']:.4f}")

    full_df = pd.DataFrame(full_rows)
    full_df.to_csv(OUT_DIR / "appendix_a_full_metrics.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(OUT_DIR / "appendix_a_per_class_f1.csv", index=False)
    pd.DataFrame(kappa_rows).to_csv(OUT_DIR / "appendix_a_kappa_split.csv", index=False)
    pd.DataFrame(ref_rows).to_csv(OUT_DIR / "appendix_a_referable_binary.csv", index=False)

    # --- ablation paired bootstrap ---
    ab_rows: list[dict] = []
    ladder_tags = [t for _, t in ABLATION_LADDER if t in preds_cache]
    baseline_tag = ladder_tags[0] if ladder_tags else None
    for i in range(1, len(ladder_tags)):
        prev_tag, curr_tag = ladder_tags[i - 1], ladder_tags[i]
        yt_a, yp_a = preds_cache[curr_tag]
        yt_b, yp_b = preds_cache[prev_tag]
        if len(yt_a) != len(yt_b) or not np.array_equal(yt_a, yt_b):
            print(f"  [warn] true-label mismatch between {curr_tag} and {prev_tag}; skipped")
            continue
        results = _paired_bootstrap(yt_a, yp_a, yp_b)
        for r in results:
            ab_rows.append(
                {"comparison_type": "step_over_previous",
                 "A": curr_tag, "B": prev_tag, **r}
            )
        # also vs baseline
        if baseline_tag and curr_tag != baseline_tag:
            yt_b, yp_b = preds_cache[baseline_tag]
            if np.array_equal(yt_a, yt_b):
                results_v = _paired_bootstrap(yt_a, yp_a, yp_b)
                for r in results_v:
                    ab_rows.append(
                        {"comparison_type": "step_over_baseline",
                         "A": curr_tag, "B": baseline_tag, **r}
                    )
    pd.DataFrame(ab_rows).to_csv(OUT_DIR / "appendix_a_ablation_bootstrap.csv", index=False)

    # --- external Messidor-2 ---
    ext_rows = []
    for summary in sorted(glob.glob(str(RESULTS_DIR / "messidor2_eval/*/messidor2_summary.txt"))):
        ext_rows.append(_parse_messidor_summary(Path(summary)))
    if ext_rows:
        pd.DataFrame(ext_rows).to_csv(OUT_DIR / "appendix_a_external_messidor2.csv", index=False)

    print("\nWrote:")
    for f in [
        "appendix_a_full_metrics.csv",
        "appendix_a_per_class_f1.csv",
        "appendix_a_kappa_split.csv",
        "appendix_a_referable_binary.csv",
        "appendix_a_ablation_bootstrap.csv",
        "appendix_a_external_messidor2.csv",
    ]:
        print(f"  {OUT_DIR / f}")


if __name__ == "__main__":
    main()
