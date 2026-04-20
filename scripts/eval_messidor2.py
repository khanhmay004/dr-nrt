"""Zero-shot evaluation of APTOS-trained models on Messidor-2.

Evaluates D1/H1/ensemble on Messidor-2 (preprocessed) without any fine-tuning.
Reports both 5-grade metrics (QWK, MacroF1, per-class F1) and binary referable
(grade ≥ 2) metrics (Sensitivity, Specificity, PPV, NPV).

Usage:
    python scripts/eval_messidor2.py --exp 300 --device cuda
    python scripts/eval_messidor2.py --exp 701 --device cuda
    python scripts/eval_messidor2.py --exp 300 701 --ensemble --device cuda
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    f1_score,
    accuracy_score,
)
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT_DIR, get_config, IMAGE_SIZE, CLASS_NAMES, NUM_CLASSES
from src.models import build_model
from src.transforms import get_tta_transforms
from src.evaluate import save_confusion_matrix

import cv2
import albumentations as A

# ── Paths ────────────────────────────────────────────────────────────────────
MESSIDOR2_IMG_DIR = ROOT_DIR / "data" / "messidor2_processed"
MESSIDOR2_CSV = ROOT_DIR / "data" / "messidor2_labels.csv"
OUTPUT_DIR = ROOT_DIR / "results" / "messidor2_eval"


# ── Dataset ──────────────────────────────────────────────────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Messidor2Dataset(Dataset):
    """Messidor-2 dataset for evaluation."""

    def __init__(self, img_dir: Path, csv_path: Path, transform: A.Compose | None = None):
        self.img_dir = img_dir
        self.transform = transform

        # Load labels
        self.samples = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append({
                    "id_code": row["id_code"],
                    "diagnosis": int(row["diagnosis"]),
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        id_code = sample["id_code"]
        label = sample["diagnosis"]

        img_path = self.img_dir / f"{id_code}.png"
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]

        # Normalize
        image = image.astype(np.float32) / 255.0
        for c in range(3):
            image[..., c] = (image[..., c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        image = image.transpose(2, 0, 1)

        return torch.from_numpy(image).float(), label, id_code


def predict_with_tta(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 32,
    use_tta: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Run inference with optional TTA. Returns (probs, targets, id_codes)."""
    model.eval()
    tta_transforms = get_tta_transforms() if use_tta else [A.Compose([])]

    all_probs = []
    all_targets = []
    all_codes = []

    for sample_idx in tqdm(range(len(dataset)), desc="Evaluating"):
        # Get raw image (before any transform)
        sample = dataset.samples[sample_idx]
        id_code = sample["id_code"]
        label = sample["diagnosis"]

        img_path = MESSIDOR2_IMG_DIR / f"{id_code}.png"
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # TTA: average over transforms
        tta_probs = []
        for tta_transform in tta_transforms:
            aug_image = tta_transform(image=image)["image"]

            # Normalize
            norm_image = aug_image.astype(np.float32) / 255.0
            for c in range(3):
                norm_image[..., c] = (norm_image[..., c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
            tensor = torch.from_numpy(norm_image.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                tta_probs.append(probs)

        avg_probs = np.mean(tta_probs, axis=0)
        all_probs.append(avg_probs)
        all_targets.append(label)
        all_codes.append(id_code)

    return np.array(all_probs), np.array(all_targets), all_codes


def compute_binary_metrics(targets: np.ndarray, preds: np.ndarray) -> dict:
    """Compute binary referable (grade >= 2) vs non-referable metrics."""
    # Binary: 0-1 = non-referable (0), 2-4 = referable (1)
    binary_targets = (targets >= 2).astype(int)
    binary_preds = (preds >= 2).astype(int)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(binary_targets, binary_preds)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    balanced_acc = (sensitivity + specificity) / 2
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return {
        "binary_sensitivity": sensitivity,
        "binary_specificity": specificity,
        "binary_ppv": ppv,
        "binary_npv": npv,
        "binary_balanced_acc": balanced_acc,
        "binary_accuracy": accuracy,
        "binary_tp": tp,
        "binary_tn": tn,
        "binary_fp": fp,
        "binary_fn": fn,
    }


def compute_5grade_metrics(targets: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> dict:
    """Compute 5-grade classification metrics."""
    qwk = cohen_kappa_score(targets, preds, weights="quadratic")
    accuracy = accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average="macro")

    # Per-class F1
    f1_per_class = f1_score(targets, preds, average=None, labels=list(range(NUM_CLASSES)))
    metrics = {
        "qwk": qwk,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }
    for i, name in enumerate(CLASS_NAMES):
        metrics[f"f1_{name}"] = f1_per_class[i] if i < len(f1_per_class) else 0.0

    # Per-class sensitivity and specificity
    cm = confusion_matrix(targets, preds, labels=list(range(NUM_CLASSES)))
    for i, name in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics[f"sens_{name}"] = sens
        metrics[f"spec_{name}"] = spec

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot Messidor-2 evaluation")
    parser.add_argument("--exp", type=int, nargs="+", required=True,
                        help="Experiment ID(s) to evaluate")
    parser.add_argument("--ensemble", action="store_true",
                        help="Ensemble multiple models (average probs)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tta", action="store_true", default=True,
                        help="Use test-time augmentation (default: True)")
    parser.add_argument("--no-tta", dest="tta", action="store_false")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Check preprocessed data exists
    if not MESSIDOR2_IMG_DIR.exists() or not MESSIDOR2_CSV.exists():
        print("ERROR: Messidor-2 not preprocessed. Run first:")
        print("  python scripts/preprocess_messidor2.py")
        sys.exit(1)

    # Load dataset
    dataset = Messidor2Dataset(MESSIDOR2_IMG_DIR, MESSIDOR2_CSV)
    print(f"Messidor-2 dataset: {len(dataset)} images")

    # Grade distribution
    from collections import Counter
    grade_dist = Counter(s["diagnosis"] for s in dataset.samples)
    print("Grade distribution:")
    for g in sorted(grade_dist):
        print(f"  Grade {g}: {grade_dist[g]}")

    # Output directory
    if args.ensemble and len(args.exp) > 1:
        exp_str = "_".join(str(e) for e in sorted(args.exp))
        out_dir = OUTPUT_DIR / f"ensemble_{exp_str}"
    else:
        out_dir = OUTPUT_DIR / f"exp{args.exp[0]}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model(s) and get predictions
    all_model_probs = []
    for exp_id in args.exp:
        cfg = get_config(exp_id)
        ckpt_path = cfg.ckpt_dir / f"{cfg.exp_name}_best.pth"

        if not ckpt_path.exists():
            print(f"ERROR: Checkpoint not found: {ckpt_path}")
            sys.exit(1)

        print(f"\nLoading exp{exp_id}: {ckpt_path}")
        model = build_model(cfg).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()

        probs, targets, codes = predict_with_tta(model, dataset, device, use_tta=args.tta)
        all_model_probs.append(probs)

        del model
        torch.cuda.empty_cache()

    # Ensemble or single
    if args.ensemble and len(all_model_probs) > 1:
        print(f"\nEnsembling {len(all_model_probs)} models (equal weights)")
        final_probs = np.mean(all_model_probs, axis=0)
    else:
        final_probs = all_model_probs[0]

    preds = final_probs.argmax(axis=1)

    # ── 5-grade metrics ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("5-GRADE METRICS (Zero-shot on Messidor-2)")
    print("=" * 60)

    metrics_5g = compute_5grade_metrics(targets, preds, final_probs)
    print(f"  QWK:        {metrics_5g['qwk']:.4f}")
    print(f"  Accuracy:   {metrics_5g['accuracy']:.4f}")
    print(f"  Macro F1:   {metrics_5g['macro_f1']:.4f}")
    for name in CLASS_NAMES:
        print(f"  F1 {name}: {metrics_5g[f'f1_{name}']:.4f}")

    # ── Binary referable metrics ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BINARY REFERABLE METRICS (grade >= 2)")
    print("=" * 60)

    metrics_bin = compute_binary_metrics(targets, preds)
    print(f"  Sensitivity: {metrics_bin['binary_sensitivity']:.4f}")
    print(f"  Specificity: {metrics_bin['binary_specificity']:.4f}")
    print(f"  PPV:         {metrics_bin['binary_ppv']:.4f}")
    print(f"  NPV:         {metrics_bin['binary_npv']:.4f}")
    print(f"  Balanced Acc: {metrics_bin['binary_balanced_acc']:.4f}")
    print(f"  (TP={metrics_bin['binary_tp']}, TN={metrics_bin['binary_tn']}, "
          f"FP={metrics_bin['binary_fp']}, FN={metrics_bin['binary_fn']})")

    # ── Save outputs ─────────────────────────────────────────────────────────
    # Confusion matrix
    cm = confusion_matrix(targets, preds, labels=list(range(NUM_CLASSES)))
    save_confusion_matrix(targets, preds, out_dir / "messidor2_cm.png")

    # Classification report
    report = classification_report(targets, preds, target_names=CLASS_NAMES, digits=4)
    with open(out_dir / "messidor2_cls_report.txt", "w") as f:
        f.write(report)
    print(f"\nClassification report:\n{report}")

    # Predictions CSV
    import pandas as pd
    df = pd.DataFrame({
        "id_code": codes,
        "true_label": targets,
        "pred_label": preds,
        "correct": (targets == preds).astype(int),
    })
    for i, name in enumerate(CLASS_NAMES):
        df[f"p_{name}"] = final_probs[:, i].round(4)
    df.to_csv(out_dir / "messidor2_predictions.csv", index=False)

    # Summary
    all_metrics = {**metrics_5g, **metrics_bin}
    with open(out_dir / "messidor2_summary.txt", "w") as f:
        f.write(f"Zero-shot Messidor-2 Evaluation\n")
        f.write(f"Experiments: {args.exp}\n")
        f.write(f"Ensemble: {args.ensemble}\n")
        f.write(f"TTA: {args.tta}\n")
        f.write(f"Images: {len(dataset)}\n\n")
        f.write("=== 5-Grade Metrics ===\n")
        for k in ["qwk", "accuracy", "macro_f1"] + [f"f1_{n}" for n in CLASS_NAMES]:
            f.write(f"  {k}: {all_metrics[k]:.4f}\n")
        f.write("\n=== Binary Referable (grade >= 2) ===\n")
        for k in ["binary_sensitivity", "binary_specificity", "binary_ppv", "binary_npv", "binary_balanced_acc"]:
            f.write(f"  {k}: {all_metrics[k]:.4f}\n")
        f.write(f"  TP: {metrics_bin['binary_tp']}, TN: {metrics_bin['binary_tn']}, "
                f"FP: {metrics_bin['binary_fp']}, FN: {metrics_bin['binary_fn']}\n")

    print(f"\n✓ Results saved to {out_dir}")


if __name__ == "__main__":
    main()
