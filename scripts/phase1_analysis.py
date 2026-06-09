"""Phase 1 analysis: Error case study, Calibration reliability diagram, Clinical risk annotation.

Generates:
1. C1: Error case study — 10 representative failure images with Grad-CAM
2. C3: Clinical risk annotation of confusion matrix
3. C6: Calibration reliability diagram (before/after temperature scaling)

Usage:
    python scripts/phase1_analysis.py --exp 300 --device cuda
"""
from __future__ import annotations

import argparse
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT_DIR, get_config, CLASS_NAMES, NUM_CLASSES, IMAGE_SIZE
from src.models import build_model
from src.dataset import ben_graham_preprocess
from src.analysis.explainers import gradcam, image_to_tensor, tensor_to_display_rgb


OUTPUT_DIR = ROOT_DIR / "results" / "phase1_analysis"
APTOS_IMG_DIR = ROOT_DIR / "data" / "train_images"


# =============================================================================
# C1: Error Case Study with Grad-CAM
# =============================================================================


def get_failure_cases(preds_csv: Path, n_cases: int = 10) -> pd.DataFrame:
    """Get representative failure cases across different error types."""
    df = pd.read_csv(preds_csv)

    # Handle both pred_label (regular) and mc_prediction (MC Dropout) column names
    pred_col = "pred_label" if "pred_label" in df.columns else "mc_prediction"

    errors = df[df["true_label"] != df[pred_col]].copy()

    if len(errors) == 0:
        print("No errors found!")
        return pd.DataFrame()

    # Standardize column name
    if pred_col == "mc_prediction":
        errors["pred_label"] = errors["mc_prediction"]

    # Categorize errors
    errors["error_type"] = errors.apply(
        lambda r: categorize_error(r["true_label"], r["pred_label"]), axis=1
    )

    # Sample diverse errors
    selected = []
    error_types = errors["error_type"].unique()

    # Prioritize clinically significant errors
    priority_types = [
        "severe_undergrade",  # Severe/PDR predicted as lower
        "moderate_undergrade",  # Moderate predicted as Mild/NoDR
        "severe_overgrade",  # Lower grades predicted as Severe/PDR
        "mild_confusion",  # Mild <-> Moderate
        "adjacent_error",  # Off-by-one errors
    ]

    for etype in priority_types:
        subset = errors[errors["error_type"] == etype]
        if len(subset) > 0:
            n_take = min(2, len(subset), n_cases - len(selected))
            selected.extend(subset.sample(n=n_take, random_state=42).to_dict("records"))
            if len(selected) >= n_cases:
                break

    # Fill remaining with random errors
    remaining = n_cases - len(selected)
    if remaining > 0:
        already_selected = {r["id_code"] for r in selected}
        remaining_errors = errors[~errors["id_code"].isin(already_selected)]
        if len(remaining_errors) > 0:
            selected.extend(
                remaining_errors.sample(
                    n=min(remaining, len(remaining_errors)), random_state=42
                ).to_dict("records")
            )

    return pd.DataFrame(selected)


def categorize_error(true_label: int, pred_label: int) -> str:
    """Categorize error by clinical significance."""
    diff = pred_label - true_label

    # Undergrading (predicted less severe than actual) — clinically dangerous
    if true_label >= 3 and pred_label < true_label:
        return "severe_undergrade"
    if true_label == 2 and pred_label < 2:
        return "moderate_undergrade"

    # Overgrading (predicted more severe than actual) — over-referral
    if true_label < 2 and pred_label >= 3:
        return "severe_overgrade"

    # Adjacent confusion
    if abs(diff) == 1:
        if true_label == 1 or pred_label == 1:
            return "mild_confusion"
        return "adjacent_error"

    return "other"


def generate_gradcam_visualization(
    model: nn.Module,
    image_path: Path,
    true_label: int,
    pred_label: int,
    device: torch.device,
    save_path: Path,
) -> None:
    """Generate Grad-CAM visualization for a single image."""
    # Load and preprocess image
    raw_image = cv2.imread(str(image_path))
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    processed = ben_graham_preprocess(raw_image, IMAGE_SIZE)
    tensor = image_to_tensor(processed, device)

    # Get target layer (ResNet-50 layer4)
    target_layer = model.backbone.layer4[-1]

    # Grad-CAM for predicted class
    cam_pred = gradcam(model, tensor, pred_label, target_layer, method="gradcam")

    # Grad-CAM for true class (if different)
    if true_label != pred_label:
        cam_true = gradcam(model, tensor, true_label, target_layer, method="gradcam")
    else:
        cam_true = cam_pred

    # Create visualization
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Original image
    axes[0].imshow(processed)
    axes[0].set_title("Original (Ben Graham)")
    axes[0].axis("off")

    # Grad-CAM for predicted class
    heatmap_pred = cv2.applyColorMap(
        (cam_pred * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_pred = cv2.cvtColor(heatmap_pred, cv2.COLOR_BGR2RGB)
    overlay_pred = cv2.addWeighted(processed, 0.5, heatmap_pred, 0.5, 0)
    axes[1].imshow(overlay_pred)
    axes[1].set_title(f"Grad-CAM: Predicted ({CLASS_NAMES[pred_label]})")
    axes[1].axis("off")

    # Grad-CAM for true class
    heatmap_true = cv2.applyColorMap(
        (cam_true * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_true = cv2.cvtColor(heatmap_true, cv2.COLOR_BGR2RGB)
    overlay_true = cv2.addWeighted(processed, 0.5, heatmap_true, 0.5, 0)
    axes[2].imshow(overlay_true)
    axes[2].set_title(f"Grad-CAM: True ({CLASS_NAMES[true_label]})")
    axes[2].axis("off")

    # Difference heatmap
    diff = np.abs(cam_pred - cam_true)
    axes[3].imshow(diff, cmap="coolwarm")
    axes[3].set_title("CAM Difference (Pred - True)")
    axes[3].axis("off")

    plt.suptitle(
        f"True: {CLASS_NAMES[true_label]} → Pred: {CLASS_NAMES[pred_label]}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_error_case_study(
    model: nn.Module, preds_csv: Path, device: torch.device, out_dir: Path
) -> None:
    """C1: Generate error case study with Grad-CAM."""
    print("\n" + "=" * 60)
    print("C1: Error Case Study with Grad-CAM")
    print("=" * 60)

    cases_dir = out_dir / "error_cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    failures = get_failure_cases(preds_csv, n_cases=10)
    if len(failures) == 0:
        print("No failure cases found.")
        return

    print(f"Selected {len(failures)} failure cases:")

    summary_rows = []
    for idx, row in failures.iterrows():
        id_code = row["id_code"]
        true_label = int(row["true_label"])
        pred_label = int(row["pred_label"]) if "pred_label" in row else int(row["mc_prediction"])
        error_type = row.get("error_type", "unknown")

        print(f"  {id_code}: {CLASS_NAMES[true_label]} → {CLASS_NAMES[pred_label]} ({error_type})")

        # Find image path
        img_path = APTOS_IMG_DIR / f"{id_code}.png"
        if not img_path.exists():
            print(f"    Warning: Image not found: {img_path}")
            continue

        save_path = cases_dir / f"{id_code}_gradcam.png"
        generate_gradcam_visualization(
            model, img_path, true_label, pred_label, device, save_path
        )

        summary_rows.append(
            {
                "id_code": id_code,
                "true_label": CLASS_NAMES[true_label],
                "pred_label": CLASS_NAMES[pred_label],
                "error_type": error_type,
                "clinical_risk": get_clinical_risk(true_label, pred_label),
            }
        )

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(cases_dir / "error_case_summary.csv", index=False)
    print(f"\n✓ Error case study saved to {cases_dir}")


def get_clinical_risk(true_label: int, pred_label: int) -> str:
    """Assess clinical risk of misclassification."""
    # Undergrading severe cases = HIGH risk (missed treatment)
    if true_label >= 3 and pred_label <= 1:
        return "HIGH - Severe DR missed as non-referable"
    if true_label >= 2 and pred_label <= 1:
        return "HIGH - Referable DR missed"

    # Overgrading = LOW risk (unnecessary referral, but safe)
    if true_label <= 1 and pred_label >= 2:
        return "LOW - Over-referral (safe but costly)"

    # Adjacent errors within referable
    if true_label >= 2 and pred_label >= 2:
        return "LOW - Within referable category"

    return "MEDIUM - Borderline case"


# =============================================================================
# C3: Clinical Risk Annotation of Confusion Matrix
# =============================================================================


def generate_clinical_risk_matrix(preds_csv: Path, out_dir: Path) -> None:
    """C3: Generate clinically-annotated confusion matrix."""
    print("\n" + "=" * 60)
    print("C3: Clinical Risk Annotation of Confusion Matrix")
    print("=" * 60)

    df = pd.read_csv(preds_csv)
    true_labels = df["true_label"].values
    pred_col = "pred_label" if "pred_label" in df.columns else "mc_prediction"
    pred_labels = df[pred_col].values

    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(NUM_CLASSES)))

    # Define clinical risk levels for each cell
    # Rows = True, Cols = Predicted
    risk_matrix = np.array(
        [
            # Pred: NoDR  Mild   Mod    Sev    PDR
            ["OK", "LOW", "LOW", "MED", "MED"],  # True: No DR
            ["LOW", "OK", "LOW", "MED", "MED"],  # True: Mild
            ["HIGH", "HIGH", "OK", "LOW", "LOW"],  # True: Moderate
            ["HIGH", "HIGH", "MED", "OK", "LOW"],  # True: Severe
            ["HIGH", "HIGH", "MED", "LOW", "OK"],  # True: Proliferative
        ]
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Custom colormap based on clinical risk
    colors = {"OK": "#2ecc71", "LOW": "#f1c40f", "MED": "#e67e22", "HIGH": "#e74c3c"}

    # Plot confusion matrix with risk coloring
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            count = cm[i, j]
            risk = risk_matrix[i, j]
            color = colors[risk]

            # Background color
            rect = plt.Rectangle(
                (j - 0.5, i - 0.5), 1, 1, facecolor=color, alpha=0.6, edgecolor="white", linewidth=2
            )
            ax.add_patch(rect)

            # Count text
            ax.text(
                j, i, f"{count}", ha="center", va="center", fontsize=14, fontweight="bold"
            )

            # Risk label (smaller, below count)
            if i != j and count > 0:
                ax.text(
                    j, i + 0.3, f"[{risk}]", ha="center", va="center", fontsize=9, color="gray"
                )

    ax.set_xlim(-0.5, NUM_CLASSES - 0.5)
    ax.set_ylim(NUM_CLASSES - 0.5, -0.5)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax.set_yticklabels(CLASS_NAMES, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix with Clinical Risk Annotation", fontsize=14, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=colors["OK"], alpha=0.6, label="OK - Correct"),
        Patch(facecolor=colors["LOW"], alpha=0.6, label="LOW - Over-referral / within-category"),
        Patch(facecolor=colors["MED"], alpha=0.6, label="MEDIUM - Borderline"),
        Patch(facecolor=colors["HIGH"], alpha=0.6, label="HIGH - Missed referable DR"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.02, 1))

    plt.tight_layout()
    plt.savefig(out_dir / "clinical_risk_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Generate text summary
    high_risk_count = 0
    high_risk_cases = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if risk_matrix[i, j] == "HIGH" and cm[i, j] > 0:
                high_risk_count += cm[i, j]
                high_risk_cases.append(
                    f"  {CLASS_NAMES[i]} → {CLASS_NAMES[j]}: {cm[i, j]} cases"
                )

    with open(out_dir / "clinical_risk_summary.txt", "w") as f:
        f.write("Clinical Risk Analysis of Confusion Matrix\n")
        f.write("=" * 50 + "\n\n")
        f.write("Risk Categories:\n")
        f.write("  OK   - Correct prediction\n")
        f.write("  LOW  - Over-referral or within-category error (safe)\n")
        f.write("  MED  - Borderline error\n")
        f.write("  HIGH - Missed referable DR (clinically dangerous)\n\n")
        f.write(f"HIGH-RISK ERRORS: {high_risk_count} total\n")
        for case in high_risk_cases:
            f.write(case + "\n")
        f.write(f"\nTotal test samples: {cm.sum()}\n")
        f.write(f"Total errors: {cm.sum() - np.trace(cm)}\n")
        f.write(f"High-risk error rate: {high_risk_count / cm.sum():.2%}\n")

    print(f"✓ Clinical risk matrix saved to {out_dir}")
    print(f"  HIGH-risk errors: {high_risk_count} ({high_risk_count / cm.sum():.2%})")


# =============================================================================
# C6: Calibration Reliability Diagram
# =============================================================================


def generate_calibration_diagram(preds_csv: Path, out_dir: Path) -> None:
    """C6: Generate calibration reliability diagram."""
    print("\n" + "=" * 60)
    print("C6: Calibration Reliability Diagram")
    print("=" * 60)

    df = pd.read_csv(preds_csv)
    true_labels = df["true_label"].values

    # Get probability columns
    prob_cols = [f"p_{name}" for name in CLASS_NAMES]
    if not all(col in df.columns for col in prob_cols):
        print("  Warning: Probability columns not found. Skipping calibration diagram.")
        return

    probs = df[prob_cols].values
    pred_labels = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    correct = (pred_labels == true_labels).astype(int)

    # Calibration curve
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_accs.append(correct[mask].mean())
            bin_confs.append(confidences[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_accs.append(np.nan)
            bin_confs.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_counts.append(0)

    bin_accs = np.array(bin_accs)
    bin_confs = np.array(bin_confs)
    bin_counts = np.array(bin_counts)

    # Expected Calibration Error
    valid_mask = ~np.isnan(bin_accs)
    ece = np.sum(
        bin_counts[valid_mask]
        * np.abs(bin_accs[valid_mask] - bin_confs[valid_mask])
    ) / bin_counts[valid_mask].sum()

    # Temperature scaling (simple optimization)
    def compute_ece_with_temp(temp: float) -> float:
        scaled_probs = np.exp(np.log(probs + 1e-10) / temp)
        scaled_probs = scaled_probs / scaled_probs.sum(axis=1, keepdims=True)
        scaled_conf = scaled_probs.max(axis=1)

        temp_bin_accs = []
        temp_bin_confs = []
        temp_bin_counts = []
        for i in range(n_bins):
            mask = (scaled_conf >= bin_edges[i]) & (scaled_conf < bin_edges[i + 1])
            if mask.sum() > 0:
                temp_bin_accs.append(correct[mask].mean())
                temp_bin_confs.append(scaled_conf[mask].mean())
                temp_bin_counts.append(mask.sum())
        if not temp_bin_accs:
            return 1.0
        temp_bin_accs = np.array(temp_bin_accs)
        temp_bin_confs = np.array(temp_bin_confs)
        temp_bin_counts = np.array(temp_bin_counts)
        return np.sum(temp_bin_counts * np.abs(temp_bin_accs - temp_bin_confs)) / temp_bin_counts.sum()

    # Find optimal temperature
    best_temp = 1.0
    best_ece = ece
    for temp in np.linspace(0.5, 3.0, 50):
        temp_ece = compute_ece_with_temp(temp)
        if temp_ece < best_ece:
            best_ece = temp_ece
            best_temp = temp

    # Recompute calibration with optimal temperature
    scaled_probs = np.exp(np.log(probs + 1e-10) / best_temp)
    scaled_probs = scaled_probs / scaled_probs.sum(axis=1, keepdims=True)
    scaled_conf = scaled_probs.max(axis=1)

    scaled_bin_accs = []
    scaled_bin_confs = []
    for i in range(n_bins):
        mask = (scaled_conf >= bin_edges[i]) & (scaled_conf < bin_edges[i + 1])
        if mask.sum() > 0:
            scaled_bin_accs.append(correct[mask].mean())
            scaled_bin_confs.append(scaled_conf[mask].mean())
        else:
            scaled_bin_accs.append(np.nan)
            scaled_bin_confs.append((bin_edges[i] + bin_edges[i + 1]) / 2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Before temperature scaling
    ax = axes[0]
    ax.bar(
        bin_confs,
        bin_counts / bin_counts.sum(),
        width=0.08,
        alpha=0.3,
        color="blue",
        label="Sample density",
    )
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    valid = ~np.isnan(bin_accs)
    ax.plot(bin_confs[valid], bin_accs[valid], "ro-", markersize=8, label="Model calibration")
    ax.fill_between(
        bin_confs[valid],
        bin_confs[valid],
        bin_accs[valid],
        alpha=0.2,
        color="red",
        label=f"ECE = {ece:.4f}",
    )
    ax.set_xlabel("Mean Predicted Confidence", fontsize=12)
    ax.set_ylabel("Fraction of Positives (Accuracy)", fontsize=12)
    ax.set_title("Before Temperature Scaling", fontsize=13)
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # After temperature scaling
    ax = axes[1]
    ax.bar(
        np.array(scaled_bin_confs),
        bin_counts / bin_counts.sum(),
        width=0.08,
        alpha=0.3,
        color="blue",
        label="Sample density",
    )
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    valid = ~np.isnan(np.array(scaled_bin_accs))
    ax.plot(
        np.array(scaled_bin_confs)[valid],
        np.array(scaled_bin_accs)[valid],
        "go-",
        markersize=8,
        label="Model calibration",
    )
    ax.fill_between(
        np.array(scaled_bin_confs)[valid],
        np.array(scaled_bin_confs)[valid],
        np.array(scaled_bin_accs)[valid],
        alpha=0.2,
        color="green",
        label=f"ECE = {best_ece:.4f}",
    )
    ax.set_xlabel("Mean Predicted Confidence", fontsize=12)
    ax.set_ylabel("Fraction of Positives (Accuracy)", fontsize=12)
    ax.set_title(f"After Temperature Scaling (T = {best_temp:.2f})", fontsize=13)
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Calibration Reliability Diagram", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "calibration_reliability_diagram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save summary
    with open(out_dir / "calibration_summary.txt", "w") as f:
        f.write("Calibration Analysis\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"ECE (before temp scaling): {ece:.4f}\n")
        f.write(f"Optimal temperature:       {best_temp:.2f}\n")
        f.write(f"ECE (after temp scaling):  {best_ece:.4f}\n")
        f.write(f"ECE reduction:             {ece - best_ece:.4f} ({(ece - best_ece) / ece * 100:.1f}%)\n")

    print(f"✓ Calibration diagram saved to {out_dir}")
    print(f"  ECE before: {ece:.4f}")
    print(f"  ECE after (T={best_temp:.2f}): {best_ece:.4f}")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 Analysis")
    parser.add_argument("--exp", type=int, default=300, help="Experiment ID")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--preds-csv",
        type=str,
        default="",
        help="Path to predictions CSV (default: use MC dropout predictions)",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = get_config(args.exp)

    out_dir = OUTPUT_DIR / f"exp{args.exp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find predictions CSV
    if args.preds_csv:
        preds_csv = Path(args.preds_csv)
    else:
        # Try MC dropout predictions first, then regular predictions
        mc_preds = cfg.results_dir / "mc_dropout" / "mc_dropout_predictions.csv"
        regular_preds = cfg.results_dir / f"{cfg.exp_name}_preds.csv"
        if mc_preds.exists():
            preds_csv = mc_preds
        elif regular_preds.exists():
            preds_csv = regular_preds
        else:
            print(f"ERROR: No predictions found for exp{args.exp}")
            print(f"  Tried: {mc_preds}")
            print(f"  Tried: {regular_preds}")
            sys.exit(1)

    print(f"Using predictions: {preds_csv}")

    # Load model for Grad-CAM
    ckpt_path = cfg.ckpt_dir / f"{cfg.exp_name}_best.pth"
    if ckpt_path.exists():
        print(f"Loading model: {ckpt_path}")
        model = build_model(cfg).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()

        # C1: Error case study with Grad-CAM
        run_error_case_study(model, preds_csv, device, out_dir)
    else:
        print(f"Warning: Checkpoint not found: {ckpt_path}")
        print("  Skipping error case study (requires model for Grad-CAM)")

    # C3: Clinical risk annotation
    generate_clinical_risk_matrix(preds_csv, out_dir)

    # C6: Calibration reliability diagram
    generate_calibration_diagram(preds_csv, out_dir)

    print(f"\n✓ All Phase 1 analysis complete. Results in: {out_dir}")


if __name__ == "__main__":
    main()
