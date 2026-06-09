from __future__ import annotations

import logging
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import ExpConfig
from src.dataset import DRDataset
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

logger = logging.getLogger(__name__)

ENSEMBLE_BACKBONES = ["resnet50", "efficientnet_b4", "convnext_small"]


def run_ensemble_inference(
    configs: list[ExpConfig],
    test_dataset: DRDataset,
    test_loader: DataLoader,
    device: torch.device,
    val_loader: DataLoader | None = None,
) -> dict[str, float]:
    all_model_preds: list[np.ndarray] = []
    base_cfg = configs[0]

    for cfg in configs:
        ckpt_path = cfg.ckpt_dir / f"{cfg.exp_name}_best.pth"
        if not ckpt_path.exists():
            swa_path = cfg.ckpt_dir / f"{cfg.exp_name}_swa.pth"
            pseudo_path = cfg.ckpt_dir / f"{cfg.exp_name}_pseudo.pth"
            if pseudo_path.exists():
                ckpt_path = pseudo_path
            elif swa_path.exists():
                ckpt_path = swa_path
            else:
                logger.warning(f"No checkpoint found for {cfg.exp_name}, skipping")
                continue

        model = build_model(cfg).to(device)
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        model.eval()

        if cfg.use_tta:
            preds, codes = predict_with_tta(model, test_dataset, device)
        else:
            preds, _, codes = predict_no_tta(model, test_loader, device, is_regression=cfg.is_regression)

        all_model_preds.append(preds)
        logger.info(f"Loaded {cfg.exp_name} ({cfg.backbone}) predictions")

    if not all_model_preds:
        raise RuntimeError("No models produced predictions for ensemble")

    avg_preds = np.mean(all_model_preds, axis=0)

    targets = np.array([test_dataset.samples[i][1] for i in range(len(test_dataset))])

    thresholds = base_cfg.default_thresholds
    if base_cfg.use_optimized_thresholds and val_loader is not None:
        val_all_preds: list[np.ndarray] = []
        for cfg in configs:
            ckpt_path = cfg.ckpt_dir / f"{cfg.exp_name}_best.pth"
            if not ckpt_path.exists():
                for alt in ["_pseudo.pth", "_swa.pth"]:
                    alt_path = cfg.ckpt_dir / f"{cfg.exp_name}{alt}"
                    if alt_path.exists():
                        ckpt_path = alt_path
                        break
            if not ckpt_path.exists():
                continue
            model = build_model(cfg).to(device)
            model.load_state_dict(torch.load(ckpt_path, weights_only=True))
            model.eval()

            val_ds: DRDataset = val_loader.dataset  # type: ignore[assignment]
            if cfg.use_tta:
                vp, _ = predict_with_tta(model, val_ds, device)
            else:
                vp, _, _ = predict_no_tta(model, val_loader, device, is_regression=cfg.is_regression)
            val_all_preds.append(vp)

        if val_all_preds:
            val_avg = np.mean(val_all_preds, axis=0)
            val_targets = np.array([val_ds.samples[i][1] for i in range(len(val_ds))])
            rounder = OptimizedRounder()
            thresholds = rounder.fit(val_avg, val_targets.astype(int))
            logger.info(f"Ensemble optimized thresholds: {thresholds}")

    pred_classes = regression_to_class(avg_preds, thresholds)
    targets_int = targets.astype(int)
    metrics = compute_metrics(targets_int, pred_classes)

    results_dir = base_cfg.results_dir.parent / f"exp{base_cfg.exp_id:02d}_ensemble"
    results_dir.mkdir(parents=True, exist_ok=True)
    exp_name = f"exp{base_cfg.exp_id:02d}_ensemble"

    save_confusion_matrix(targets_int, pred_classes, results_dir / f"{exp_name}_cm.png")
    save_classification_report(targets_int, pred_classes, results_dir / f"{exp_name}_cls_report.txt")

    codes_list = [test_dataset.samples[i][0] for i in range(len(test_dataset))]
    save_predictions(codes_list, avg_preds, pred_classes, targets_int, results_dir / f"{exp_name}_preds.csv")

    logger.info(
        f"Ensemble Results | QWK: {metrics['qwk']:.4f} | "
        f"Macro F1: {metrics['macro_f1']:.4f} | Acc: {metrics['accuracy']:.4f}"
    )
    return metrics


def build_ensemble_configs(base_cfg: ExpConfig) -> list[ExpConfig]:
    configs: list[ExpConfig] = []
    for backbone in ENSEMBLE_BACKBONES:
        cfg = deepcopy(base_cfg)
        cfg.backbone = backbone
        cfg.name = f"ensemble_{backbone}"
        configs.append(cfg)
    return configs
