from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import CLASS_NAMES, NUM_CLASSES, ExpConfig
from src.evaluate import (
    OptimizedRounder,
    compute_metrics,
    quadratic_weighted_kappa,
    regression_to_class,
    save_classification_report,
    save_confusion_matrix,
    save_predictions,
    save_training_curves,
)
from src.models import build_model, freeze_backbone, unfreeze_all
from src.losses import build_loss
from src.tta import predict_no_tta, predict_with_tta

logger = logging.getLogger(__name__)


def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def cutmix_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    _, _, h, w = x.shape
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    cx = np.random.randint(w)
    cy = np.random.randint(h)

    x1 = max(0, cx - cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    x2 = min(w, cx + cut_w // 2)
    y2 = min(h, cy + cut_h // 2)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    lam = 1 - (x2 - x1) * (y2 - y1) / (w * h)
    return mixed_x, y, y[index], lam


def compute_mixed_loss(
    criterion: nn.Module,
    outputs: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
    is_regression: bool,
) -> torch.Tensor:
    if is_regression:
        outputs = outputs.squeeze(1)
        return lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)

    return lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: ExpConfig,
    epoch: int,
    total_epochs: int,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=True)
    for batch in pbar:
        images, targets = batch[0].to(device), batch[1].to(device)
        sample_weights = batch[3].to(device) if len(batch) > 3 else None

        use_mix = cfg.use_mixup or cfg.use_cutmix
        if use_mix and (cfg.use_mixup and cfg.use_cutmix):
            if np.random.rand() < 0.5:
                images, y_a, y_b, lam = mixup_data(images, targets, cfg.mixup_alpha)
            else:
                images, y_a, y_b, lam = cutmix_data(images, targets, cfg.cutmix_alpha)
        elif cfg.use_mixup:
            images, y_a, y_b, lam = mixup_data(images, targets, cfg.mixup_alpha)
        elif cfg.use_cutmix:
            images, y_a, y_b, lam = cutmix_data(images, targets, cfg.cutmix_alpha)
        else:
            y_a, y_b, lam = targets, targets, 1.0

        outputs = model(images)
        loss = compute_mixed_loss(criterion, outputs, y_a, y_b, lam, cfg.is_regression)

        if sample_weights is not None:
            loss = (loss * sample_weights).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        if not (cfg.use_mixup or cfg.use_cutmix):
            if cfg.is_regression:
                preds = torch.round(outputs.squeeze(1)).clamp(0, NUM_CLASSES - 1).long()
                correct += (preds == targets.long()).sum().item()
            else:
                correct += (outputs.argmax(1) == targets).sum().item()

        avg_loss = total_loss / total_samples
        pbar.set_postfix(loss=f"{avg_loss:.4f}")

    avg_loss = total_loss / max(total_samples, 1)
    acc = correct / max(total_samples, 1) if not (cfg.use_mixup or cfg.use_cutmix) else 0.0
    return avg_loss, acc


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    cfg: ExpConfig,
    epoch: int,
    total_epochs: int,
) -> tuple[float, float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    all_preds: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Val]  ", leave=True)
    for batch in pbar:
        images, targets = batch[0].to(device), batch[1].to(device)
        outputs = model(images)

        if cfg.is_regression:
            loss = criterion(outputs.squeeze(1), targets)
            raw = outputs.squeeze(1).cpu().numpy()
            all_preds.append(raw)
            preds_cls = torch.round(outputs.squeeze(1)).clamp(0, NUM_CLASSES - 1).long()
            correct += (preds_cls == targets.long()).sum().item()
        else:
            loss = criterion(outputs, targets)
            all_preds.append(outputs.argmax(dim=1).cpu().numpy())
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probs.append(probs)
            correct += (outputs.argmax(1) == targets).sum().item()

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        all_targets.append(targets.cpu().numpy())

        avg_loss = total_loss / total_samples
        pbar.set_postfix(loss=f"{avg_loss:.4f}")

    val_loss = total_loss / max(total_samples, 1)
    val_acc = correct / max(total_samples, 1)
    preds = np.concatenate(all_preds)
    targets_np = np.concatenate(all_targets)

    y_pred_probs = np.concatenate(all_probs) if all_probs else None

    if cfg.is_regression:
        pred_classes = regression_to_class(preds)
        targets_int = targets_np.astype(int)
    else:
        pred_classes = preds
        targets_int = targets_np

    metrics = compute_metrics(targets_int, pred_classes, y_pred_probs)
    return val_loss, val_acc, metrics


def run_training(cfg: ExpConfig, train_loader: DataLoader, val_loader: DataLoader, device: torch.device) -> nn.Module:
    model = build_model(cfg).to(device)
    criterion = build_loss(cfg, device)

    freeze_backbone(model, cfg.backbone)
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(head_params, lr=cfg.lr_head, weight_decay=cfg.weight_decay)

    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    log_path = cfg.results_dir / f"{cfg.exp_name}_log.csv"
    log_rows: list[dict[str, float]] = []
    best_qwk = -1.0

    swa_model: AveragedModel | None = None
    if cfg.use_swa:
        swa_model = AveragedModel(model).to(device)

    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        total = torch.cuda.get_device_properties(device).total_memory / 1024**3
        logger.info(
            f"VRAM before training — Total: {total:.2f} GB | "
            f"Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB | "
            f"Free (approx): {total - reserved:.2f} GB"
        )

    for epoch in range(1, cfg.total_epochs + 1):
        if epoch == cfg.freeze_epochs + 1:
            unfreeze_all(model)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr_finetune, weight_decay=cfg.weight_decay)

            if cfg.scheduler == "step":
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=cfg.step_size, gamma=cfg.step_gamma,
                )
            elif cfg.scheduler == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=cfg.cosine_t0, T_mult=cfg.cosine_tmult, eta_min=cfg.cosine_eta_min,
                )

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, cfg, epoch, cfg.total_epochs,
        )

        if cfg.use_swa and swa_model is not None and epoch >= cfg.swa_start_epoch:
            swa_model.update_parameters(model)

        val_loss, val_acc, val_metrics = validate(
            model, val_loader, criterion, device, cfg, epoch, cfg.total_epochs,
        )
        current_lr = optimizer.param_groups[0]["lr"]

        log_row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_qwk": round(val_metrics["qwk"], 6),
            "val_macro_f1": round(val_metrics["macro_f1"], 6),
            "lr": current_lr,
        }
        log_rows.append(log_row)

        # --- Epoch summary ---
        print(f"\nEpoch {epoch}/{cfg.total_epochs}")
        print(f"    Train  \u2014 loss: {train_loss:.4f}  acc: {train_acc:.4f}")
        print(f"    Val    \u2014 loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        qwk = val_metrics['qwk']
        auc = val_metrics.get('auc_roc', 0.0)
        sens = val_metrics.get('sensitivity', 0.0)
        spec = val_metrics.get('specificity', 0.0)
        mf1 = val_metrics['macro_f1']
        print(
            f"    Val \u03ba: {qwk:.4f}  AUC: {auc:.4f}  "
            f"Sens: {sens:.4f}  Spec: {spec:.4f}  "
            f"F1-macro: {mf1:.4f}  LR: {current_lr:.2e}"
        )

        f1_parts = []
        short_names = ["No D", "Mild", "Mode", "Seve", "PDR"]
        for sn, cn in zip(short_names, CLASS_NAMES):
            f1_parts.append(f"{sn}={val_metrics.get(f'f1_{cn}', 0.0):.3f}")
        print(f"    Per-class F1:  {'  '.join(f1_parts)}")

        if val_metrics["qwk"] > best_qwk:
            best_qwk = val_metrics["qwk"]
            torch.save(model.state_dict(), cfg.ckpt_dir / f"{cfg.exp_name}_best.pth")
            print(f"  New best model saved (kappa = {best_qwk:.4f})")

        if scheduler is not None and epoch > cfg.freeze_epochs:
            scheduler.step()

    torch.save(model.state_dict(), cfg.ckpt_dir / f"{cfg.exp_name}_last.pth")

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(log_path, index=False)
    save_training_curves(log_df, cfg.results_dir / f"{cfg.exp_name}_curves.png")

    if cfg.use_swa and swa_model is not None:
        update_bn(train_loader, swa_model, device=device)
        torch.save(
            swa_model.module.state_dict(),
            cfg.ckpt_dir / f"{cfg.exp_name}_swa.pth",
        )
        return swa_model.module  # type: ignore[return-value]

    model.load_state_dict(torch.load(cfg.ckpt_dir / f"{cfg.exp_name}_best.pth", weights_only=True))
    return model


def evaluate_on_test(
    model: nn.Module,
    test_dataset: object,
    test_loader: DataLoader,
    cfg: ExpConfig,
    device: torch.device,
    val_loader: DataLoader | None = None,
) -> dict[str, float]:
    from src.dataset import DRDataset

    assert isinstance(test_dataset, DRDataset)

    if cfg.is_regression:
        if cfg.use_tta:
            raw_preds, codes = predict_with_tta(model, test_dataset, device)
            targets = np.array([test_dataset.samples[i][1] for i in range(len(test_dataset))])
        else:
            raw_preds, targets, codes = predict_no_tta(model, test_loader, device, is_regression=True)

        thresholds = cfg.default_thresholds
        if cfg.use_optimized_thresholds and val_loader is not None:
            rounder = OptimizedRounder()
            if cfg.use_tta:
                val_ds: DRDataset = val_loader.dataset  # type: ignore[assignment]
                val_preds, _ = predict_with_tta(model, val_ds, device)
                val_targets = np.array([val_ds.samples[i][1] for i in range(len(val_ds))])
            else:
                val_preds, val_targets, _ = predict_no_tta(model, val_loader, device, is_regression=True)
            thresholds = rounder.fit(val_preds, val_targets.astype(int))
            logger.info(f"Optimized thresholds: {thresholds}")

        pred_classes = regression_to_class(raw_preds, thresholds)
        targets_int = targets.astype(int)
    else:
        raw_preds, targets, codes = predict_no_tta(model, test_loader, device, is_regression=False)
        pred_classes = raw_preds.argmax(axis=1) if raw_preds.ndim == 2 else raw_preds
        targets_int = targets.astype(int)

    metrics = compute_metrics(targets_int, pred_classes)

    save_confusion_matrix(
        targets_int, pred_classes,
        cfg.results_dir / f"{cfg.exp_name}_cm.png",
    )
    save_classification_report(
        targets_int, pred_classes,
        cfg.results_dir / f"{cfg.exp_name}_cls_report.txt",
    )

    raw_for_csv = raw_preds if cfg.is_regression else pred_classes.astype(float)
    save_predictions(
        codes, raw_for_csv, pred_classes, targets_int,
        cfg.results_dir / f"{cfg.exp_name}_preds.csv",
    )

    logger.info(
        f"Test Results | QWK: {metrics['qwk']:.4f} | "
        f"Macro F1: {metrics['macro_f1']:.4f} | "
        f"Acc: {metrics['accuracy']:.4f}"
    )
    return metrics
