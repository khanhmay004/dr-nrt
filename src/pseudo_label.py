from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import TEST_IMG_DIR, ExpConfig
from src.dataset import DRDataset, PseudoLabelDataset, load_labels
from src.losses import build_loss
from src.tta import predict_with_tta, predict_no_tta

logger = logging.getLogger(__name__)


def generate_pseudo_labels(
    model: nn.Module,
    test_dataset: DRDataset,
    test_loader: DataLoader,
    cfg: ExpConfig,
    device: torch.device,
) -> dict[str, float]:
    if cfg.use_tta:
        raw_preds, codes = predict_with_tta(model, test_dataset, device)
    else:
        raw_preds, _, codes = predict_no_tta(model, test_loader, device, is_regression=cfg.is_regression)

    pseudo_labels: dict[str, float] = {}
    for code, pred in zip(codes, raw_preds):
        if cfg.is_regression:
            pseudo_labels[code] = float(pred)
        else:
            # raw_preds is shape [N, num_classes]; take argmax as hard label
            pseudo_labels[code] = float(int(np.argmax(pred)))

    logger.info(f"Generated pseudo labels for {len(pseudo_labels)} test images")
    return pseudo_labels


def finetune_with_pseudo(
    model: nn.Module,
    train_dataset: DRDataset,
    pseudo_labels: dict[str, float],
    cfg: ExpConfig,
    device: torch.device,
) -> nn.Module:
    pseudo_codes = list(pseudo_labels.keys())
    transform = train_dataset.transform

    combined_ds = PseudoLabelDataset(
        real_dataset=train_dataset,
        pseudo_codes=pseudo_codes,
        pseudo_labels=pseudo_labels,
        pseudo_img_dir=TEST_IMG_DIR,
        transform=transform,
        pseudo_weight=cfg.pseudo_weight,
        is_regression=cfg.is_regression,
    )

    loader = DataLoader(
        combined_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    criterion = build_loss(cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.pseudo_lr, weight_decay=cfg.weight_decay)

    model.train()
    for epoch in range(1, cfg.pseudo_epochs + 1):
        total_loss = 0.0
        n_batches = 0

        for images, targets, codes, weights in loader:
            images = images.to(device)
            targets = targets.to(device)
            weights = weights.to(device)

            outputs = model(images)
            if cfg.is_regression:
                loss_per_sample = torch.nn.functional.smooth_l1_loss(
                    outputs.squeeze(1), targets, reduction="none",
                )
            else:
                loss_per_sample = torch.nn.functional.cross_entropy(
                    outputs, targets.long(), reduction="none",
                )

            loss = (loss_per_sample * weights).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        logger.info(f"Pseudo-label epoch {epoch}/{cfg.pseudo_epochs} | Loss: {avg_loss:.4f}")

    torch.save(
        model.state_dict(),
        cfg.ckpt_dir / f"{cfg.exp_name}_pseudo.pth",
    )
    return model
