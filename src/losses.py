from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import CLASS_COUNTS, NUM_CLASSES, TOTAL_TRAIN, ExpConfig


def compute_class_weights(device: torch.device) -> torch.Tensor:
    inv_freq = torch.tensor([TOTAL_TRAIN / c for c in CLASS_COUNTS], dtype=torch.float32)
    weights = inv_freq / inv_freq.sum() * NUM_CLASSES
    return weights.to(device)


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha: torch.Tensor | None = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(1)

        if targets.dim() == 1:
            one_hot = F.one_hot(targets, num_classes).float()
        else:
            one_hot = targets

        if self.label_smoothing > 0:
            one_hot = (
                one_hot * (1 - self.label_smoothing)
                + self.label_smoothing / num_classes
            )

        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        focal_weight = (1 - probs).pow(self.gamma)
        loss = -focal_weight * one_hot * log_probs

        if self.alpha is not None:
            loss = loss * self.alpha.unsqueeze(0)

        return loss.sum(dim=1).mean()


def build_loss(cfg: ExpConfig, device: torch.device) -> nn.Module:
    if cfg.loss_type == "ce":
        return nn.CrossEntropyLoss()

    if cfg.loss_type == "focal":
        alpha = compute_class_weights(device) if cfg.use_class_weights else None
        return FocalLoss(
            gamma=cfg.focal_gamma,
            alpha=alpha,
            label_smoothing=cfg.label_smoothing,
        )

    if cfg.loss_type == "smoothl1":
        return nn.SmoothL1Loss()

    raise ValueError(f"Unknown loss type: {cfg.loss_type}")
