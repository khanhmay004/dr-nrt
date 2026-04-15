from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

from src.config import ExpConfig


class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (x.size(-2), x.size(-1)),
        ).pow(1.0 / self.p)


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning.

    Linear(in_dim -> hidden) -> BN -> ReLU -> Linear(hidden -> out_dim) -> L2-normalize.
    """

    def __init__(self, in_dim: int = 2048, hidden_dim: int = 512, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=1)


def build_model(cfg: ExpConfig) -> nn.Module:
    if cfg.backbone == "resnet50":
        model = _build_resnet50(cfg)
    elif cfg.backbone == "efficientnet_b4":
        model = _build_efficientnet_b4(cfg)
    elif cfg.backbone == "convnext_small":
        model = _build_convnext_small(cfg)
    else:
        raise ValueError(f"Unknown backbone: {cfg.backbone}")
    return model


def _build_resnet50(cfg: ExpConfig) -> nn.Module:
    weights = tvm.ResNet50_Weights.IMAGENET1K_V2
    model = tvm.resnet50(weights=weights)

    if cfg.use_gem:
        model.avgpool = GeM(p=cfg.gem_p)

    in_features = model.fc.in_features
    if cfg.head_dropout > 0.0:
        model.fc = nn.Sequential(
            nn.Dropout(p=cfg.head_dropout),
            nn.Linear(in_features, cfg.num_outputs),
        )
    else:
        model.fc = nn.Linear(in_features, cfg.num_outputs)
    return model


def _build_efficientnet_b4(cfg: ExpConfig) -> nn.Module:
    weights = tvm.EfficientNet_B4_Weights.IMAGENET1K_V1
    model = tvm.efficientnet_b4(weights=weights)

    if cfg.use_gem:
        model.avgpool = GeM(p=cfg.gem_p)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Linear(in_features, cfg.num_outputs)
    return model


def _build_convnext_small(cfg: ExpConfig) -> nn.Module:
    weights = tvm.ConvNeXt_Small_Weights.IMAGENET1K_V1
    model = tvm.convnext_small(weights=weights)

    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, cfg.num_outputs)
    return model


def freeze_backbone(model: nn.Module, backbone: str) -> None:
    if backbone in ("resnet50",):
        for name, param in model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False
    elif backbone == "efficientnet_b4":
        for name, param in model.named_parameters():
            if not name.startswith("classifier"):
                param.requires_grad = False
    elif backbone == "convnext_small":
        for name, param in model.named_parameters():
            if not name.startswith("classifier"):
                param.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def build_contrastive_model(cfg: ExpConfig) -> tuple[nn.Module, ProjectionHead]:
    """Build backbone (with fc=Identity) + projection head for contrastive pre-training."""
    weights = tvm.ResNet50_Weights.IMAGENET1K_V2
    model = tvm.resnet50(weights=weights)

    if cfg.use_gem:
        model.avgpool = GeM(p=cfg.gem_p)

    in_features = model.fc.in_features  # 2048
    model.fc = nn.Identity()

    projector = ProjectionHead(
        in_dim=in_features,
        hidden_dim=512,
        out_dim=cfg.contrastive_proj_dim,
    )
    return model, projector
