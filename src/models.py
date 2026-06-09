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


class OrdinalPrototypeHead(nn.Module):
    """Cosine classifier with K prototypes on a geodesic (1-D manifold on the unit sphere).

    Prototypes are placed at angles ``theta_k = (k - (K-1)/2) * angular_spacing`` on the
    great circle spanned by two learnable orthonormal axes ``v1, v2``:

        mu_k = cos(theta_k) * v1 + sin(theta_k) * v2

    This preserves the "1-D ordinal axis" intent while guaranteeing K distinct unit
    prototypes (the naive single-axis formulation ``mu_k = k * v`` collapses to at most
    two unit vectors after L2-normalization, and produces a degenerate zero prototype
    at k = (K-1)/2).
    """

    def __init__(
        self,
        feat_dim: int = 2048,
        num_classes: int = 5,
        scale: float = 20.0,
        angular_spacing: float = 0.4,
        learnable_axis: bool = True,
    ) -> None:
        super().__init__()
        self.K = num_classes
        self.scale = scale
        self.angular_spacing = angular_spacing

        v1 = torch.randn(feat_dim)
        v1 = v1 / v1.norm().clamp(min=1e-8)
        v2 = torch.randn(feat_dim)
        v2 = v2 - (v2 @ v1) * v1
        v2 = v2 / v2.norm().clamp(min=1e-8)
        self.v1 = nn.Parameter(v1, requires_grad=learnable_axis)
        self.v2 = nn.Parameter(v2, requires_grad=learnable_axis)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        feat_n = F.normalize(feat, dim=1)
        v1_n = F.normalize(self.v1, dim=0)
        v2_perp = self.v2 - (self.v2 @ v1_n) * v1_n
        v2_n = F.normalize(v2_perp, dim=0)

        ks = torch.arange(self.K, device=feat.device, dtype=feat.dtype)
        angles = (ks - (self.K - 1) / 2) * self.angular_spacing
        mu = torch.cos(angles).unsqueeze(1) * v1_n.unsqueeze(0) \
             + torch.sin(angles).unsqueeze(1) * v2_n.unsqueeze(0)
        mu_n = F.normalize(mu, dim=1)
        logits = self.scale * (feat_n @ mu_n.T)
        return logits


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
    actual_outputs = cfg.num_outputs
    if cfg.loss_type in ("corn", "cumlink"):
        actual_outputs = cfg.num_outputs - 1

    if cfg.head_type == "ordinal_prototype":
        model.fc = OrdinalPrototypeHead(
            feat_dim=in_features,
            num_classes=cfg.num_outputs,
            scale=cfg.proto_scale,
            learnable_axis=cfg.proto_learnable_axis,
        )
    elif cfg.head_dropout > 0.0:
        model.fc = nn.Sequential(
            nn.Dropout(p=cfg.head_dropout),
            nn.Linear(in_features, actual_outputs),
        )
    else:
        model.fc = nn.Linear(in_features, actual_outputs)
    return model


def _build_efficientnet_b4(cfg: ExpConfig) -> nn.Module:
    if cfg.loss_type in ("corn", "cumlink"):
        raise NotImplementedError(f"CORN/CumLink not yet supported for {cfg.backbone}")

    weights = tvm.EfficientNet_B4_Weights.IMAGENET1K_V1
    model = tvm.efficientnet_b4(weights=weights)

    if cfg.use_gem:
        model.avgpool = GeM(p=cfg.gem_p)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Linear(in_features, cfg.num_outputs)
    return model


def _build_convnext_small(cfg: ExpConfig) -> nn.Module:
    if cfg.loss_type in ("corn", "cumlink"):
        raise NotImplementedError(f"CORN/CumLink not yet supported for {cfg.backbone}")

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
