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


class CORNLoss(nn.Module):
    """CORN (Conditional Ordinal Regression) loss wrapper."""

    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        from coral_pytorch.losses import corn_loss
        return corn_loss(logits, targets, num_classes=self.num_classes)


class CumulativeLinkLoss(nn.Module):
    """Cumulative link model: K-1 independent binary cross-entropy losses."""

    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        K = self.num_classes
        levels = torch.arange(1, K, device=targets.device).unsqueeze(0)
        binary_targets = (targets.unsqueeze(1) >= levels).float()
        return F.binary_cross_entropy_with_logits(logits, binary_targets)


class EMDLoss(nn.Module):
    """Earth Mover's Distance loss for ordinal classification."""

    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, self.num_classes).float()
        cdf_pred = torch.cumsum(probs, dim=1)
        cdf_true = torch.cumsum(one_hot, dim=1)
        emd = (cdf_pred - cdf_true).pow(2).sum(dim=1).mean()
        return emd


class SORDLoss(nn.Module):
    """Soft ORDinal regression loss (Diaz & Marathe, CVPR 2019).

    For target y_i in {0, ..., K-1}, build a soft label
        p_k proportional to exp(-phi(|y_i - k|))
    and minimize KL-divergence between predicted softmax and p.
    """

    def __init__(
        self,
        num_classes: int = 5,
        phi: str = "abs",
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.K = num_classes
        self.phi = phi
        if class_weights is not None:
            self.register_buffer("w", class_weights)
        else:
            self.w: torch.Tensor | None = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        r = torch.arange(self.K, device=logits.device, dtype=torch.float32)
        dist = (targets.float().unsqueeze(1) - r.unsqueeze(0)).abs()
        if self.phi == "square":
            dist = dist ** 2
        soft = F.softmax(-dist, dim=1)
        logp = F.log_softmax(logits, dim=1)
        loss = -(soft * logp).sum(dim=1)
        if self.w is not None:
            loss = loss * self.w[targets]
        return loss.mean()


class LogitAdjustedCE(nn.Module):
    """Logit Adjustment (Menon et al., ICLR 2021).

    Subtract tau * log(pi_c) from logits during training.
    Fisher-consistent for balanced error at tau=1.
    """

    def __init__(
        self,
        class_counts: list[int],
        tau: float = 1.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        pi = torch.tensor(class_counts, dtype=torch.float32)
        pi = pi / pi.sum()
        self.register_buffer("log_pi", torch.log(pi))
        self.tau = tau
        self.ls = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        adj = logits + self.tau * self.log_pi.unsqueeze(0)
        return F.cross_entropy(adj, targets, label_smoothing=self.ls)


class CLOCLoss(nn.Module):
    """Multi-margin N-pair contrastive loss for ordinal classification.

    Reference: Pitawela et al., CVPR 2025. Adapted for 5-class DR grading.
    """

    def __init__(
        self,
        num_classes: int = 5,
        temperature: float = 0.07,
        margin_init: float = 0.5,
        min_margin_23: float = 0.8,
        margin_reg: float = 1.0,
    ) -> None:
        super().__init__()
        self.K = num_classes
        self.T = temperature
        self.deltas = nn.Parameter(torch.full((num_classes - 1,), margin_init))
        self.min_margin_23 = min_margin_23
        self.margin_reg = margin_reg

    def cumulative_margin(self, dist: torch.Tensor) -> torch.Tensor:
        """For |delta_y| = d, margin = sum of softplus(deltas)[0..d-1].

        ``softplus`` enforces positivity of each per-adjacent-pair margin so that
        ``cumulative_margin`` is monotonically non-decreasing in ``d``.  Without
        it, unconstrained learned ``deltas`` can go negative, which would *add*
        similarity to distant-class pairs — the opposite of the intended
        separation.  Only ``deltas[2]`` is explicitly reg-constrained (via
        ``min_margin_23``); the other three would otherwise be free to drift.
        """
        pos_deltas = F.softplus(self.deltas)
        c = torch.zeros(self.K, device=dist.device, dtype=pos_deltas.dtype)
        c[1:] = pos_deltas.cumsum(0)
        return c[dist]

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        sim = features @ features.T / self.T
        d = (labels.unsqueeze(0) - labels.unsqueeze(1)).abs()
        sim_adj = sim - self.cumulative_margin(d)
        eye = torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
        sim_adj = sim_adj.masked_fill(eye, -1e4)
        pos_mask = (d == 0) & ~eye
        logp = sim_adj - torch.logsumexp(sim_adj, dim=1, keepdim=True)
        n_pos = pos_mask.sum(dim=1).clamp(min=1).float()
        per_anchor = -(logp * pos_mask.float()).sum(dim=1) / n_pos
        loss = per_anchor.mean()
        # Regularizer operates on the effective (positive) margin actually used
        # in ``cumulative_margin``, so the clinical constraint on G2<->G3 matches
        # what the loss sees.
        effective_23 = F.softplus(self.deltas[2])
        reg = F.relu(self.min_margin_23 - effective_23) ** 2
        return loss + self.margin_reg * reg


class RnCLoss(nn.Module):
    """Rank-N-Contrast loss for continuous/ordinal labels (Zha et al., NeurIPS 2023).

    For anchor i and every other sample j, the per-pair loss is

        L_{i,j} = -log( exp(s_{i,j}) /
                        sum_{k: d(y_i, y_k) >= d(y_i, y_j), k != i} exp(s_{i,k}) )

    where s is feature similarity (negative squared-L2 or cosine, both scaled by 1/T)
    and d is label distance (L1 or squared).  The final loss is the mean of L_{i,j}
    over all (i, j), j != i.  Unlike the previous implementation, the numerator is a
    single exp(s_{i,j}) — not a sum over closer-than-j pairs — which matches the
    published formulation.
    """

    def __init__(
        self,
        temperature: float = 2.0,
        label_diff: str = "l1",
        feature_sim: str = "l2",
    ) -> None:
        super().__init__()
        self.T = temperature
        self.label_diff = label_diff
        self.feature_sim = feature_sim

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        N = features.size(0)
        device = features.device

        if self.feature_sim == "l2":
            feat_dist = torch.cdist(features, features, p=2) ** 2
            sim = -feat_dist / self.T
        else:
            sim = features @ features.T / self.T

        if self.label_diff == "l1":
            ldist = (labels.unsqueeze(0) - labels.unsqueeze(1)).abs().float()
        else:
            ldist = (labels.unsqueeze(0) - labels.unsqueeze(1)).float() ** 2

        eye = torch.eye(N, dtype=torch.bool, device=device)

        total_loss = torch.zeros((), device=device)
        total_pairs = 0
        # One outer loop over anchors keeps memory at O(N^2); inner pair loop is
        # vectorized via a [N_j, N_k] inclusion mask.
        for i in range(N):
            li = ldist[i]                                       # [N]
            # include[j, k] = True iff k is in the denominator for pair (i, j):
            #   k != i  and  d(y_i, y_k) >= d(y_i, y_j)
            include = li.unsqueeze(0) >= li.unsqueeze(1)        # [N_j, N_k]
            include[:, i] = False
            sim_i = sim[i]                                       # [N]
            sim_masked = sim_i.unsqueeze(0).expand(N, N).masked_fill(~include, -1e4)
            denom_log = torch.logsumexp(sim_masked, dim=1)       # [N_j]
            losses_j = denom_log - sim_i                          # [N_j]
            # Drop j == i (the anchor itself has no pair with itself).
            keep = ~eye[i]
            total_loss = total_loss + losses_j[keep].sum()
            total_pairs += int(keep.sum().item())

        if total_pairs == 0:
            return torch.zeros((), device=device)
        return total_loss / total_pairs


def build_loss(cfg: ExpConfig, device: torch.device) -> nn.Module:
    if cfg.loss_type == "none":
        return nn.Identity()

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

    if cfg.loss_type == "corn":
        assert cfg.num_outputs == 5, f"CORN assumes num_outputs=5, got {cfg.num_outputs}"
        return CORNLoss(num_classes=NUM_CLASSES)

    if cfg.loss_type == "cumlink":
        assert cfg.num_outputs == 5, f"CumLink assumes num_outputs=5, got {cfg.num_outputs}"
        return CumulativeLinkLoss(num_classes=NUM_CLASSES)

    if cfg.loss_type == "emd":
        return EMDLoss(num_classes=NUM_CLASSES)

    if cfg.loss_type == "sord":
        w = compute_class_weights(device) if cfg.use_class_weights else None
        return SORDLoss(num_classes=NUM_CLASSES, phi=cfg.sord_phi, class_weights=w)

    if cfg.loss_type == "la_ce":
        return LogitAdjustedCE(
            class_counts=CLASS_COUNTS,
            tau=cfg.la_ce_tau,
            label_smoothing=cfg.label_smoothing,
        )

    raise ValueError(f"Unknown loss type: {cfg.loss_type}")


class OrdSupConLoss(nn.Module):
    """Ordinal-Aware Supervised Contrastive Loss (canonical Khosla form).

    Positives are same-class and adjacent-class pairs, weighted by
    W(i,j) = 1 - |g_i - g_j| / (K - 1). Distant classes (|g_i - g_j| >= 2)
    are treated as pure negatives (repulsion only via the denominator).
    This matches Khosla SupCon 2020 semantics: negatives repel uniformly.
    """

    def __init__(
        self,
        num_classes: int = 5,
        temperature: float = 0.07,
        pos_distance: int = 1,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.num_classes = num_classes
        self.pos_distance = pos_distance  # max label distance considered a positive

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: L2-normalised embeddings [2N, D] (two views concatenated).
            labels:   integer labels [2N].
        """
        device = features.device
        batch_size = features.shape[0]

        # Pairwise ordinal weight: W(i,j) = 1 - |g_i - g_j| / (K-1)
        labels_f = labels.float()
        dist = (labels_f.unsqueeze(0) - labels_f.unsqueeze(1)).abs()
        W = 1.0 - dist / (self.num_classes - 1)  # [2N, 2N]

        # Cosine similarity (features already L2-normalised)
        sim = torch.mm(features, features.t()) / self.temperature  # [2N, 2N]

        # Mask out self-similarities
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        sim.masked_fill_(self_mask, -1e4)  # FP16-safe (max half ≈ 65504)

        # Log-softmax over the denominator — uniform over all non-self pairs,
        # which is the canonical SupCon denominator. Negatives repel uniformly.
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

        # Positive mask: same or adjacent class only. Distant-class pairs are
        # pure negatives (zero weight in numerator; they still repel via the
        # denominator). Previous bug: W alone gave soft positive weight to all
        # pairs incl. distant negatives — breaking negative repulsion.
        pos_mask = (dist <= self.pos_distance).float()
        pos_mask.masked_fill_(self_mask, 0.0)
        W_pos = W * pos_mask  # non-zero only for same/adjacent pairs

        # Per-anchor normalisation: sum of positive weights
        Z = W_pos.sum(dim=1, keepdim=True).clamp(min=1e-8)

        # -sum_j [W_pos_ij * log_prob_ij] / Z_i. Clamp guards 0 * -inf = NaN.
        loss = -(W_pos * log_prob.clamp(min=-100)).sum(dim=1) / Z.squeeze(1)
        return loss.mean()
