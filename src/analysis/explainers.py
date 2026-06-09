"""Unified wrappers for Grad-CAM / HiResCAM / Occlusion / Integrated Gradients.

All explainers take a *preprocessed* image tensor of shape ``[1, 3, H, W]``
(already Ben-Graham-processed, ImageNet-normalised) and return a 2-D
saliency map of shape ``[H, W]`` in ``[0, 1]``.

Model assumption: `model(image_tensor)` returns ``[1, K]`` logits (K=5 for
DR classification) and ``model.layer4[-1]`` is the CAM target layer for
ResNet-50. A custom ``target_layer`` can be supplied for other backbones.
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def image_to_tensor(
    ben_graham_rgb: np.ndarray,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """Match `src/dataset.py` normalisation. Input is a Ben-Graham-processed
    uint8 RGB image of shape [H, W, 3]."""
    img = ben_graham_rgb.astype(np.float32) / 255.0
    for c in range(3):
        img[..., c] = (img[..., c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
    t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
    return t.to(device)


def tensor_to_display_rgb(tensor: torch.Tensor) -> np.ndarray:
    """Invert the normalisation for display. Accepts [1,3,H,W] or [3,H,W]."""
    t = tensor.detach().cpu().numpy()
    if t.ndim == 4:
        t = t[0]
    t = t.transpose(1, 2, 0)
    for c in range(3):
        t[..., c] = t[..., c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]
    return np.clip(t * 255.0, 0, 255).astype(np.uint8)


def _normalize_cam(cam: np.ndarray) -> np.ndarray:
    cam = cam.astype(np.float32)
    m, M = float(cam.min()), float(cam.max())
    if M - m < 1e-8:
        return np.zeros_like(cam)
    return (cam - m) / (M - m)


# ---------------------------------------------------------------------------
# Grad-CAM family via pytorch-grad-cam
# ---------------------------------------------------------------------------


def _make_cam_target(target_class: int):
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    return [ClassifierOutputTarget(int(target_class))]


def gradcam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
    target_layer: nn.Module,
    method: str = "gradcam",
) -> np.ndarray:
    """One of: 'gradcam', 'gradcam++', 'hirescam', 'eigencam', 'scorecam'."""
    from pytorch_grad_cam import (
        GradCAM, GradCAMPlusPlus, HiResCAM, EigenCAM, ScoreCAM,
    )

    cls = {
        "gradcam": GradCAM,
        "gradcam++": GradCAMPlusPlus,
        "hirescam": HiResCAM,
        "eigencam": EigenCAM,
        "scorecam": ScoreCAM,
    }[method.lower()]
    cam_obj = cls(model=model, target_layers=[target_layer])
    grayscale = cam_obj(
        input_tensor=image_tensor,
        targets=_make_cam_target(target_class),
    )[0]
    return _normalize_cam(grayscale)


# ---------------------------------------------------------------------------
# Occlusion sensitivity (gradient-free baseline)
# ---------------------------------------------------------------------------


def occlusion(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
    patch_size: int = 48,
    stride: int = 24,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Slide a patch, zero the region, record drop in target-class logit."""
    assert image_tensor.ndim == 4 and image_tensor.shape[0] == 1
    device = image_tensor.device
    _, _, H, W = image_tensor.shape

    model.eval()
    with torch.no_grad():
        base_logit = float(model(image_tensor)[0, int(target_class)])

    ys = list(range(0, H - patch_size + 1, stride))
    xs = list(range(0, W - patch_size + 1, stride))
    heat = np.zeros((H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.float32)

    for y in ys:
        for x in xs:
            patched = image_tensor.clone()
            patched[0, :, y:y + patch_size, x:x + patch_size] = fill_value
            with torch.no_grad():
                occ_logit = float(model(patched)[0, int(target_class)])
            drop = base_logit - occ_logit
            heat[y:y + patch_size, x:x + patch_size] += drop
            counts[y:y + patch_size, x:x + patch_size] += 1.0

    counts = np.maximum(counts, 1.0)
    heat = heat / counts
    return _normalize_cam(heat)


# ---------------------------------------------------------------------------
# Integrated Gradients via captum
# ---------------------------------------------------------------------------


def integrated_gradients(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
    n_steps: int = 32,
    baseline: str = "zero",
) -> np.ndarray:
    """Returns a 2-D map by taking the per-pixel sum of absolute channel
    attributions. Baseline: 'zero' (mid-gray in normalised space) or 'black'
    (true 0 before normalisation)."""
    from captum.attr import IntegratedGradients

    model.eval()
    if baseline == "zero":
        base = torch.zeros_like(image_tensor)
    else:
        base = torch.full_like(image_tensor, -1.5)  # roughly -mean/std

    ig = IntegratedGradients(model)
    attrs = ig.attribute(
        image_tensor,
        baselines=base,
        target=int(target_class),
        n_steps=n_steps,
        internal_batch_size=8,
    )
    a = attrs[0].detach().cpu().numpy()
    heat = np.abs(a).sum(axis=0)
    return _normalize_cam(heat)


# ---------------------------------------------------------------------------
# SHAP GradientExplainer — optional; expensive
# ---------------------------------------------------------------------------


def shap_gradient(
    model: nn.Module,
    image_tensor: torch.Tensor,
    background: torch.Tensor,
    target_class: int,
) -> np.ndarray:
    """SHAP GradientExplainer on a [1,3,H,W] sample with a small background.
    Returns a 2-D absolute-sum map. Expects `background` shape [B_bg,3,H,W]."""
    import shap

    model.eval()
    e = shap.GradientExplainer(model, background)
    sv = e.shap_values(image_tensor)
    if isinstance(sv, list):
        arr = sv[int(target_class)]
    else:
        arr = sv[..., int(target_class)]
    if arr.ndim == 4:
        arr = arr[0]
    heat = np.abs(arr).sum(axis=0)
    return _normalize_cam(heat)


# ---------------------------------------------------------------------------
# Ensemble-weighted CAM
# ---------------------------------------------------------------------------


def weighted_cam(cams: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Weighted-average of aligned CAMs (e.g., per ensemble member)."""
    assert len(cams) == len(weights) and len(cams) > 0
    w = np.asarray(weights, dtype=np.float32)
    w = w / (w.sum() + 1e-12)
    stack = np.stack([c.astype(np.float32) for c in cams], axis=0)
    combined = (stack * w.reshape(-1, 1, 1)).sum(axis=0)
    return _normalize_cam(combined)


# ---------------------------------------------------------------------------
# TTA-aligned CAM (for consistency checks)
# ---------------------------------------------------------------------------


def tta_aligned_cams(
    cam_fn: Callable[[torch.Tensor, int], np.ndarray],
    image_tensor: torch.Tensor,
    target_class: int,
) -> list[np.ndarray]:
    """Compute CAMs under {identity, hflip, rot90, rot180, rot270} and
    un-rotate back to the original frame. Returns 5 aligned 2-D maps."""
    H, W = image_tensor.shape[-2:]
    variants = {
        "identity": (lambda t: t, lambda m: m),
        "hflip": (lambda t: torch.flip(t, dims=[3]),
                  lambda m: np.ascontiguousarray(m[:, ::-1])),
        "rot90": (lambda t: torch.rot90(t, k=1, dims=[2, 3]),
                  lambda m: np.rot90(m, k=-1).copy()),
        "rot180": (lambda t: torch.rot90(t, k=2, dims=[2, 3]),
                   lambda m: np.rot90(m, k=-2).copy()),
        "rot270": (lambda t: torch.rot90(t, k=3, dims=[2, 3]),
                   lambda m: np.rot90(m, k=-3).copy()),
    }
    out = []
    for _name, (fwd, inv) in variants.items():
        cam = cam_fn(fwd(image_tensor), target_class)
        out.append(_normalize_cam(inv(cam)))
    return out


# ---------------------------------------------------------------------------
# Model inference helper
# ---------------------------------------------------------------------------


def predict_probs(model: nn.Module, image_tensor: torch.Tensor) -> np.ndarray:
    """Softmax over classification logits. Returns shape [K]."""
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        if logits.ndim == 2 and logits.shape[1] == 1:
            # regression head — fake a one-hot centred on the predicted grade
            s = float(logits.item())
            probs = np.zeros(5, dtype=np.float32)
            idx = int(np.clip(round(s), 0, 4))
            probs[idx] = 1.0
            return probs
        return torch.softmax(logits, dim=1)[0].cpu().numpy()


# ---------------------------------------------------------------------------
# Lesion-Evidence-Weighted CAM (Method A) + Annotated Overlay (Method C)
# ---------------------------------------------------------------------------


def lesion_weighted_cam(
    base_cam: np.ndarray,
    image_bg_rgb: np.ndarray,
    alpha: float = 0.85,
    dilate_k: int = 9,
) -> np.ndarray:
    """Lesion-Evidence-Weighted CAM (LEW-CAM).

    Gates ``base_cam`` by a soft lesion-evidence map derived from classical-CV
    proxies in ``src.analysis.fundus_cv``. ``alpha`` controls gating strength:

    * ``alpha == 0.0`` — returns ``base_cam`` unchanged (no gating).
    * ``alpha == 1.0`` — hard mask: CAM is zero outside lesion evidence.
    * ``alpha == 0.85`` (default) — non-lesion CAM attenuated to 15%, lesion
      regions keep full CAM value. Good balance for visualization.

    For Grade-0 images the evidence map is mostly empty by design, so the
    returned heatmap is near-dark. That is the correct behavior.

    Args:
        base_cam: 2-D saliency map in [0, 1], any resolution.
        image_bg_rgb: Ben-Graham-processed uint8 RGB image at the resolution
                      you want the gate computed at (usually IMAGE_SIZE).
        alpha: Gating strength in [0, 1].
        dilate_k: Dilation kernel side for the lesion union (px).

    Returns:
        A 2-D map in [0, 1] at the same resolution as ``base_cam``.
    """
    from src.analysis.fundus_cv import lesion_evidence_map

    evidence = lesion_evidence_map(image_bg_rgb, dilate_k=dilate_k)
    if evidence.shape != base_cam.shape:
        evidence = cv2.resize(
            evidence,
            (base_cam.shape[1], base_cam.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    gate = (1.0 - alpha) + alpha * evidence
    return _normalize_cam(base_cam.astype(np.float32) * gate)


def render_lesion_annotated_panel(
    image_bg_rgb: np.ndarray,
    cam: np.ndarray,
    pred_grade: int,
    true_grade: int | None = None,
    top_pct: float = 0.20,
    heat_alpha: float = 0.30,
) -> tuple[np.ndarray, dict]:
    """Clinician-facing visualization.

    Renders the Ben-Graham image with:

    * a faint jet-colormap CAM overlay (top-k region highlighted),
    * color-coded circles around lesion candidates **inside** the CAM top-k
      region (red=MA, orange=HE, yellow=EX),
    * a caption summarising predicted grade and lesion counts.

    Args:
        image_bg_rgb: [H, W, 3] uint8, Ben-Graham output.
        cam: 2-D saliency map in [0, 1]; resized to [H, W] if needed.
        pred_grade: Predicted DR grade (0–4).
        true_grade: Ground-truth DR grade, if known; included in caption.
        top_pct: Quantile threshold for the CAM "attended region" (default 0.20).
        heat_alpha: Opacity of the CAM overlay (0 = none, 1 = fully opaque).

    Returns:
        canvas: [H, W, 3] uint8 RGB rendered image.
        meta:   dict with ``counts`` (per lesion type inside CAM top-k) and
                ``caption`` (natural-language summary).
    """
    from src.analysis.fundus_cv import (
        retinal_fov_mask,
        ma_candidates,
        hemorrhage_candidates,
        hard_exudate_candidates,
    )

    H, W = image_bg_rgb.shape[:2]
    cam_resized = cam
    if cam.shape != (H, W):
        cam_resized = cv2.resize(
            cam.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR
        )

    thr = float(np.quantile(cam_resized, 1.0 - top_pct))
    top_mask = cam_resized >= thr

    fov = retinal_fov_mask(image_bg_rgb)
    lesions = {
        "ma":         (ma_candidates(image_bg_rgb, fov_mask=fov),          (255,   0,   0)),
        "hemorrhage": (hemorrhage_candidates(image_bg_rgb, fov_mask=fov),  (255, 140,   0)),
        "exudate":    (hard_exudate_candidates(image_bg_rgb, fov_mask=fov),(255, 220,   0)),
    }

    # Faint jet-colormap overlay, blended with the fundus
    heat_u8 = np.clip(cam_resized * 255.0, 0, 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    canvas = cv2.addWeighted(
        image_bg_rgb, 1.0 - heat_alpha, heat_rgb, heat_alpha, 0.0
    ).astype(np.uint8)

    # Draw circles around lesion components that intersect the CAM top-k region
    counts: dict[str, int] = {}
    for name, (mask, color) in lesions.items():
        inside = (mask.astype(bool) & top_mask).astype(np.uint8)
        nlab, _, stats, centroids = cv2.connectedComponentsWithStats(inside)
        counts[name] = 0
        for i in range(1, nlab):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < 3:
                continue
            cx, cy = int(centroids[i, 0]), int(centroids[i, 1])
            r = max(6, int(np.sqrt(area / np.pi)) + 4)
            cv2.circle(canvas, (cx, cy), r, color, thickness=2)
            counts[name] += 1

    caption = (
        f"Pred: Grade {pred_grade}"
        + (f" | True: Grade {true_grade}" if true_grade is not None else "")
        + f" | In attended region: "
        f"{counts['ma']} MA, {counts['hemorrhage']} HE, {counts['exudate']} EX"
    )
    return canvas, {"counts": counts, "caption": caption}
