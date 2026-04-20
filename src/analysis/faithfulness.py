"""Quantitative CAM validation without pixel-level lesion masks.

Metrics:
* **FOV sanity** — fraction of top-k% CAM energy that lands inside the
  retinal field-of-view (should be ≥0.90 for a well-set-up explainer).
* **Anatomy breakdown** — share of CAM energy on optic-disc vs fovea vs
  rest-of-retina vs background.
* **Lesion proxies** — pointing-game and Dice overlap against classical-CV
  candidate masks (MA / hemorrhage / exudate) from ``fundus_cv``.
* **Insertion / deletion curves** (Petsiuk 2018) — faithfulness AUC.
* **TTA IoU** — pairwise IoU of CAMs computed under H-flip and rotations.
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
import torch

from src.analysis import fundus_cv


# ---------------------------------------------------------------------------
# Spatial sanity checks
# ---------------------------------------------------------------------------


def _top_mask(cam: np.ndarray, top_pct: float) -> np.ndarray:
    flat = cam.flatten()
    if flat.size == 0:
        return np.zeros_like(cam, dtype=bool)
    thr = np.quantile(flat, 1.0 - top_pct)
    return cam >= thr


def fov_energy_fraction(cam: np.ndarray, fov_mask: np.ndarray, top_pct: float = 0.2) -> float:
    """Share of the top-``top_pct`` CAM pixels that lie inside the FOV."""
    top = _top_mask(cam, top_pct)
    if not top.any():
        return 0.0
    inside = np.logical_and(top, fov_mask.astype(bool))
    return float(inside.sum() / top.sum())


def anatomy_breakdown(
    cam: np.ndarray,
    fov_mask: np.ndarray,
    disc_mask: np.ndarray | None = None,
    fovea_mask: np.ndarray | None = None,
    top_pct: float = 0.2,
) -> dict[str, float]:
    """Split top-``top_pct`` CAM mass across {optic_disc, fovea, rest, background}."""
    top = _top_mask(cam, top_pct)
    total = float(top.sum()) + 1e-12
    out = {"optic_disc": 0.0, "fovea": 0.0, "rest_of_retina": 0.0, "background": 0.0}
    bg = top & ~fov_mask.astype(bool)
    out["background"] = float(bg.sum()) / total

    inside = top & fov_mask.astype(bool)
    rest = inside.copy()
    if disc_mask is not None:
        d = inside & disc_mask.astype(bool)
        out["optic_disc"] = float(d.sum()) / total
        rest = rest & ~disc_mask.astype(bool)
    if fovea_mask is not None:
        f = inside & fovea_mask.astype(bool)
        out["fovea"] = float(f.sum()) / total
        rest = rest & ~fovea_mask.astype(bool)
    out["rest_of_retina"] = float(rest.sum()) / total
    return out


# ---------------------------------------------------------------------------
# Lesion-proxy overlap
# ---------------------------------------------------------------------------


def pointing_game(cam: np.ndarray, lesion_mask: np.ndarray) -> bool:
    """True if the CAM argmax falls inside any lesion candidate."""
    if not lesion_mask.any():
        return False
    yx = np.unravel_index(int(cam.argmax()), cam.shape)
    return bool(lesion_mask.astype(bool)[yx])


def cam_lesion_dice(cam: np.ndarray, lesion_mask: np.ndarray, top_pct: float = 0.2) -> float:
    top = _top_mask(cam, top_pct)
    m = lesion_mask.astype(bool)
    if not top.any() and not m.any():
        return 1.0
    inter = float((top & m).sum())
    return float(2.0 * inter / (top.sum() + m.sum() + 1e-12))


def compute_lesion_proxies(ben_graham_rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Convenience — run every classical-CV lesion detector on one image."""
    return {
        "ma": fundus_cv.ma_candidates(ben_graham_rgb),
        "hemorrhage": fundus_cv.hemorrhage_candidates(ben_graham_rgb),
        "exudate": fundus_cv.hard_exudate_candidates(ben_graham_rgb),
    }


# ---------------------------------------------------------------------------
# Insertion / deletion AUC (Petsiuk 2018)
# ---------------------------------------------------------------------------


def _ordered_pixel_indices(cam: np.ndarray, descending: bool = True) -> np.ndarray:
    """Indices into a flattened image, ordered by CAM saliency."""
    flat = cam.flatten()
    order = np.argsort(flat)
    if descending:
        order = order[::-1]
    return order


def insertion_curve(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    cam: np.ndarray,
    target_class: int,
    n_steps: int = 20,
    baseline: str = "zero",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Insertion: start from baseline, insert pixels in descending saliency.
    Returns (fractions, scores, AUC)."""
    device = image_tensor.device
    _, C, H, W = image_tensor.shape
    base = torch.zeros_like(image_tensor) if baseline == "zero" \
           else torch.full_like(image_tensor, -1.5)

    order = _ordered_pixel_indices(cam, descending=True)
    fractions = np.linspace(0, 1, n_steps + 1)
    scores = []
    model.eval()
    for f in fractions:
        k = int(f * len(order))
        mask = torch.zeros(H * W, device=device)
        if k > 0:
            mask[torch.as_tensor(order[:k], device=device, dtype=torch.long)] = 1.0
        mask = mask.view(1, 1, H, W)
        blended = image_tensor * mask + base * (1.0 - mask)
        with torch.no_grad():
            out = model(blended)
            if out.shape[1] == 1:  # regression head
                score = float(out.item())
            else:
                score = float(torch.softmax(out, dim=1)[0, int(target_class)].item())
        scores.append(score)
    scores_arr = np.array(scores)
    auc = float(np.trapz(scores_arr, fractions))
    return fractions, scores_arr, auc


def deletion_curve(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    cam: np.ndarray,
    target_class: int,
    n_steps: int = 20,
    baseline: str = "zero",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Deletion: start from full image, remove pixels in descending saliency.
    Returns (fractions, scores, AUC). Lower AUC = more faithful."""
    device = image_tensor.device
    _, _, H, W = image_tensor.shape
    base = torch.zeros_like(image_tensor) if baseline == "zero" \
           else torch.full_like(image_tensor, -1.5)

    order = _ordered_pixel_indices(cam, descending=True)
    fractions = np.linspace(0, 1, n_steps + 1)
    scores = []
    model.eval()
    for f in fractions:
        k = int(f * len(order))
        mask = torch.ones(H * W, device=device)
        if k > 0:
            mask[torch.as_tensor(order[:k], device=device, dtype=torch.long)] = 0.0
        mask = mask.view(1, 1, H, W)
        blended = image_tensor * mask + base * (1.0 - mask)
        with torch.no_grad():
            out = model(blended)
            if out.shape[1] == 1:
                score = float(out.item())
            else:
                score = float(torch.softmax(out, dim=1)[0, int(target_class)].item())
        scores.append(score)
    scores_arr = np.array(scores)
    auc = float(np.trapz(scores_arr, fractions))
    return fractions, scores_arr, auc


# ---------------------------------------------------------------------------
# TTA consistency
# ---------------------------------------------------------------------------


def cam_pairwise_iou(cams: list[np.ndarray], top_pct: float = 0.2) -> float:
    """Mean pairwise IoU over a list of CAMs (same shape), thresholded at top-k%."""
    masks = [_top_mask(c, top_pct) for c in cams]
    n = len(masks)
    if n < 2:
        return 1.0
    ious = []
    for i in range(n):
        for j in range(i + 1, n):
            u = masks[i] | masks[j]
            if not u.any():
                continue
            inter = float((masks[i] & masks[j]).sum())
            ious.append(inter / u.sum())
    return float(np.mean(ious)) if ious else 0.0


# ---------------------------------------------------------------------------
# Quick all-in-one evaluator for one sample
# ---------------------------------------------------------------------------


def evaluate_sample(
    cam: np.ndarray,
    ben_graham_rgb: np.ndarray,
    fov_mask: np.ndarray | None = None,
    disc_mask: np.ndarray | None = None,
    fovea_mask: np.ndarray | None = None,
    lesion_masks: dict[str, np.ndarray] | None = None,
    top_pct: float = 0.2,
) -> dict[str, float]:
    """Run every no-model-needed sanity check on one CAM."""
    if fov_mask is None:
        fov_mask = fundus_cv.retinal_fov_mask(ben_graham_rgb)
    # Resize CAM to image resolution if needed
    if cam.shape != fov_mask.shape:
        cam = cv2.resize(cam, (fov_mask.shape[1], fov_mask.shape[0]),
                         interpolation=cv2.INTER_LINEAR)

    out: dict[str, float] = {}
    out["fov_energy_top20"] = fov_energy_fraction(cam, fov_mask, top_pct=top_pct)
    ana = anatomy_breakdown(cam, fov_mask, disc_mask, fovea_mask, top_pct=top_pct)
    out.update({f"share_{k}": v for k, v in ana.items()})
    if lesion_masks is None:
        lesion_masks = compute_lesion_proxies(ben_graham_rgb)
    for name, mask in lesion_masks.items():
        if cam.shape != mask.shape:
            mask_r = cv2.resize(mask.astype(np.uint8),
                                (cam.shape[1], cam.shape[0]),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            mask_r = mask.astype(bool)
        out[f"pointing_{name}"] = float(pointing_game(cam, mask_r))
        out[f"dice_{name}"] = cam_lesion_dice(cam, mask_r, top_pct=top_pct)
    return out
