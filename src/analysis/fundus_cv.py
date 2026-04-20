"""Classical-CV utilities for fundus images.

Everything here operates on RGB uint8 arrays unless noted. Keep dependencies
to opencv / numpy / scikit-image so the module is cheap to import.
"""
from __future__ import annotations

import cv2
import numpy as np


def retinal_fov_mask(image: np.ndarray, threshold: int = 10) -> np.ndarray:
    """Binary mask of the retinal field of view.

    Mirrors the logic used in ``ben_graham_preprocess``: threshold on grayscale
    and keep the largest contour as the retinal disk.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray, dtype=np.uint8)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], -1, color=255, thickness=cv2.FILLED)
    return (mask > 0).astype(np.uint8)


def fov_fraction(image: np.ndarray) -> float:
    """Ratio of retinal-FOV pixels to total pixels. Useful quality metric."""
    mask = retinal_fov_mask(image)
    return float(mask.mean())


def detect_optic_disc(
    image: np.ndarray,
    fov_mask: np.ndarray | None = None,
) -> tuple[tuple[int, int] | None, int]:
    """Approximate optic-disc center + radius via brightness-peak search.

    Returns ((cx, cy), radius) or (None, 0) when no disc-like structure is
    detectable. Uses red channel (OD is red-dominant) masked by FOV.
    """
    if fov_mask is None:
        fov_mask = retinal_fov_mask(image)
    if image.ndim == 3:
        red = image[..., 0]
    else:
        red = image
    red = cv2.GaussianBlur(red, (31, 31), 0)
    red = red.astype(np.float32) * fov_mask
    if red.max() <= 0:
        return None, 0
    _, _, _, maxloc = cv2.minMaxLoc(red)
    h = image.shape[0]
    radius = int(h * 0.08)
    return (int(maxloc[0]), int(maxloc[1])), radius


def detect_fovea(
    image: np.ndarray,
    fov_mask: np.ndarray | None = None,
    optic_disc: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    """Approximate fovea center via darkest-region search.

    Fovea is the darkest retinal region and lies ~2.5 disc-diameters temporal
    to the optic disc. We use the darker pixels inside the FOV, excluding the
    optic-disc area, and take the centroid of the lowest-intensity percentile.
    """
    if fov_mask is None:
        fov_mask = retinal_fov_mask(image)
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    gray = cv2.GaussianBlur(gray, (51, 51), 0)

    mask = fov_mask.copy().astype(bool)
    if optic_disc is not None:
        h, w = gray.shape
        cy, cx = np.ogrid[:h, :w]
        od_x, od_y = optic_disc
        r = int(h * 0.12)
        od_mask = (cx - od_x) ** 2 + (cy - od_y) ** 2 <= r ** 2
        mask = mask & ~od_mask

    if not mask.any():
        return None
    vals = gray[mask]
    thresh = np.percentile(vals, 2)
    dark_mask = (gray <= thresh) & mask
    if not dark_mask.any():
        return None
    ys, xs = np.where(dark_mask)
    return int(xs.mean()), int(ys.mean())


def ma_candidates(
    image: np.ndarray,
    fov_mask: np.ndarray | None = None,
    min_sigma: float = 1.0,
    max_sigma: float = 4.0,
) -> np.ndarray:
    """Microaneurysm candidate mask (DoG blobs on inverted green channel).

    Returns a uint8 mask of candidate MA locations. Intended as a noisy proxy,
    not a clinical detector.
    """
    from skimage.feature import blob_dog

    if fov_mask is None:
        fov_mask = retinal_fov_mask(image)
    green = image[..., 1].astype(np.float32) / 255.0
    inv = 1.0 - green
    inv = inv * fov_mask
    blobs = blob_dog(inv, min_sigma=min_sigma, max_sigma=max_sigma, threshold=0.05)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for y, x, s in blobs:
        r = int(max(2, s * 1.414))
        cv2.circle(mask, (int(x), int(y)), r, 255, -1)
    mask = (mask > 0).astype(np.uint8) * fov_mask
    return mask


def hard_exudate_candidates(
    image: np.ndarray,
    fov_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Hard-exudate candidate mask (bright yellow clusters via top-hat)."""
    if fov_mask is None:
        fov_mask = retinal_fov_mask(image)
    # Yellow = high R and G, low B
    r = image[..., 0].astype(np.float32)
    g = image[..., 1].astype(np.float32)
    b = image[..., 2].astype(np.float32)
    yellow = (r + g) / 2 - b
    yellow = np.clip(yellow, 0, 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(yellow, cv2.MORPH_TOPHAT, kernel)
    _, mask = cv2.threshold(tophat, 25, 255, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8) * fov_mask
    return (mask > 0).astype(np.uint8)


def hemorrhage_candidates(
    image: np.ndarray,
    fov_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Hemorrhage candidate mask (large red-dark regions)."""
    if fov_mask is None:
        fov_mask = retinal_fov_mask(image)
    green = image[..., 1].astype(np.uint8)
    # Dark regions: inverse + morphological
    inv = 255 - green
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    tophat = cv2.morphologyEx(inv, cv2.MORPH_TOPHAT, kernel)
    _, mask = cv2.threshold(tophat, 30, 255, cv2.THRESH_BINARY)
    # Remove tiny specks (those are MAs) — keep only >100 px connected comps
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
    out = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, nlab):
        if stats[i, cv2.CC_STAT_AREA] >= 100:
            out[lab == i] = 1
    return out * fov_mask
