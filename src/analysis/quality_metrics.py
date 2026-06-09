"""Image-quality metrics for fundus images.

All metrics are computed on a retinal-FOV-masked region so that the dark
background does not dominate statistics.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.analysis.fundus_cv import retinal_fov_mask


def laplacian_blur(image: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Variance of the Laplacian — standard blur proxy (higher = sharper)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    if mask is None:
        mask = retinal_fov_mask(image)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    vals = lap[mask.astype(bool)]
    if vals.size == 0:
        return 0.0
    return float(vals.var())


def mean_intensity(image: np.ndarray, mask: np.ndarray | None = None) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    if mask is None:
        mask = retinal_fov_mask(image)
    vals = gray[mask.astype(bool)]
    return float(vals.mean()) if vals.size else 0.0


def contrast_std(image: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Intra-FOV intensity std — simple global-contrast proxy."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    if mask is None:
        mask = retinal_fov_mask(image)
    vals = gray[mask.astype(bool)]
    return float(vals.std()) if vals.size else 0.0


def snr_estimate(image: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Crude SNR: mean / std over FOV on green channel."""
    if image.ndim == 3:
        g = image[..., 1]
    else:
        g = image
    if mask is None:
        mask = retinal_fov_mask(image)
    vals = g[mask.astype(bool)].astype(np.float32)
    if vals.size == 0 or vals.std() == 0:
        return 0.0
    return float(vals.mean() / vals.std())


def radial_intensity_profile(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    n_bins: int = 20,
) -> np.ndarray:
    """Mean intensity as a function of radial distance from the FOV centroid.

    Returns a 1-D array of length ``n_bins`` normalized to [0, 1] along the
    radial axis. Useful for clustering images into illumination regimes.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    if mask is None:
        mask = retinal_fov_mask(image)
    mask_b = mask.astype(bool)
    if not mask_b.any():
        return np.zeros(n_bins, dtype=np.float32)

    ys, xs = np.where(mask_b)
    cy, cx = ys.mean(), xs.mean()
    r = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    r_max = r.max() + 1e-6
    r_norm = r / r_max

    intensities = gray[mask_b].astype(np.float32) / 255.0
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(r_norm, bins) - 1, 0, n_bins - 1)
    profile = np.zeros(n_bins, dtype=np.float32)
    counts = np.zeros(n_bins, dtype=np.int64)
    np.add.at(profile, idx, intensities)
    np.add.at(counts, idx, 1)
    counts = np.maximum(counts, 1)
    return profile / counts


def illumination_uniformity(image: np.ndarray, mask: np.ndarray | None = None) -> float:
    """How uniform the illumination is across the retina.

    1.0 = perfectly uniform; smaller values = strong center-to-edge falloff.
    Computed as 1 - (max - min) / mean on the radial intensity profile.
    """
    prof = radial_intensity_profile(image, mask=mask, n_bins=10)
    if prof.mean() == 0:
        return 0.0
    return float(1.0 - (prof.max() - prof.min()) / (prof.mean() + 1e-6))


def compute_all(image: np.ndarray) -> dict[str, float]:
    """Convenience: compute every quality metric at once (shares FOV mask)."""
    mask = retinal_fov_mask(image)
    return {
        "laplacian_var": laplacian_blur(image, mask),
        "mean_intensity": mean_intensity(image, mask),
        "contrast_std": contrast_std(image, mask),
        "snr": snr_estimate(image, mask),
        "illum_uniformity": illumination_uniformity(image, mask),
        "fov_fraction": float(mask.mean()),
    }
