"""EDA helpers: perceptual hashing, MMD, feature extraction, linear probe."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Perceptual hashing (pHash) — near-duplicate detection
# ---------------------------------------------------------------------------


def phash(image: np.ndarray, hash_size: int = 16) -> np.ndarray:
    """Perceptual hash via DCT. Returns a packed bit vector."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    img = cv2.resize(gray, (hash_size * 4, hash_size * 4), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)
    dct = cv2.dct(img)
    low = dct[:hash_size, :hash_size]
    # Exclude DC term from median
    med = np.median(low.flatten()[1:])
    bits = (low > med).astype(np.uint8).flatten()
    return bits


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a != b))


def find_near_duplicates(
    hashes: dict[str, np.ndarray],
    threshold: int = 5,
) -> list[tuple[str, str, int]]:
    """All id-code pairs with Hamming distance <= threshold.

    O(N^2) — fine for ~5000 images, use LSH if scaling up.
    """
    keys = list(hashes.keys())
    mat = np.stack([hashes[k] for k in keys])  # (N, B)
    pairs: list[tuple[str, str, int]] = []
    for i in range(len(keys)):
        diff = np.sum(mat[i:i + 1] != mat[i + 1:], axis=1)
        for j_rel, d in enumerate(diff):
            if d <= threshold:
                pairs.append((keys[i], keys[i + 1 + j_rel], int(d)))
    return pairs


# ---------------------------------------------------------------------------
# Maximum Mean Discrepancy (MMD) — two-sample distribution test
# ---------------------------------------------------------------------------


def mmd_rbf(X: np.ndarray, Y: np.ndarray, sigma: float | None = None) -> float:
    """MMD^2 with RBF kernel. Biased estimator; fine for relative comparison."""
    XX = X @ X.T
    YY = Y @ Y.T
    XY = X @ Y.T
    x_sq = np.diag(XX)
    y_sq = np.diag(YY)
    d_xx = x_sq[:, None] + x_sq[None, :] - 2 * XX
    d_yy = y_sq[:, None] + y_sq[None, :] - 2 * YY
    d_xy = x_sq[:, None] + y_sq[None, :] - 2 * XY
    if sigma is None:
        # median heuristic on pooled distances
        pooled = np.concatenate([
            d_xx[np.triu_indices_from(d_xx, k=1)],
            d_yy[np.triu_indices_from(d_yy, k=1)],
            d_xy.flatten(),
        ])
        sigma = np.sqrt(0.5 * np.median(pooled[pooled > 0]) + 1e-8)
    gamma = 1.0 / (2 * sigma ** 2)
    k_xx = np.exp(-gamma * d_xx).mean()
    k_yy = np.exp(-gamma * d_yy).mean()
    k_xy = np.exp(-gamma * d_xy).mean()
    return float(k_xx + k_yy - 2 * k_xy)


def mmd_permutation_test(
    X: np.ndarray,
    Y: np.ndarray,
    n_permutations: int = 500,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Permutation-based p-value for MMD^2 between X and Y. Returns (mmd, p)."""
    if rng is None:
        rng = np.random.default_rng(42)
    observed = mmd_rbf(X, Y)
    Z = np.vstack([X, Y])
    n = len(X)
    count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(len(Z))
        Xp, Yp = Z[perm[:n]], Z[perm[n:]]
        if mmd_rbf(Xp, Yp) >= observed:
            count += 1
    p = (count + 1) / (n_permutations + 1)
    return observed, p


# ---------------------------------------------------------------------------
# ImageNet ResNet-50 feature extraction (for MMD + linear probes)
# ---------------------------------------------------------------------------


def load_imagenet_resnet50(device: str = "cuda"):
    """Pretrained ResNet-50 with classifier head removed. Returns (model, preproc)."""
    import torch
    import torchvision.models as tvm

    weights = tvm.ResNet50_Weights.IMAGENET1K_V2
    model = tvm.resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    preproc = weights.transforms()
    return model, preproc


def extract_features(
    images: Iterable[np.ndarray],
    model,
    preproc,
    device: str = "cuda",
    batch_size: int = 32,
) -> np.ndarray:
    """Extract 2048-dim ResNet-50 features from an iterable of RGB uint8 images."""
    import torch
    from PIL import Image

    feats: list[np.ndarray] = []
    batch: list = []
    for img in images:
        pil = Image.fromarray(img)
        batch.append(preproc(pil))
        if len(batch) == batch_size:
            with torch.no_grad():
                b = torch.stack(batch).to(device)
                f = model(b).cpu().numpy()
            feats.append(f)
            batch = []
    if batch:
        with torch.no_grad():
            b = torch.stack(batch).to(device)
            f = model(b).cpu().numpy()
        feats.append(f)
    return np.vstack(feats) if feats else np.zeros((0, 2048), dtype=np.float32)


# ---------------------------------------------------------------------------
# Linear probe — frozen features + logistic regression, evaluate with QWK
# ---------------------------------------------------------------------------


def linear_probe_qwk(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    C: float = 1.0,
) -> dict[str, float]:
    """Fit multinomial logistic regression on frozen features; return metrics."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score

    clf = LogisticRegression(
        C=C, max_iter=1000, class_weight="balanced",
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_val)
    return {
        "qwk": cohen_kappa_score(y_val, pred, weights="quadratic"),
        "macro_f1": f1_score(y_val, pred, average="macro"),
        "accuracy": accuracy_score(y_val, pred),
    }


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def read_rgb(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
