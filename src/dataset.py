from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMAGE_SIZE,
    TEST_CSV,
    TEST_IMG_DIR,
    TRAIN_CSV,
    TRAIN_IMG_DIR,
    ExpConfig,
)


def ben_graham_preprocess(image: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        image = image[y : y + h, x : x + w]

    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)

    gauss = cv2.GaussianBlur(image, (0, 0), sigmaX=size / 30)
    image = cv2.addWeighted(image, 4, gauss, -4, 128)

    return image


def load_labels(csv_path: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["id_code"]] = int(row["diagnosis"])
    return labels


class DRDataset(Dataset):
    def __init__(
        self,
        id_codes: list[str],
        labels: dict[str, int],
        img_dir: Path,
        transform: object | None = None,
        is_regression: bool = False,
        extra_img_dir: Path | None = None,
        extra_img_dirs: list[Path] | None = None,
    ) -> None:
        self.img_dir = img_dir
        # Support both legacy single extra_img_dir and new extra_img_dirs list
        self.extra_img_dirs: list[Path] = []
        if extra_img_dirs:
            self.extra_img_dirs = list(extra_img_dirs)
        elif extra_img_dir is not None:
            self.extra_img_dirs = [extra_img_dir]
        # Keep legacy attr for ContrastiveDRDataset compatibility
        self.extra_img_dir = self.extra_img_dirs[0] if self.extra_img_dirs else None
        self.transform = transform
        self.is_regression = is_regression

        self.samples: list[tuple[str, int]] = []
        for code in id_codes:
            if self._find_image(code) is not None:
                self.samples.append((code, labels[code]))

    def _find_image(self, code: str) -> Path | None:
        """Locate an image across primary and extra directories."""
        img_path = self.img_dir / f"{code}.png"
        if img_path.exists():
            return img_path
        for d in self.extra_img_dirs:
            p = d / f"{code}.png"
            if p.exists():
                return p
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        code, label = self.samples[idx]
        img_path = self._find_image(code)

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = ben_graham_preprocess(image, IMAGE_SIZE)

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]

        image_t = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        for c in range(3):
            image_t[c] = (image_t[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]

        if self.is_regression:
            target = torch.tensor(label, dtype=torch.float32)
        else:
            target = torch.tensor(label, dtype=torch.long)

        return image_t, target, code


class PseudoLabelDataset(Dataset):
    def __init__(
        self,
        real_dataset: DRDataset,
        pseudo_codes: list[str],
        pseudo_labels: dict[str, float],
        pseudo_img_dir: Path,
        transform: object | None = None,
        pseudo_weight: float = 0.5,
        is_regression: bool = False,
    ) -> None:
        self.real_dataset = real_dataset
        self.pseudo_weight = pseudo_weight
        self.pseudo_img_dir = pseudo_img_dir
        self.transform = transform
        self.is_regression = is_regression

        self.pseudo_samples: list[tuple[str, float]] = []
        for code in pseudo_codes:
            img_path = pseudo_img_dir / f"{code}.png"
            if img_path.exists():
                self.pseudo_samples.append((code, pseudo_labels[code]))

        self.n_real = len(self.real_dataset)
        self.n_pseudo = len(self.pseudo_samples)

    def __len__(self) -> int:
        return self.n_real + self.n_pseudo

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, str, float]:
        if idx < self.n_real:
            img, target, code = self.real_dataset[idx]
            return img, target, code, 1.0

        pseudo_idx = idx - self.n_real
        code, soft_label = self.pseudo_samples[pseudo_idx]
        img_path = self.pseudo_img_dir / f"{code}.png"

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = ben_graham_preprocess(image, IMAGE_SIZE)

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]

        image_t = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        for c in range(3):
            image_t[c] = (image_t[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]

        target = torch.tensor(soft_label, dtype=torch.float32) if self.is_regression \
            else torch.tensor(int(soft_label), dtype=torch.long)
        return image_t, target, code, self.pseudo_weight


class ContrastiveDRDataset(Dataset):
    """Wraps a DRDataset to produce two augmented views per image for contrastive learning."""

    def __init__(self, base_dataset: DRDataset, transform: object) -> None:
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        code, label = self.base_dataset.samples[idx]
        img_path = self.base_dataset.img_dir / f"{code}.png"
        if not img_path.exists() and self.base_dataset.extra_img_dir is not None:
            img_path = self.base_dataset.extra_img_dir / f"{code}.png"

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = ben_graham_preprocess(image, IMAGE_SIZE)

        # Two independent augmented views
        view1 = self._to_tensor(self.transform(image=image)["image"])
        view2 = self._to_tensor(self.transform(image=image)["image"])

        target = torch.tensor(label, dtype=torch.long)
        return view1, view2, target, code

    @staticmethod
    def _to_tensor(image: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        for c in range(3):
            t[c] = (t[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        return t


def build_datasets(
    cfg: ExpConfig,
    transform_train: object | None = None,
    transform_val: object | None = None,
) -> tuple[DRDataset, DRDataset, DRDataset]:
    from sklearn.model_selection import train_test_split

    train_labels = load_labels(TRAIN_CSV)
    test_labels = load_labels(TEST_CSV)

    all_codes = list(train_labels.keys())
    all_targets = [train_labels[c] for c in all_codes]

    train_codes, val_codes = train_test_split(
        all_codes,
        test_size=cfg.val_ratio,
        stratify=all_targets,
        random_state=cfg.seed,
    )

    # If offline oversampling is enabled, add oversampled image codes
    oversample_codes: list[str] = []
    if cfg.oversample_target > 0 and cfg.oversample_dir:
        oversample_path = Path(cfg.oversample_dir)
        if oversample_path.exists():
            for img_file in oversample_path.glob("*.png"):
                code = img_file.stem
                # Oversampled files encode original label in filename: {orig_code}_aug{N}_{label}
                parts = code.rsplit("_", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    label = int(parts[1])
                    train_labels[code] = label
                    oversample_codes.append(code)

    # IDRiD supplement: inject preprocessed Grade 3+4 images into training set
    idrid_codes: list[str] = []
    idrid_img_dir: Path | None = None
    if cfg.use_idrid_supplement and cfg.idrid_csv and cfg.idrid_processed_dir:
        idrid_path = Path(cfg.idrid_processed_dir)
        if idrid_path.exists():
            idrid_labels = load_labels(Path(cfg.idrid_csv))
            for code, label in idrid_labels.items():
                img_file = idrid_path / f"{code}.png"
                if img_file.exists():
                    train_labels[code] = label
                    idrid_codes.append(code)
            idrid_img_dir = idrid_path

    # Determine extra image directories (oversample + IDRiD may coexist)
    extra_dirs: list[Path] = []
    if oversample_codes and cfg.oversample_dir:
        extra_dirs.append(Path(cfg.oversample_dir))
    if idrid_codes and idrid_img_dir is not None:
        extra_dirs.append(idrid_img_dir)

    train_ds = DRDataset(
        train_codes + oversample_codes + idrid_codes, train_labels,
        TRAIN_IMG_DIR,
        transform=transform_train, is_regression=cfg.is_regression,
        extra_img_dirs=extra_dirs if extra_dirs else None,
    )
    val_ds = DRDataset(
        val_codes, train_labels, TRAIN_IMG_DIR,
        transform=transform_val, is_regression=cfg.is_regression,
    )
    test_ds = DRDataset(
        list(test_labels.keys()), test_labels, TEST_IMG_DIR,
        transform=transform_val, is_regression=cfg.is_regression,
    )

    return train_ds, val_ds, test_ds


def build_eyepacs_dataset(cfg: ExpConfig) -> DRDataset:
    """Build a DRDataset from pre-processed EyePACS images for contrastive pre-training.

    EyePACS CSV uses the same ``id_code,diagnosis`` format as APTOS, so
    ``load_labels`` and ``DRDataset`` work directly.  No transform is applied
    here — the caller wraps the result in ``ContrastiveDRDataset`` which
    handles dual-view augmentation.
    """
    eyepacs_labels = load_labels(Path(cfg.eyepacs_csv))
    eyepacs_codes = list(eyepacs_labels.keys())
    return DRDataset(
        eyepacs_codes, eyepacs_labels, Path(cfg.eyepacs_dir), transform=None,
    )
