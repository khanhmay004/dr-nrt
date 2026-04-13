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
    ) -> None:
        self.img_dir = img_dir
        self.transform = transform
        self.is_regression = is_regression

        self.samples: list[tuple[str, int]] = []
        for code in id_codes:
            img_path = img_dir / f"{code}.png"
            if img_path.exists():
                self.samples.append((code, labels[code]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        code, label = self.samples[idx]
        img_path = self.img_dir / f"{code}.png"

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

    train_ds = DRDataset(
        train_codes, train_labels, TRAIN_IMG_DIR,
        transform=transform_train, is_regression=cfg.is_regression,
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
