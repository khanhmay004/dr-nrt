from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE
from src.dataset import DRDataset, ben_graham_preprocess
from src.transforms import get_tta_transforms


@torch.no_grad()
def predict_with_tta(
    model: nn.Module,
    dataset: DRDataset,
    device: torch.device,
) -> tuple[np.ndarray, list[str]]:
    model.eval()
    tta_transforms = get_tta_transforms()

    all_preds: list[np.ndarray] = []
    all_codes: list[str] = []

    for idx in range(len(dataset)):
        code, label = dataset.samples[idx]
        img_path = dataset.img_dir / f"{code}.png"
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = ben_graham_preprocess(image, IMAGE_SIZE)

        view_preds: list[float] = []
        for tta_t in tta_transforms:
            aug = tta_t(image=image)
            aug_img = aug["image"]

            img_t = torch.from_numpy(aug_img.transpose(2, 0, 1)).float() / 255.0
            for c in range(3):
                img_t[c] = (img_t[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]

            img_t = img_t.unsqueeze(0).to(device)
            output = model(img_t)
            view_preds.append(output.squeeze().cpu().item())

        avg_pred = np.mean(view_preds)
        all_preds.append(avg_pred)
        all_codes.append(code)

    return np.array(all_preds), all_codes


@torch.no_grad()
def predict_no_tta(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_regression: bool,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    raw_preds_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []
    codes_list: list[str] = []

    for batch in loader:
        images, targets, codes = batch[0], batch[1], batch[2]
        images = images.to(device)
        outputs = model(images)

        if is_regression:
            raw_preds_list.append(outputs.squeeze(1).cpu().numpy())
        else:
            raw_preds_list.append(outputs.cpu().numpy())

        targets_list.append(targets.cpu().numpy())
        codes_list.extend(codes)

    raw_preds = np.concatenate(raw_preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    return raw_preds, targets, codes_list
