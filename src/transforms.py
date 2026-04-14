from __future__ import annotations

import albumentations as A

from src.config import IMAGE_SIZE


def get_train_transform(aug_level: int) -> A.Compose | None:
    if aug_level == 0:
        return None

    base = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.15, rotate_limit=180, p=0.5,
            border_mode=0,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.5,
        ),
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.5,
        ),
    ]

    if aug_level >= 2:
        base.extend([
            A.CoarseDropout(
                max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.3,
            ),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
            A.ElasticTransform(alpha=120, sigma=6, p=0.2, border_mode=0),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2, border_mode=0),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        ])

    return A.Compose(base)


def get_offline_oversample_transform() -> A.Compose:
    """Level 1.5: DR-safe augmentation for offline oversampling.

    Keeps spatial + color transforms and mild CLAHE/blur.
    Drops CoarseDropout, ElasticTransform, GridDistortion that can
    destroy diagnostic pathology (microaneurysms, hemorrhages, vessels).
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.15, rotate_limit=180, p=0.5,
            border_mode=0,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.5,
        ),
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.5,
        ),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    ])


def get_val_transform() -> None:
    return None


def get_tta_transforms() -> list[A.Compose]:
    return [
        A.Compose([]),  # original
        A.Compose([A.HorizontalFlip(p=1.0)]),
        A.Compose([A.VerticalFlip(p=1.0)]),
        A.Compose([A.Rotate(limit=(90, 90), p=1.0, border_mode=0)]),
        A.Compose([A.Rotate(limit=(180, 180), p=1.0, border_mode=0)]),
        A.Compose([A.Rotate(limit=(270, 270), p=1.0, border_mode=0)]),
    ]
