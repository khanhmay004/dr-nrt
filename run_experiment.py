from __future__ import annotations

import argparse
import logging
import sys

import torch
from torch.utils.data import DataLoader

from src.config import get_config
from src.dataset import build_datasets
from src.transforms import get_train_transform, get_val_transform
from src.train import run_training, evaluate_on_test
from src.pseudo_label import generate_pseudo_labels, finetune_with_pseudo
from src.ensemble import build_ensemble_configs, run_ensemble_inference
from src.models import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _add_file_logger(exp_name: str, results_dir) -> None:
    """Save full run output to results/<exp>/<exp>.log automatically."""
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / f"{exp_name}.log"
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)
    logger.info(f"Logging to {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DR-NRT Experiment Runner")
    parser.add_argument("--exp", type=int, required=True, help="Experiment ID (0-14)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    cfg = get_config(args.exp)
    logger.info(f"Running experiment {cfg.exp_name}")
    _add_file_logger(cfg.exp_name, cfg.results_dir)

    if args.exp == 14:
        _run_ensemble(cfg, device, args.workers)
        return

    transform_train = get_train_transform(cfg.aug_level)
    transform_val = get_val_transform()

    train_ds, val_ds, test_ds = build_datasets(cfg, transform_train, transform_val)
    logger.info(f"Dataset sizes — Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    model = run_training(cfg, train_loader, val_loader, device)

    if cfg.use_pseudo_labels:
        pseudo_labels = generate_pseudo_labels(model, test_ds, test_loader, cfg, device)
        model = finetune_with_pseudo(model, train_ds, pseudo_labels, cfg, device)

    metrics = evaluate_on_test(
        model, test_ds, test_loader, cfg, device,
        val_loader=val_loader if cfg.use_optimized_thresholds else None,
    )

    logger.info(f"=== Experiment {cfg.exp_name} Complete ===")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")


def _run_ensemble(base_cfg, device: torch.device, workers: int) -> None:
    ensemble_configs = build_ensemble_configs(base_cfg)

    transform_train = get_train_transform(base_cfg.aug_level)
    transform_val = get_val_transform()

    for cfg in ensemble_configs:
        logger.info(f"--- Training ensemble member: {cfg.backbone} ---")
        train_ds, val_ds, test_ds = build_datasets(cfg, transform_train, transform_val)

        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=workers, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=workers, pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=workers, pin_memory=True,
        )

        model = run_training(cfg, train_loader, val_loader, device)

        if cfg.use_pseudo_labels:
            pseudo_labels = generate_pseudo_labels(model, test_ds, test_loader, cfg, device)
            model = finetune_with_pseudo(model, train_ds, pseudo_labels, cfg, device)

    _, val_ds, test_ds = build_datasets(base_cfg, None, transform_val)
    val_loader = DataLoader(val_ds, batch_size=base_cfg.batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=base_cfg.batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    metrics = run_ensemble_inference(
        ensemble_configs, test_ds, test_loader, device,
        val_loader=val_loader,
    )

    logger.info("=== Ensemble Experiment Complete ===")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
