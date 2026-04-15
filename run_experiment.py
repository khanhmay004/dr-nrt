from __future__ import annotations

import argparse
import logging
import sys

import torch
from torch.utils.data import DataLoader

from src.config import get_config
from src.dataset import build_datasets, build_eyepacs_dataset, ContrastiveDRDataset
from src.transforms import get_train_transform, get_val_transform
from src.train import run_training, evaluate_on_test, run_contrastive_pretraining
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
    parser.add_argument("--exp", type=int, required=True, help="Experiment ID")
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

    # --- A0: eval-only (load checkpoint, no training) ---
    if cfg.load_checkpoint and not cfg.use_contrastive_pretrain:
        import torch as _torch
        model = build_model(cfg).to(device)
        model.load_state_dict(_torch.load(cfg.load_checkpoint, weights_only=True))
        logger.info(f"Loaded checkpoint: {cfg.load_checkpoint}")

        test_loader = DataLoader(
            test_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )
        metrics = evaluate_on_test(
            model, test_ds, test_loader, cfg, device,
            val_loader=val_loader if cfg.use_optimized_thresholds else None,
        )
        logger.info(f"=== Experiment {cfg.exp_name} Complete (eval only) ===")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")
        return

    # --- Build DataLoaders ---
    sampler = None
    shuffle_train = True
    if cfg.use_weighted_sampler:
        from torch.utils.data import WeightedRandomSampler
        targets = [s[1] for s in train_ds.samples]
        class_counts = [0] * 5
        for t in targets:
            class_counts[t] += 1
        weights = [1.0 / class_counts[t] for t in targets]
        sampler = WeightedRandomSampler(weights, num_samples=len(targets), replacement=True)
        shuffle_train = False
        logger.info(f"Using WeightedRandomSampler — class counts: {class_counts}")

    # Wrap train dataset for joint contrastive (dual-view augmentation)
    train_ds_for_loader = train_ds
    if cfg.use_joint_contrastive:
        train_ds_for_loader = ContrastiveDRDataset(train_ds, transform_train)
        logger.info(f"Joint contrastive — dual-view training ({len(train_ds_for_loader)} samples)")

    train_loader = DataLoader(
        train_ds_for_loader, batch_size=cfg.batch_size, shuffle=shuffle_train,
        sampler=sampler,
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

    # --- A1/A2: Contrastive pre-training → supervised fine-tuning ---
    pretrained_backbone_sd = None
    if cfg.use_contrastive_pretrain:
        if cfg.contrastive_data == "eyepacs":
            eyepacs_ds = build_eyepacs_dataset(cfg)
            contrastive_ds = ContrastiveDRDataset(eyepacs_ds, transform_train)
            logger.info(f"Contrastive data: EyePACS ({len(eyepacs_ds)} images)")
        else:
            contrastive_ds = ContrastiveDRDataset(train_ds, transform_train)
            logger.info(f"Contrastive data: APTOS ({len(train_ds)} images)")

        contrastive_sampler = None
        contrastive_shuffle = True
        from torch.utils.data import WeightedRandomSampler as WRS
        c_targets = [s[1] for s in contrastive_ds.base_dataset.samples]
        c_counts = [0] * 5
        for t in c_targets:
            c_counts[t] += 1
        c_weights = [1.0 / c_counts[t] for t in c_targets]
        contrastive_sampler = WRS(c_weights, num_samples=len(c_targets), replacement=True)
        contrastive_shuffle = False
        logger.info(f"Contrastive WRS — class counts: {c_counts}")

        contrastive_loader = DataLoader(
            contrastive_ds, batch_size=cfg.batch_size,
            shuffle=contrastive_shuffle, sampler=contrastive_sampler,
            num_workers=args.workers, pin_memory=True,
        )
        pretrained_backbone_sd = run_contrastive_pretraining(cfg, contrastive_loader, device)

        # Free VRAM from contrastive stage before fine-tuning
        del contrastive_loader, contrastive_ds
        if cfg.contrastive_data == "eyepacs":
            del eyepacs_ds
        if device.type == "cuda":
            torch.cuda.empty_cache()
            logger.info("Cleared VRAM after contrastive pre-training")
    elif cfg.load_backbone:
        pretrained_backbone_sd = torch.load(cfg.load_backbone, weights_only=True)
        logger.info(f"Loaded backbone from {cfg.load_backbone}")

    model = run_training(cfg, train_loader, val_loader, device, pretrained_backbone_sd)

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
