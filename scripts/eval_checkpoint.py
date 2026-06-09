"""Evaluate a trained checkpoint on val+test.
Usage:
    python scripts/eval_checkpoint.py --exp 701 --ckpt checkpoints/exp701_*/*_best.pth
    python scripts/eval_checkpoint.py --exp 300 --ckpt <path> --thresh cumulative
"""
import argparse
from pathlib import Path
import torch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch.utils.data import DataLoader
from src.config import get_config
from src.dataset import build_datasets
from src.models import build_model
from src.train import evaluate_on_test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=int, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--thresh", choices=["argmax", "expected", "cumulative"],
                    default="argmax")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--suffix", type=str, default="eval",
                    help="Suffix for output files to avoid clobbering training outputs")
    args = ap.parse_args()

    cfg = get_config(args.exp)
    cfg.use_tta = False
    cfg.use_optimized_thresholds = args.thresh != "argmax"
    cfg.threshold_strategy = args.thresh if args.thresh != "argmax" else None
    cfg.eval_suffix = args.suffix

    _, val_ds, test_ds = build_datasets(cfg)
    model = build_model(cfg).to(args.device)
    sd = torch.load(args.ckpt, map_location=args.device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True,
    )

    metrics = evaluate_on_test(
        model, test_ds, test_loader, cfg, torch.device(args.device),
        val_loader=val_loader if cfg.use_optimized_thresholds else None,
    )
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()