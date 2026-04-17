"""Nearest-Class-Mean classifier on any backbone checkpoint.
Usage:
    python scripts/ncm_eval.py --exp 701 --ckpt checkpoints/exp701_.../*_best.pth
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config, NUM_CLASSES
from src.dataset import build_datasets
from src.models import build_model
from src.evaluate import compute_metrics, save_confusion_matrix


def extract_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Forward pre-FC; captures features via hook on avgpool/GeM."""
    feats_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    codes_list: list[str] = []
    model.eval()
    bucket: dict[str, torch.Tensor] = {}

    def hook(_: torch.nn.Module, __: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        bucket["f"] = out.detach().flatten(1)

    h = model.avgpool.register_forward_hook(hook)
    with torch.no_grad():
        for batch in loader:
            imgs, y, c = batch[0], batch[1], batch[2]
            _ = model(imgs.to(device))
            feats_list.append(F.normalize(bucket["f"], dim=1).cpu())
            labels_list.append(y.cpu())
            codes_list.extend(c)
    h.remove()
    return torch.cat(feats_list), torch.cat(labels_list), codes_list


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=int, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = get_config(args.exp)
    cfg.use_tta = False
    train_ds, val_ds, test_ds = build_datasets(cfg)

    model = build_model(cfg).to(args.device)
    sd = torch.load(args.ckpt, map_location=args.device)
    model.load_state_dict(sd, strict=False)

    train_ds.transform = val_ds.transform
    loaders = {
        "train": DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=4),
        "val":   DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=4),
        "test":  DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=4),
    }

    feats_tr, y_tr, _ = extract_features(model, loaders["train"], torch.device(args.device))

    mus: list[torch.Tensor] = []
    for k in range(NUM_CLASSES):
        mask = y_tr == k
        mu_k = feats_tr[mask].mean(0)
        mu_k = F.normalize(mu_k, dim=0)
        mus.append(mu_k)
    mu = torch.stack(mus, dim=0)

    for split in ("val", "test"):
        feats, y, codes = extract_features(
            model, loaders[split], torch.device(args.device),
        )
        sim = feats @ mu.T
        preds = sim.argmax(dim=1).numpy()
        probs = F.softmax(sim, dim=1).numpy()
        metrics = compute_metrics(y.numpy(), preds, y_pred_probs=probs)
        print(
            f"[{split}] " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        )

        out_dir = Path(cfg.results_dir) / "ncm"
        out_dir.mkdir(parents=True, exist_ok=True)
        save_confusion_matrix(y.numpy(), preds, out_dir / f"{split}_cm.png")


if __name__ == "__main__":
    main()
