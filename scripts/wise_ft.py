"""WiSE-FT α-sweep: interpolate pretrained backbone with fine-tuned checkpoint.
Usage:
    python scripts/wise_ft.py --exp 701 \
        --pretrained checkpoints/exp200_.../backbone.pth \
        --finetuned checkpoints/exp701_.../*_best.pth \
        --alphas 0.0 0.2 0.3 0.5 0.7 0.8 1.0
"""
import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.dataset import build_datasets
from src.models import build_model
from src.train import evaluate_on_test


def interpolate(
    sd_pre: dict[str, torch.Tensor],
    sd_ft: dict[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    """θ(α) = (1-α)·θ_pre + α·θ_ft. Non-overlapping keys → take ft."""
    out: dict[str, torch.Tensor] = {}
    for k, v_ft in sd_ft.items():
        if k in sd_pre and sd_pre[k].shape == v_ft.shape:
            out[k] = (1 - alpha) * sd_pre[k].float() + alpha * v_ft.float()
            out[k] = out[k].to(v_ft.dtype)
        else:
            out[k] = v_ft
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=int, required=True,
                    help="Config exp_id of the fine-tuned run (e.g., 701 for H1)")
    ap.add_argument("--pretrained", required=True,
                    help="Path to backbone checkpoint (e.g., A2 backbone)")
    ap.add_argument("--finetuned", required=True,
                    help="Path to fine-tuned checkpoint (e.g., H1 best.pth)")
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = get_config(args.exp)
    cfg.use_tta = False
    _, val_ds, test_ds = build_datasets(cfg)

    sd_pre = torch.load(args.pretrained, map_location=args.device)
    sd_ft = torch.load(args.finetuned, map_location=args.device)

    model = build_model(cfg).to(args.device)
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True,
    )

    results: list[tuple[float, dict[str, float]]] = []
    for alpha in args.alphas:
        sd_alpha = interpolate(sd_pre, sd_ft, alpha)
        model.load_state_dict(sd_alpha, strict=False)
        model.eval()
        cfg.eval_suffix = f"wiseft_a{alpha:.2f}"
        metrics = evaluate_on_test(
            model, test_ds, test_loader, cfg, torch.device(args.device),
            val_loader=val_loader,
        )
        results.append((alpha, metrics))
        print(
            f"[α={alpha:.2f}] QWK={metrics['qwk']:.4f}  "
            f"MacroF1={metrics['macro_f1']:.4f}  "
            f"Severe={metrics.get('f1_Severe', 0.0):.4f}"
        )

    out = Path(cfg.results_dir) / "wiseft_sweep.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write("alpha,qwk,macro_f1,severe_f1,moderate_f1,auc_roc,ece\n")
        for a, m in results:
            f.write(
                f"{a},{m['qwk']},{m['macro_f1']},"
                f"{m.get('f1_Severe', 0.0)},{m.get('f1_Moderate', 0.0)},"
                f"{m.get('auc_roc', 0.0)},{m.get('ece', 0.0)}\n"
            )
    print(f"\nSweep results saved to {out}")


if __name__ == "__main__":
    main()
