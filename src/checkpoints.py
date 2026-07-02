from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from src.config import ExpConfig


def resolve_checkpoint(
    cfg: ExpConfig,
    fallbacks: Sequence[str] = ("_pseudo.pth", "_swa.pth"),
) -> Path | None:
    """Locate an experiment's checkpoint, preferring the QWK-best weights.

    Tries ``<exp>_best.pth`` first, then each suffix in ``fallbacks`` (in order).
    Returns ``None`` when nothing is found so callers can warn-and-skip rather
    than crash — the behaviour every ensemble/eval path already relied on.
    """
    best = cfg.ckpt_dir / f"{cfg.exp_name}_best.pth"
    if best.exists():
        return best
    for suffix in fallbacks:
        candidate = cfg.ckpt_dir / f"{cfg.exp_name}{suffix}"
        if candidate.exists():
            return candidate
    return None
