#!/bin/bash
# run_phase_subA.sh — Run all Phase A experiments sequentially.
# Execute from project root: bash run_phase_subA.sh

set -euo pipefail

PYTHON=".venv/Scripts/python"

echo "====================================================="
echo "  Phase A: Ordinal Contrastive Experiments"
echo "  Started: $(date)"
echo "====================================================="

# Sanity: verify exp08 checkpoint exists for A0
EXP08_CKPT="checkpoints/exp08_gem/exp08_gem_best.pth"
if [ ! -f "$EXP08_CKPT" ]; then
    echo "WARNING: $EXP08_CKPT not found — skipping A0 (eval-only)"
    SKIP_A0=true
else
    SKIP_A0=false
fi

# ── A0: Eval-only — load Exp 8, report ECE ──────────────────────────────────
if [ "$SKIP_A0" = false ]; then
    echo ""
    echo "─── [A0]  Exp 100 — Baseline ECE (eval only) ───────────────"
    "$PYTHON" run_experiment.py --exp 100
    echo "    Done: $(date)"
fi

# ── A0b: WeightedRandomSampler ───────────────────────────────────────────────
echo ""
echo "─── [A0b] Exp 101 — Weighted Random Sampler (50 epochs) ────────"
"$PYTHON" run_experiment.py --exp 101
echo "    Done: $(date)"

# ── A0c: Offline oversample → train ─────────────────────────────────────────
echo ""
echo "─── [A0c] Generating oversampled images (target=1000) ──────────"
"$PYTHON" scripts/offline_oversample.py
echo ""
echo "─── [A0c] Exp 102 — Offline Oversample training (50 epochs) ────"
"$PYTHON" run_experiment.py --exp 102
echo "    Done: $(date)"

# ── A1: OrdSupCon pre-train + fine-tune ─────────────────────────────────────
echo ""
echo "─── [A1]  Exp 103 — OrdSupCon APTOS ────────────────────────────"
echo "          Stage 1: 50 contrastive epochs"
echo "          Stage 2: 60 fine-tuning epochs"
"$PYTHON" run_experiment.py --exp 103
echo "    Done: $(date)"

echo ""
echo "====================================================="
echo "  Phase A Complete: $(date)"
echo "  Artifacts → results/exp10{0,1,2,3}_*/"
echo "====================================================="