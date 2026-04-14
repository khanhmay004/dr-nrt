#!/bin/bash
# run_phase_subA.sh — Re-run A0c (Level 1.5 oversample) + A1-v2 (fixed contrastive).
# Execute from project root: bash run_phase_subA.sh

set -euo pipefail


echo "====================================================="
echo "  Phase A v2: A0c + A1 re-run with fixes"
echo "  Started: $(date)"
echo "====================================================="

# ── Step 1: Regenerate oversampled images with Level 1.5 aug ────────────────
echo ""
echo "─── Clearing old oversampled images ─────────────────────────────"
rm -rf data/train_oversampled
echo "─── Generating oversampled images (Level 1.5, target=1000) ──────"
python scripts/offline_oversample.py
echo "    Done: $(date)"

# ── A0c-v2: Train with Level 1.5 oversampled data ──────────────────────────
echo ""
echo "─── [A0c-v2] Exp 102 — Offline Oversample (50 epochs) ──────────"
python run_experiment.py --exp 102
echo "    Done: $(date)"

# ── A1-v2: OrdSupCon with offline oversample, 30 contrastive + 60 fine-tune ─
echo ""
echo "─── [A1-v2] Exp 103 — OrdSupCon APTOS ──────────────────────────"
echo "            Stage 1: 30 contrastive epochs"
echo "            Stage 2: 60 fine-tune epochs (freeze=2)"
python run_experiment.py --exp 103
echo "    Done: $(date)"

echo ""
echo "====================================================="
echo "  Phase A v2 Complete: $(date)"
echo "  Artifacts → results/exp102_*/ and results/exp103_*/"
echo "====================================================="