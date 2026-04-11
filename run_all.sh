#!/bin/bash
# ==============================================================
# Run Experiments 2–14 sequentially, then archive & upload results
# Usage:  bash run_all.sh
# ==============================================================
set -e  # exit on first failure

LOGFILE="run_5_7.log"

echo "=============================================" | tee -a "$LOGFILE"
echo " DR-NRT  —  Experiments 5 → 7  (batch run)" | tee -a "$LOGFILE"
echo " Started: $(date)"                             | tee -a "$LOGFILE"
echo "=============================================" | tee -a "$LOGFILE"

for EXP in $(seq 5 7); do
    echo "" | tee -a "$LOGFILE"
    echo ">>> Experiment $EXP  —  $(date)" | tee -a "$LOGFILE"
    python run_experiment.py --exp "$EXP" 2>&1 | tee -a "$LOGFILE"
    echo "<<< Experiment $EXP finished  —  $(date)" | tee -a "$LOGFILE"
done

echo "" | tee -a "$LOGFILE"
echo "=============================================" | tee -a "$LOGFILE"
echo " All experiments finished  —  $(date)"        | tee -a "$LOGFILE"
echo " Archiving results ..."                        | tee -a "$LOGFILE"
echo "=============================================" | tee -a "$LOGFILE"

tar -czf /workspace/run_5_7.tar.gz results/ checkpoints/
echo "Archive created: /workspace/run_5_7.tar.gz" | tee -a "$LOGFILE"

echo "Uploading to Google Drive ..." | tee -a "$LOGFILE"
rclone copy /workspace/run_5_7.tar.gz gdrive:DR-Results-Exp -P
echo "Upload complete  —  $(date)" | tee -a "$LOGFILE"
