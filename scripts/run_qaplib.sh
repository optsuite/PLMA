#!/usr/bin/env bash
# Evaluate PLMA on the QAPLIB benchmark.
# Usage: Run from the project root directory after activating the conda environment.

echo "Experiment started at $(date)"
echo "Running experiments on QAPLIB"
python -u driver/eval_qaplib.py \
    --checkpoint ./pretrained/qaplib.pth \
    --clipping_value 11.0 --temperature 1.0 \
    -r 10 --output_dir ./results/main/qaplib
