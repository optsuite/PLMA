#!/usr/bin/env bash
# Evaluate PLMA on the Taixxeyy benchmark instances.
# Usage: Run from the project root directory after activating the conda environment.

echo "Experiment started at $(date)"
echo "Running experiments on Taixxeyy instances"
python -u driver/eval_tai.py \
    --checkpoint ./pretrained/qaplib.pth \
    --clipping_value 0.5 \
    -r 10 --output_dir ./results/main/tai
