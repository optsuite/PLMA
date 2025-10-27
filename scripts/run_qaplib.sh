#!/usr/bin/env bash
conda activate plma

echo "Experiment started at $(date)"
echo "Running experiments on QAPLIB"
python -u driver/eval_qaplib.py --checkpoint ./pretrained/qaplib.pth --dataset qaplib --clipping_value 11.0 --temperature 1.0 -r 10 --output_dir ./results/qaplib
