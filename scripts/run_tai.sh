#!/usr/bin/env bash
conda activate plma

echo "Experiment started at $(date)"
echo "Running experiments on Taixxeyy instances"
python -u driver/eval_qaplib.py --checkpoint ./pretrained/qaplib.pth --dataset tai --clipping_value 0.5 -r 10 --output_dir ./results/tai
