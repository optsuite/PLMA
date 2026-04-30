#!/usr/bin/env bash
# Ablation studies on Taixxeyy benchmark: AR, No-Attention, No-Sinkhorn, Gradient-Free.
# Usage: Run from the project root directory after activating the conda environment.

echo "Ablation experiments started at $(date)"

# --- 1. Autoregressive (AR) ---
echo "=== Running AR ablation ==="
python -u driver/eval_tai.py \
    --checkpoint ./pretrained/qaplib.pth \
    --clipping_value 0.5 --ar \
    -r 10 --output_dir ./results/misc/ablations/tai/ar

# --- 2. No-Attention (num_att_layers=0, num_gcn_layers=20) ---
echo "=== Running No-Attention ablation ==="
python -u driver/eval_tai.py \
    --checkpoint ./pretrained/ablations/uniform100-no-att.pth \
    --num_att_layers 0 --num_gcn_layers 20 \
    --clipping_value 0.5 \
    -r 10 --output_dir ./results/misc/ablations/tai/no-attn

# --- 3. No-Sinkhorn (num_iterations=0) ---
echo "=== Running No-Sinkhorn ablation ==="
python -u driver/eval_tai.py \
    --checkpoint ./pretrained/qaplib.pth \
    --num_iterations 0 \
    --clipping_value 0.5 \
    -r 10 --output_dir ./results/misc/ablations/tai/no-sinkhorn

# --- 4. Gradient-Free (lr=0) ---
echo "=== Running Gradient-Free ablation ==="
python -u driver/eval_tai.py \
    --checkpoint ./pretrained/qaplib.pth \
    --lr 0 \
    --clipping_value 0.5 \
    -r 10 --output_dir ./results/misc/ablations/tai/gd-free

echo "All ablation experiments completed at $(date)"
