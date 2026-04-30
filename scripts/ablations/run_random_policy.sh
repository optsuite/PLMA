#!/usr/bin/env bash
# Evaluate PLMA with random policy (no pretrained checkpoint).
# Tests on Taixxeyy and Uniform n=100 datasets.
# Usage: Run from the project root directory after activating the conda environment.

echo "Random policy evaluation started at $(date)"

# --- Taixxeyy ---
echo "=== Running random policy on Taixxeyy ==="
python -u driver/eval_tai.py \
    --clipping_value 0.5 \
    -r 10 --output_dir ./results/misc/pretrain_vs_random/random_policy/tai

# --- Uniform n=100 ---
echo "=== Running random policy on Uniform n=100 ==="
python -u driver/eval.py \
    --dataset uniform --n 100 \
    --embed_dim 256 --num_gcn_layers 10 \
    --clipping_value 11.0 --temperature 1.0 \
    --save_path ./results/misc/pretrain_vs_random/random_policy/uniform/uniform100.csv

echo "Random policy evaluation completed at $(date)"
