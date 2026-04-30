#!/usr/bin/env bash
# Evaluate PLMA on large-scale synthetic QAP datasets (zero-shot generalization).
# Usage: Run from the project root directory after activating the conda environment.

# SAWT n=200
python -u driver/eval.py --dataset sawt --n 200 --embed_dim 256 --num_gcn_layers 10     --checkpoint ./pretrained/uniform100.pth --clipping_value 0.5     -K 10 -M 10 -T 50     --local_search_iter 400 --num_actions 400     --save_path ./results/main/large-scale/sawt200.csv

# SAWT n=500
python -u driver/eval.py --dataset sawt --n 500 --embed_dim 256 --num_gcn_layers 10     --checkpoint ./pretrained/uniform100.pth --clipping_value 1.5     -K 10 -M 10 -T 50     --local_search_iter 500 --num_actions 1000     --save_path ./results/main/large-scale/sawt500.csv
