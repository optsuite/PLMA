#!/usr/bin/env bash
# Evaluate PLMA on uniformly random synthetic QAP datasets.
# Usage: Run from the project root directory after activating the conda environment.

python driver/eval.py --n 20 --embed_dim 128 --num_gcn_layers 1 \
    --checkpoint ./pretrained/uniform20.pth --clipping_value 1 --temperature 4.0 \
    --save_path ./results/main/uniform/uniform20.csv

python driver/eval.py --n 50 --embed_dim 128 --num_gcn_layers 8 \
    --checkpoint ./pretrained/uniform50.pth --clipping_value 1.4 --temperature 1.7 \
    --save_path ./results/main/uniform/uniform50.csv

python driver/eval.py --n 100 --embed_dim 256 --num_gcn_layers 10 \
    --checkpoint ./pretrained/uniform100.pth --clipping_value 6.2 --temperature 1.0 \
    --save_path ./results/main/uniform/uniform100.csv
