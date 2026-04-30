#!/usr/bin/env bash
# Evaluate PLMA on geometrically structured (SAWT) synthetic QAP datasets.
# Usage: Run from the project root directory after activating the conda environment.

python driver/eval.py --dataset sawt --n 20 --embed_dim 128 --num_gcn_layers 1 \
    --checkpoint ./pretrained/sawt20.pth --clipping_value 0.5 \
    --save_path ./results/main/sawt/sawt20.csv

python driver/eval.py --dataset sawt --n 50 --embed_dim 128 --num_gcn_layers 8 \
    --checkpoint ./pretrained/sawt50.pth --clipping_value 0.5 \
    --save_path ./results/main/sawt/sawt50.csv

python driver/eval.py --dataset sawt --n 100 --embed_dim 256 --num_gcn_layers 10 \
    --checkpoint ./pretrained/sawt100.pth --clipping_value 0.5 \
    --save_path ./results/main/sawt/sawt100.csv
