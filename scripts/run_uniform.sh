#!/usr/bin/env bash
conda activate plma

python driver/eval.py --n 50 --embed_dim 128 --num_gcn_layers 8 --checkpoint ./pretrained/uniform50.pth --clipping_value 1.4 --temperature 1.7