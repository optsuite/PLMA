#!/usr/bin/env bash
conda activate plma

echo "Experiment started at $(date)"
echo "Running experiments on large scale datasets"


python driver/eval.py --dataset sawt --n 200 --embed_dim 256 --num_gcn_layers 10 --checkpoint ./pretrained/sawt100.pth --clipping_value 0.5  --save_path ./results/sawt/sawt200.csv

# python driver/eval.py --dataset sawt --n 20 --embed_dim 128 --num_gcn_layers 1 --checkpoint ./pretrained/sawt20.pth --clipping_value 0.5
# python driver/eval.py --dataset sawt --n 50 --embed_dim 128 --num_gcn_layers 8 --checkpoint ./pretrained/sawt50.pth --clipping_value 0.5
# python driver/eval.py --dataset sawt --n 100 --embed_dim 256 --num_gcn_layers 10 --checkpoint ./pretrained/sawt100.pth --clipping_value 0.5 
# python driver/eval.py --dataset sawt --n 500 --embed_dim 256 --num_gcn_layers 10 --checkpoint ./pretrained/sawt100.pth --clipping_value 0.5  

