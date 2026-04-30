#!/usr/bin/env bash
# Ablation study: varying (K, M) sampling strategies with fixed total samples = 400.
# Usage: Run from the project root directory after activating the conda environment.

n=100

declare -a sample_chain_pairs=(
    "1 400"
    "10 40" 
    "20 20"
    "40 10"
    "400 1"
)
seeds=(0 1 2 3 4 5 6 7 8 9)
echo "Running experiments with n=$n"
echo "Testing different (num_starts, num_chains) combinations with total samples = 400"

for pair in "${sample_chain_pairs[@]}"; do
    read -r num_starts num_chains <<< "$pair"
    echo "Running experiment with num_starts=$num_starts, num_chains=$num_chains"
    for i in "${seeds[@]}"; do
        python driver/eval.py --n $n --embed_dim 256 --num_gcn_layers 10 \
            --clipping_value 6.2 --temperature 1.0 \
            --checkpoint ./pretrained/uniform${n}.pth \
            --seed ${seeds[$i]} -K $num_starts -M $num_chains \
            --save_path ./results/misc/parameters/sampling_strategy/K${num_starts}-M${num_chains}/run${i}.csv
    done 
done

echo "All experiments completed!"
