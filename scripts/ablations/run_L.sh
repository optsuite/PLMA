#!/usr/bin/env bash
# Ablation study: varying chain length L.
# Usage: Run from the project root directory after activating the conda environment.

n=100

L_values=(10 33 50 100)
seeds=(0 1 2 3 4 5 6 7 8 9)
echo "Running experiments with n=$n"
echo "L values: ${L_values[@]}"

for L in "${L_values[@]}"; do
    echo "Running experiment with L=$L"
    for i in "${seeds[@]}"; do
        python driver/eval.py --n $n --embed_dim 256 --num_gcn_layers 10 \
            --clipping_value 6.2 --temperature 1.0 \
            --checkpoint ./pretrained/uniform${n}.pth \
            --seed ${seeds[$i]} -L $L \
            --save_path ./results/misc/parameters/chain_lengths/L${L}/run${i}.csv
    done
done
