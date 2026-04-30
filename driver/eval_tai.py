import os
import sys

curr_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(curr_dir, os.pardir))
sys.path.append(project_root)
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.utils.data_utils import load_tai
from src.finetune import mcmc_finetune, autoregressive_finetune


def get_config(n, lr=1e-4):
    """Build fine-tuning config for a Taixxeyy instance."""
    config = {
        "num_finetune_steps": 200,
        "learning_rate": lr,
        "num_starts": 20,
        "num_chains": 20,
        "chain_length": np.clip(n // 3, 10, 50),
        "local_search_iter": np.clip(n, 20, 200),
        "num_actions": np.clip(2 * n, 50, 200),
        "entropy_weight": 1 / n / np.log(n),
        "initial_samples": None,
    }
    return config


def driver(output_dir, repetitions=10, model_params=None, checkpoint=None,
           verbose=False, ar=False, lr=1e-4):
    os.makedirs(output_dir, exist_ok=True)

    output_path_all = os.path.join(output_dir, 'all_results.csv')
    output_path_stats = os.path.join(output_dir, 'stats_results.csv')

    data_loader = load_tai(data_root="./data/tai")

    for problem_name, (n, opt, A, B) in data_loader:
        print(f"Processing Instance: {problem_name} (n={n}, opt={opt})")
        A, B = A.unsqueeze(0), B.unsqueeze(0)
        
        results = []
        for i in range(repetitions):
            config = get_config(n, lr=lr)
            
            distance_scaler, flow_scaler = A.max().item(), B.max().item()

            if ar:
                config["num_samples"] = config["num_starts"] * config["num_chains"]
                _, cost, _, run_time = autoregressive_finetune(
                    A / distance_scaler, B / flow_scaler, config,
                    opt / (distance_scaler * flow_scaler),
                    model_params=model_params,
                    checkpoint=checkpoint, verbose=verbose,
                )
            else:
                _, cost, _, run_time = mcmc_finetune(
                    A / distance_scaler, B / flow_scaler, config,
                    opt / (distance_scaler * flow_scaler),
                    model_params=model_params,
                    checkpoint=checkpoint, verbose=verbose,
                )
            cost = cost * distance_scaler * flow_scaler
                
            cost_value = cost.item() if hasattr(cost, 'item') else cost
            gap = (cost_value - opt) / opt * 100 if opt != 0 else float('inf')
            print(f"cost: {cost_value:.0f} | gap: {gap:.3f}% | runtime: {run_time:.2f}s")

            results.append({
                "problem_name": problem_name, "n": n, "repetition": i + 1,
                "cost": cost_value, "time": run_time, "gap": gap,
            })

        df = pd.DataFrame(results)
        if repetitions == 1:
            df = df.drop(columns=["repetition"])
        df.to_csv(output_path_all, mode='a', header=not os.path.exists(output_path_all), index=False, float_format='%.4f')

        if repetitions > 1:
            stats = pd.DataFrame([{
                "problem_name": problem_name, "n": n, "repetition": repetitions,
                "cost_min": df['cost'].min(), "cost_mean": df['cost'].mean(), "cost_max": df['cost'].max(),
                "gap_min": df['gap'].min(), "gap_mean": df['gap'].mean(), "gap_max": df['gap'].max(),
                "time_mean": df['time'].mean(),
            }])
            stats.to_csv(output_path_stats, mode='a', header=not os.path.exists(output_path_stats), index=False, float_format='%.4f')
            print(f"Finished. Gap(avg): {stats['gap_mean'].iloc[0]:.2f}%, Time(avg): {stats['time_mean'].iloc[0]:.2f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate PLMA on Taixxeyy benchmark instances")
    parser.add_argument('--output_dir', type=str, default='./results/tai', help='Directory to save results')
    parser.add_argument('--repetitions', '-r', type=int, default=10, help='Number of repetitions per instance')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

    parser.add_argument("--init_dim", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_gcn_layers", type=int, default=10)
    parser.add_argument("--num_att_layers", type=int, default=1)
    parser.add_argument("--clipping_value", type=float, default=0.5)
    parser.add_argument("--num_iterations", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--checkpoint", type=str, default=None, help='Path to model checkpoint')

    parser.add_argument('--ar', action='store_true', help='Use autoregressive finetuning')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (set to 0 for gradient-free)')

    args = parser.parse_args()
    
    model_params = {
        "init_dim": args.init_dim,
        "embed_dim": args.embed_dim,
        "num_heads": args.num_heads,
        "num_gcn_layers": args.num_gcn_layers,
        "num_att_layers": args.num_att_layers,
        "num_iterations": args.num_iterations,
        "temperature": args.temperature,
        "clipping_value": args.clipping_value,
    }
    driver(args.output_dir, args.repetitions, model_params,
           checkpoint=args.checkpoint, verbose=args.verbose,
           ar=args.ar, lr=args.lr)
