import os
import sys

curr_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(curr_dir, os.pardir))
sys.path.append(project_root)
import yaml
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.utils.data_utils import load_qaplib
from src.finetune import mcmc_finetune


def load_instance_configs(config_path):
    """Load instance-specific configurations from a YAML file."""
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    defaults = raw.get("defaults", {})
    instances = raw.get("instances", {})
    return defaults, instances


def get_config(n, instance_name=None, default_clipping_value=11.0,
               yaml_defaults=None, yaml_instances=None):
    """
    Build fine-tuning config for a QAPLIB instance.

    Applies instance-specific overrides from the YAML config when available.
    Factor fields (local_search_iter_factor, num_actions_factor) are multiplied
    by n to obtain the final parameter value.
    """
    config = {
        "num_finetune_steps": 200,
        "learning_rate": 1e-4,
        "num_starts": yaml_defaults.get("num_starts", 20) if yaml_defaults else 20,
        "num_chains": yaml_defaults.get("num_chains", 20) if yaml_defaults else 20,
        "chain_length": np.clip(n // 3, 10, 50),
        "local_search_iter": np.clip(n, 20, 200),
        "num_actions": np.clip(2 * n, 50, 200),
        "entropy_weight": 1 / n / np.log(n),
        "initial_samples": None,
    }

    clipping_value = default_clipping_value

    # Apply instance-specific overrides from YAML
    if yaml_instances and instance_name in yaml_instances:
        inst_cfg = yaml_instances[instance_name]
        if "clipping_value" in inst_cfg:
            clipping_value = inst_cfg["clipping_value"]
        if "num_starts" in inst_cfg:
            config["num_starts"] = inst_cfg["num_starts"]
        if "num_chains" in inst_cfg:
            config["num_chains"] = inst_cfg["num_chains"]
        if "local_search_iter_factor" in inst_cfg:
            config["local_search_iter"] = int(inst_cfg["local_search_iter_factor"] * n)
        if "num_actions_factor" in inst_cfg:
            config["num_actions"] = int(inst_cfg["num_actions_factor"] * n)

    return config, clipping_value


def driver(output_dir, repetitions=10, model_params=None, checkpoint=None,
           verbose=False, instance_config_path=None):
    os.makedirs(output_dir, exist_ok=True)

    output_path_all = os.path.join(output_dir, 'all_results.csv')
    output_path_stats = os.path.join(output_dir, 'stats_results.csv')

    # Load YAML instance configs
    yaml_defaults, yaml_instances = None, None
    if instance_config_path and os.path.exists(instance_config_path):
        yaml_defaults, yaml_instances = load_instance_configs(instance_config_path)

    data_loader = load_qaplib(directory="./data/qaplib")
    default_clipping_value = model_params.get("clipping_value", 11.0)

    for problem_name, (n, opt, A, B) in data_loader:
        instance_name = problem_name.replace(".dat", "")

        config, clipping_value = get_config(
            n, instance_name, default_clipping_value,
            yaml_defaults, yaml_instances,
        )

        current_model_params = model_params.copy()
        current_model_params["clipping_value"] = clipping_value

        print(f"Processing Instance: {instance_name} (n={n}, opt={opt}, C={clipping_value})")
        if verbose:
            print(
                f"  Config: K={config['num_starts']}, M={config['num_chains']}, "
                f"num_ls={config['local_search_iter']}, K_LS={config['num_actions']}"
            )

        A, B = A.unsqueeze(0), B.unsqueeze(0)

        results = []
        for i in range(repetitions):
            distance_scaler, flow_scaler = A.max().item(), B.max().item()
            _, cost, _, run_time = mcmc_finetune(
                A / distance_scaler, B / flow_scaler, config,
                opt / (distance_scaler * flow_scaler),
                model_params=current_model_params,
                checkpoint=checkpoint, verbose=verbose,
            )
            cost = cost * distance_scaler * flow_scaler

            cost_value = cost.item() if hasattr(cost, 'item') else cost
            gap = (cost_value - opt) / opt * 100 if opt != 0 else float('inf')
            print(f"cost: {cost_value:.0f} | gap: {gap:.3f}% | runtime: {run_time:.2f}s")

            results.append({
                "problem_name": instance_name, "n": n, "repetition": i + 1,
                "cost": cost_value, "time": run_time, "gap": gap,
                "clipping_value": clipping_value,
                "num_starts": config["num_starts"],
                "num_chains": config["num_chains"],
            })

        df = pd.DataFrame(results)
        if repetitions == 1:
            df = df.drop(columns=["repetition"])
        df.to_csv(output_path_all, mode='a', header=not os.path.exists(output_path_all), index=False, float_format='%.4f')

        if repetitions > 1:
            stats = pd.DataFrame([{
                "problem_name": instance_name, "n": n, "repetition": repetitions,
                "cost_min": df['cost'].min(), "cost_mean": df['cost'].mean(), "cost_max": df['cost'].max(),
                "gap_min": df['gap'].min(), "gap_mean": df['gap'].mean(), "gap_max": df['gap'].max(),
                "time_mean": df['time'].mean(),
            }])
            stats.to_csv(output_path_stats, mode='a', header=not os.path.exists(output_path_stats), index=False, float_format='%.4f')
            print(f"Finished. Gap(avg): {stats['gap_mean'].iloc[0]:.2f}%, Time(avg): {stats['time_mean'].iloc[0]:.2f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate PLMA on QAPLIB benchmark instances")
    parser.add_argument('--output_dir', type=str, default='./results/qaplib', help='Directory to save results')
    parser.add_argument('--repetitions', '-r', type=int, default=10, help='Number of repetitions per instance')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--instance_config', type=str,
                        default='./configs/qaplib_instance_configs.yaml',
                        help='Path to YAML file with instance-specific parameter overrides')

    parser.add_argument("--init_dim", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_gcn_layers", type=int, default=10)
    parser.add_argument("--num_att_layers", type=int, default=1)
    parser.add_argument("--clipping_value", type=float, default=11.0, help='Default clipping value')
    parser.add_argument("--num_iterations", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--checkpoint", type=str, default=None, help='Path to model checkpoint')

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
    driver(
        args.output_dir, args.repetitions, model_params,
        checkpoint=args.checkpoint, verbose=args.verbose,
        instance_config_path=args.instance_config,
    )
