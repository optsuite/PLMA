import os
import sys

curr_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(curr_dir, os.pardir))
sys.path.append(project_root)
import pandas as pd
import numpy as np
import torch
import warnings
from src.finetune import mcmc_finetune, autoregressive_finetune
from src.utils.utils import seed_everything


def load_synthetic(F_root, D_root, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Load synthetic QAP data from .npy files.
    """
    F_all = np.load(F_root).astype(np.float32)
    D_all = np.load(D_root).astype(np.float32)
    F = torch.tensor(F_all).to(device)
    D = torch.tensor(D_all).to(device)
    return F, D


def config_batch(args):
    """
    Build a configuration dict for MCMC fine-tuning on synthetic datasets.
    """
    n = args.n
    config = {
        "num_finetune_steps": args.num_finetune_steps,
        "learning_rate": args.lr,
        "num_starts": args.num_starts,
        "num_chains": args.num_chains,
        "chain_length": args.chain_length if args.chain_length is not None else n // 3,
        "local_search_iter": args.local_search_iter if args.local_search_iter is not None else n,
        "num_actions": args.num_actions if args.num_actions is not None else 2 * n,
        "entropy_weight": 1 / n / np.log(n) if args.entropy_regu else 0,
        "initial_samples": None,
    }
    return config


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate PLMA on synthetic QAP datasets")
    parser.add_argument('--dataset', type=str, default='uniform', choices=['uniform', 'sawt'], help='Dataset to use')
    parser.add_argument('--n', type=int, default=50, help='Size of the QAP instance')

    # Fine-tuning parameters
    parser.add_argument('--num_finetune_steps', '-T', type=int, default=200, help='Number of fine-tuning steps')
    parser.add_argument('--num_starts', '-K', type=int, default=20, help='Number of independent starts')
    parser.add_argument('--num_chains', '-M', type=int, default=20, help='Number of MCMC chains per start')
    parser.add_argument('--chain_length', '-L', type=int, default=None, help='MCMC chain length; default n//3')
    parser.add_argument('--local_search_iter', type=int, default=None, help='Local search iterations; default n')
    parser.add_argument('--num_actions', type=int, default=None, help='Number of local-search swap actions; default 2*n')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--entropy_regu', action='store_true', help='Whether to use entropy regularization')

    # Model parameters
    parser.add_argument("--init_dim", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_gcn_layers", type=int, default=8)
    parser.add_argument("--num_att_layers", type=int, default=1)
    parser.add_argument("--clipping_value", "-C", type=float, default=1.0)
    parser.add_argument("--num_iterations", type=int, default=1)
    parser.add_argument("--temperature", "-t", type=float, default=1.0)

    # Misc
    parser.add_argument('--ar', action='store_true', help='Whether to use autoregressive finetuning')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument("--checkpoint", type=str, default=None, help='Path to the model checkpoint')
    parser.add_argument("--save_path", type=str, default=None, help='Path to save the results')
    parser.add_argument('--verbose', type=bool, default=True, help='Whether to print progress information')
    
    args = parser.parse_args()
    n = args.n

    # Load data
    D_root = f"./data/{args.dataset}/{args.dataset}_{n}_D.npy"
    F_root = f"./data/{args.dataset}/{args.dataset}_{n}_F.npy"
    F, D = load_synthetic(F_root, D_root)

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

    config = config_batch(args)
    
    rots_costs = pd.read_csv(f'./data/rots_results/{args.dataset}/{args.dataset}{n}_rots(5k)_all_results.csv', usecols=['cost']).to_numpy().flatten()
    rots_costs = torch.tensor(rots_costs, device="cuda")

    seed_everything(args.seed)
    if args.ar:
        config["num_samples"] = config["num_starts"] * config["num_chains"]
        solutions, costs, gaps, runtime = autoregressive_finetune(D, F, config, rots_costs, model_params=model_params, checkpoint=args.checkpoint, verbose=args.verbose, save_path=args.save_path)
    else:
        solutions, costs, gaps, runtime = mcmc_finetune(D, F, config, rots_costs, early_stop=False, model_params=model_params, checkpoint=args.checkpoint, verbose=args.verbose, save_path=args.save_path)
    print(f"Cost: {costs.mean():.2f} | Gap: {gaps.mean():.2f} | Time: {runtime:.2f}s")
