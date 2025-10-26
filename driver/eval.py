import os
import sys

curr_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(curr_dir, os.pardir))
sys.path.append(project_root)
import pandas as pd
import numpy as np
import torch
import warnings
from src.finetune import mcmc_finetune
from src.utils.utils import seed_everything


def load_synthetic(F_root, D_root, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    一个生成器函数，用于加载'synthetic'数据集。
    """
    F_all = np.load(F_root).astype(np.float32)
    D_all = np.load(D_root).astype(np.float32)
    F = torch.tensor(F_all).to(device)
    D = torch.tensor(D_all).to(device)
    return F, D


def config_batch(n):
    config = {
        "num_finetune_steps": 200,
        "learning_rate": 1e-4,
        "num_starts": 20,
        "num_chains": 20,
        "chain_length": np.clip(n // 3, 10, 50),
        "local_search_iter": np.clip(n, 20, 200),
        "num_actions": np.clip(n*2, 20, 200),
        "entropy_weight": None,
        "initial_samples": None,
    }

    return config


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import argparse

    parser = argparse.ArgumentParser(description="Run MCPG on synthetic data")
    parser.add_argument('--n', type=int, default=50, help='Size of the QAP instance')
    parser.add_argument('--lr', type=float, default=1e-4)
   
    parser.add_argument('--entropy_regu', action='store_true', help='Whether to use entropy regularization')
    parser.add_argument('--num_finetune_steps', type=int, default=200)
    parser.add_argument('--chain_length', '-L', type=int, default=None, help='MCMC steps, default to n/3')
    parser.add_argument('--num_starts', '-K', type=int, default=None)
    parser.add_argument('--num_chains', '-M', type=int, default=None)
    
    parser.add_argument("--init_dim", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_gcn_layers", type=int, default=8)
    parser.add_argument("--num_att_layers", type=int, default=1)
    parser.add_argument("--clipping_value", "-C", type=float, default=1.0)
    parser.add_argument("--num_iterations", type=int, default=1)
    parser.add_argument("--temperature", "-t", type=float, default=1.0)

    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument("--checkpoint", type=str, default=None, help='Path to the model checkpoint')
    parser.add_argument("--save_path", type=str, default=None, help='Path to save the results')
    parser.add_argument('--verbose', type=bool, default=True, help='Whether to print progress information')
    parser.add_argument('--dataset', type=str, default='uniform', help='Dataset: synthetic or uniform')

    args = parser.parse_args()
    n = args.n

    if args.dataset == 'synthetic':
        D_root = f"../data/synthetic/{n}_D_test.npy"
        F_root = f"../data/synthetic/{n}_F_test.npy"
    elif args.dataset == 'uniform':
        D_root = f"../data/uniform/uniform_{n}_D.npy"
        F_root = f"../data/uniform/uniform_{n}_F.npy"
    elif args.dataset == 'sawt':
        D_root = f"../data/sawt/sawt_{n}_D.npy"
        F_root = f"../data/sawt/sawt_{n}_F.npy"

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

    config = config_batch(n)
    config['learning_rate'] = args.lr


    if args.entropy_regu:
        config["entropy_weight"] = 1 / n / np.log(n)
    else:
        config["entropy_weight"] = 0

    config["chain_length"] = args.chain_length if args.chain_length is not None else config["chain_length"]
    config["num_starts"] = args.num_starts if args.num_starts is not None else config["num_starts"]
    config["num_chains"] = args.num_chains if args.num_chains is not None else config["num_chains"]
    
    rots_costs = pd.read_csv(f'../results/{args.dataset}/{args.dataset}{n}_rots(5k)_all_results.csv', usecols=['cost']).to_numpy().flatten()
    rots_costs = torch.tensor(rots_costs, device="cuda")

    seed_everything(args.seed)

    solutions, costs, gaps, runtime = mcmc_finetune(D, F, config, rots_costs, model_params=model_params, checkpoint=args.checkpoint, verbose=args.verbose, save_path=args.save_path)
    print(f"Cost: {costs.mean():.2f} | Gap: {gaps.mean():.2f} | Time: {runtime:.2f}s")
