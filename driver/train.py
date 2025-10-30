import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)
import pytz
import yaml
import argparse
import pprint as pp
from datetime import datetime
import torch
from src.trainer import Trainer
from src.utils.utils import seed_everything


def args2dict(args):
    dataset_configs = {
        "uniform": {"D_type": "random", "F_type": "random"},
        "sawt": {"D_type": "2d_euc", "F_type": "erdos", "p_erdos": 0.7},
    }
    generator_params = {"problem_size": args.problem_size, **dataset_configs.get(args.dataset_type, {})}

    model_params = {
        "init_dim": args.init_dim,
        "embed_dim": args.embed_dim,
        "num_heads": args.num_heads,
        "num_gcn_layers": args.num_gcn_layers,
        "num_att_layers": args.num_att_layers,
        "clipping_value": args.clipping_value,
        "num_iterations": args.num_iterations,
        "temperature": args.temperature,
    }

    optimizer_params = {
        "optimizer": {"lr": args.lr},
        "scheduler": {"milestones": args.milestones, "gamma": args.gamma},
    }

    trainer_params = {
        "epochs": args.epochs,
        "train_episodes": args.train_episodes,
        "batch_size": args.batch_size,
        "model_save_interval": args.model_save_interval,
    }

    plma_configs = {
        "num_samples": args.num_samples,
        "chain_length": args.chain_length,
        "local_search_iter": args.local_search_iter,
        "num_actions": args.num_actions,
        "entropy_weight": args.entropy_weight,
    }
    return generator_params, model_params, trainer_params, optimizer_params, plma_configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # generator params
    parser.add_argument("--dataset_type", type=str, default="uniform", choices=["uniform", "sawt"])
    parser.add_argument("--problem_size", type=int, default=50)

    # model params
    parser.add_argument("--init_dim", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_gcn_layers", type=int, default=10)
    parser.add_argument("--num_att_layers", type=int, default=1)
    parser.add_argument("--num_iterations", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--clipping_value", type=float, default=11.0)
    parser.add_argument("--autoregressive", action="store_true")

    # optimizer params
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--milestones", nargs="+", type=int, default=[9001])
    parser.add_argument("--gamma", type=float, default=0.1)

    # trainer params
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--train_episodes", type=int, default=100_000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model_save_interval", type=int, default=50)

    # plma training configs
    parser.add_argument("--num_samples", type=int, default=400)
    parser.add_argument("--chain_length", type=int, default=None)
    parser.add_argument("--local_search_iter", type=int, default=1)
    parser.add_argument("--num_actions", type=int, default=None)
    parser.add_argument("--entropy_weight", type=float, default=0)

    # misc configs
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp_name", type=str, default="uniform50")
    parser.add_argument("--config", default=None, type=str, help="Path to a config file to override default args.")
    args = parser.parse_args()
    
    if args.config is not None:
        with open("configs/" + args.config, "r") as f:
            config_args = yaml.safe_load(f)
        for k, v in config_args.items():
            assert hasattr(args, k), f'Unknown config key: {k}'
            setattr(args, k, v)
    args.num_actions = args.problem_size if args.num_actions is None else args.num_actions
    args.chain_length = args.problem_size if args.chain_length is None else args.chain_length


    # Set log path
    process_start_time = datetime.now(pytz.timezone("Asia/Shanghai"))
    log_dir = os.path.join(project_dir, "logs")
    args.log_path = os.path.join(log_dir, args.exp_name, process_start_time.strftime("%Y-%m-%d_%H_%M"))
    os.makedirs(args.log_path, exist_ok=True)
    print(">> Log path: ", args.log_path)

    # Set device
    assert torch.cuda.is_available(), "CUDA is not available. Please run on a machine with a GPU."
    args.device = "cuda"
    torch.set_default_device(args.device)

    # Set random seed
    seed_everything(args.seed)

    generator_params, model_params, trainer_params, optimizer_params, plma_configs = args2dict(args)
    # Save structured arguments to a file for easier inspection
    with open(os.path.join(args.log_path, "args.txt"), "w") as f:
        pp.pprint(
            {
                "generator_params": generator_params,
                "model_params": model_params,
                "trainer_params": trainer_params,
                "optimizer_params": optimizer_params,
                "plma_configs": plma_configs,
            },
            stream=f,
        )

    trainer = Trainer(args, generator_params, model_params, trainer_params, optimizer_params, plma_configs)
    print(">> Start training...")
    trainer.run()
    print(">> Training finished!")
