import os
import time
import torch
import logging
import pandas as pd
from src.backend.qap_backend import QAPBackendGPU
from src.models.model import QAPNet
from src.backend.sampling import run_mcmc_and_improve, ar_decode_and_improve, sequential_sampling

def setup_logger_and_log_header():
    """Setup console logger and print header."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        force=True
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 90)
    logger.info(f"{'Iter':^6} | {'BestCost':^12} {'Gap':^10} {'Incumbent':^12} | "
               f"{'Loss':^8} {'Entropy':^8} {'Ratio':^8}")
    logger.info("=" * 90)
    
    return logger

def mcmc_finetune(
    D_batch,
    F_batch,
    config,
    opt=None,
    early_stop=True,
    net=None,
    model_params=None,
    checkpoint=None,
    verbose=True,
    save_path=None,
):
    """
    Args:
        opt: (num_instances,)
        D_batch: (num_instances, n, n)
        F_batch: (num_instances, n, n)
    Returns:
        
    """
    num_instances, problem_size, _ = D_batch.size()
    if torch.cuda.is_available():
        device = torch.device("cuda")
        backend = QAPBackendGPU(device=device)
        total_random_threads = num_instances * config["num_starts"] * config["num_chains"] * config.get("num_actions")
        backend.setup_rand_states(total_random_threads)
    else:
        raise RuntimeError("CUDA is not available. Please run on a machine with a GPU.")
    if net is None:
        net = QAPNet(**model_params).to(device)
    if checkpoint is not None:
        net.load_state_dict(torch.load(checkpoint))

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=config["learning_rate"])

    global_best_solutions = torch.zeros(num_instances, problem_size, dtype=torch.long, device=device)
    global_best_costs = torch.full((num_instances,), float("inf"), device=device, dtype=torch.float64)
    global_best_costs_list, global_best_gap_list, iter_best_costs_list = [], [], []
    logger = setup_logger_and_log_header() if verbose else None
    start_time = time.time()

    # Get initial states
    if config.get("initial_samples") is None:
        heatmap_init = net(D_batch, F_batch).detach()
        states, _, _ = sequential_sampling(heatmap_init, config["num_starts"], D_batch, F_batch)
    else:
        states = config["initial_samples"].to(device)

    for iter in range(config["num_finetune_steps"]):
        heatmap = net(D_batch, F_batch)
        states, iter_best_sols, iter_best_costs, loss, entropy, ratio = run_mcmc_and_improve(D_batch, F_batch, states, heatmap, backend, **config)

        optimizer.zero_grad()    
        loss.backward()
        optimizer.step()

        improvement_mask = iter_best_costs < global_best_costs
        global_best_solutions[improvement_mask] = iter_best_sols[improvement_mask]
        global_best_costs[improvement_mask] = iter_best_costs[improvement_mask]

        gap = (global_best_costs - opt) / opt * 100 if opt is not None else torch.tensor(float('inf'), device=device)

        if verbose:
            best_gap_str = f"{gap.mean().item():.2f}%" if gap is not None else ""
            logger.info(
                f"{iter:^6} | "
                f"{global_best_costs.mean().item():^12.2f} "
                f"{best_gap_str:^10} "
                f"{iter_best_costs.mean().item():^12.2f} | "
                f"{loss.item():^8.2f} "
                f"{entropy.item():^8.2f} "
                f"{ratio:^8.2f}"
            )

        if early_stop and (gap < 0.0005).all():
            print("All instances in the batch have reached or exceeded the known optimal solution. Terminating early.")
            break

        global_best_costs_list.append(global_best_costs.mean().item())
        iter_best_costs_list.append(iter_best_costs.mean().item())
        global_best_gap_list.append(gap.mean().item())

    run_time = time.time() - start_time

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df = pd.DataFrame({
            'iter': range(len(global_best_costs_list)),
            'global_best_cost': global_best_costs_list,
            'iter_best_cost': iter_best_costs_list,
            'gap': global_best_gap_list
        })
        df.to_csv(save_path, index=False)

    return global_best_solutions, global_best_costs, gap, run_time


def autoregressive_finetune(
    D_batch,
    F_batch,
    config,
    opt=None,
    early_stop=True,
    net=None,
    model_params=None,
    checkpoint=None,
    verbose=True,
    save_path=None,
):
    """
    Args:
        opt: (num_instances,)
        D_batch: (num_instances, n, n)
        F_batch: (num_instances, n, n)
    Returns:

    """
    num_instances, problem_size, _ = D_batch.size()
    if torch.cuda.is_available():
        device = torch.device("cuda")
        backend = QAPBackendGPU(device=device)
        total_random_threads = num_instances * config["num_samples"] * config.get("num_actions")
        backend.setup_rand_states(total_random_threads)
    else:
        raise RuntimeError("CUDA is not available. Please run on a machine with a GPU.")
    if net is None:
        net = QAPNet(**model_params).to(device)
    if checkpoint is not None:
        net.load_state_dict(torch.load(checkpoint))

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=config["learning_rate"])

    global_best_solutions = torch.zeros(num_instances, problem_size, dtype=torch.long, device=device)
    global_best_costs = torch.full((num_instances,), float("inf"), device=device, dtype=torch.float64)
    global_best_costs_list, global_best_gap_list, iter_best_costs_list = [], [], []
    logger = setup_logger_and_log_header() if verbose else None
    start_time = time.time()

    for iter in range(config["num_finetune_steps"]):
        heatmap = net(D_batch, F_batch)
        iter_best_sols, iter_best_costs, loss, entropy, ratio = ar_decode_and_improve(D_batch, F_batch, heatmap, backend, **config)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        improvement_mask = iter_best_costs < global_best_costs
        global_best_solutions[improvement_mask] = iter_best_sols[improvement_mask]
        global_best_costs[improvement_mask] = iter_best_costs[improvement_mask]

        gap = (global_best_costs - opt) / opt * 100 if opt is not None else torch.tensor(float('inf'), device=device)

        if verbose:
            best_gap_str = f"{gap.mean().item():.2f}%" if gap is not None else ""
            logger.info(
                f"{iter:^6} | "
                f"{global_best_costs.mean().item():^12.2f} "
                f"{best_gap_str:^10} "
                f"{iter_best_costs.mean().item():^12.2f} | "
                f"{loss.item():^8.2f} "
                f"{entropy.item():^8.2f} "
                f"{ratio:^8.2f}"
            )

        if early_stop and (gap < 0.0005).all():
            print("All instances in the batch have reached or exceeded the known optimal solution. Terminating early.")
            break

        global_best_costs_list.append(global_best_costs.mean().item())
        iter_best_costs_list.append(iter_best_costs.mean().item())
        global_best_gap_list.append(gap.mean().item())

    run_time = time.time() - start_time

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df = pd.DataFrame(
            {
                'iter': range(len(global_best_costs_list)),
                'global_best_cost': global_best_costs_list,
                'iter_best_cost': iter_best_costs_list,
                'gap': global_best_gap_list,
            }
        )
        df.to_csv(save_path, index=False)

    return global_best_solutions, global_best_costs, gap, run_time
