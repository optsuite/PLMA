import os
import time
import torch
import logging
import pandas as pd
from src.backend.qap_backend import QAPBackendGPU
from src.models.simple_model import SimpleNet
from src.backend.sampling import run_mcmc_and_improve, sequential_sampling
from src.utils.ops import log_sinkhorn

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


def apply_heatmap_offset(heatmap, heatmap_offset=None, renorm=False, sinkhorn_iterations=1):
    if heatmap_offset is None:
        return heatmap

    adjusted_heatmap = heatmap + heatmap_offset
    if renorm:
        adjusted_heatmap = log_sinkhorn(adjusted_heatmap, sinkhorn_iterations, 1.0)
    return adjusted_heatmap


def get_sinkhorn_params(net):
    if hasattr(net, "iterations") and hasattr(net, "temperature"):
        return net.iterations, net.temperature
    raise AttributeError("Model does not expose Sinkhorn parameters.")


def compute_adjusted_heatmap(
    net,
    D_batch,
    F_batch,
    heatmap_offset=None,
    heatmap_offset_renorm=False,
    heatmap_offset_sinkhorn_iterations=1,
    logits_offset=None,
):
    if logits_offset is not None:
        logits = net.forward_logits(D_batch, F_batch)
        adjusted_logits = logits + logits_offset
        sinkhorn_iterations, temperature = get_sinkhorn_params(net)
        heatmap = log_sinkhorn(adjusted_logits, sinkhorn_iterations, temperature)
        return heatmap, adjusted_logits

    heatmap = net(D_batch, F_batch)
    heatmap = apply_heatmap_offset(
        heatmap,
        heatmap_offset=heatmap_offset,
        renorm=heatmap_offset_renorm,
        sinkhorn_iterations=heatmap_offset_sinkhorn_iterations,
    )
    return heatmap, None


def resolve_backend_seed(config):
    explicit_seed = config.get("random_seed")
    if explicit_seed is not None:
        return int(explicit_seed)

    base_seed = config.get("random_seed_base")
    if base_seed is None:
        return None

    round_idx = int(config.get("random_seed_round", 0))
    seed_stride = int(config.get("random_seed_stride", 1))
    return int(base_seed) + round_idx * seed_stride


def should_extend_finetune(global_best_costs_history, completed_steps, base_steps, config, has_extended):
    if has_extended or not config.get("adaptive_step_extension", True):
        return False

    window = int(config.get("adaptive_extension_window", 25))
    extra_steps = int(config.get("adaptive_extension_extra_steps", 50))
    threshold = float(config.get("adaptive_extension_best_cost_threshold", 40.0))

    if window <= 1 or extra_steps <= 0:
        return False
    if completed_steps != base_steps:
        return False
    if len(global_best_costs_history) < window:
        return False

    recent = global_best_costs_history[-window:]
    return any(recent[i] < recent[i - 1] and recent[i] < threshold for i in range(1, len(recent)))

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
    return_aux=False,
    heatmap_offset=None,
    heatmap_offset_renorm=False,
    heatmap_offset_sinkhorn_iterations=1,
    logits_offset=None,
):
    """
    Args:
        opt: (num_instances,)
        D_batch: (num_instances, n, n)
        F_batch: (num_instances, n, n)
    Returns:
        
    """
    if heatmap_offset is not None and logits_offset is not None:
        raise ValueError("heatmap_offset and logits_offset cannot be enabled at the same time.")

    num_instances, problem_size, _ = D_batch.size()
    if torch.cuda.is_available():
        device = torch.device("cuda")
        backend = QAPBackendGPU(device=device)
        total_random_threads = num_instances * config["num_starts"] * config["num_chains"] * config.get("num_actions")
        backend_seed = resolve_backend_seed(config)
        backend.setup_rand_states(total_random_threads, seed=backend_seed)
    else:
        raise RuntimeError("CUDA is not available. Please run on a machine with a GPU.")
    if net is None:
        model_type = model_params.get("model_type", "simplenet").lower()
        if model_type == "simplenet":
            initial_W = model_params.get("initial_W")
            if initial_W is not None:
                initial_W = initial_W.to(device)
            net = SimpleNet(
                n=problem_size,
                iterations=model_params.get("num_iterations", 1),
                temperature=model_params.get("temperature", 1.0),
                initial_W=initial_W,
            ).to(device)
        else:
            raise ValueError("This reproduction package only includes model_type='simplenet'.")
    if checkpoint is not None:
        net.load_state_dict(torch.load(checkpoint))


    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=config["learning_rate"])

    global_best_solutions = torch.zeros(num_instances, problem_size, dtype=torch.long, device=device)
    global_best_costs = torch.full((num_instances,), float("inf"), device=device, dtype=torch.float64)
    global_best_costs_list, global_best_gap_list, iter_best_costs_list = [], [], []
    logger = setup_logger_and_log_header() if verbose else None
    start_time = time.time()
    base_steps = int(config["num_finetune_steps"])
    max_steps = base_steps
    has_extended = False

    # Get initial states
    if config.get("initial_samples") is None:
        heatmap_init, _ = compute_adjusted_heatmap(
            net,
            D_batch,
            F_batch,
            heatmap_offset=heatmap_offset,
            heatmap_offset_renorm=heatmap_offset_renorm,
            heatmap_offset_sinkhorn_iterations=heatmap_offset_sinkhorn_iterations,
            logits_offset=logits_offset,
        )
        heatmap_init = heatmap_init.detach()
        states, _, _ = sequential_sampling(heatmap_init, config["num_starts"], D_batch, F_batch)
    else:
        states = config["initial_samples"].to(device)

    iter = 0
    while iter < max_steps:
        heatmap, _ = compute_adjusted_heatmap(
            net,
            D_batch,
            F_batch,
            heatmap_offset=heatmap_offset,
            heatmap_offset_renorm=heatmap_offset_renorm,
            heatmap_offset_sinkhorn_iterations=heatmap_offset_sinkhorn_iterations,
            logits_offset=logits_offset,
        )
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
        completed_steps = iter + 1
        if should_extend_finetune(global_best_costs_list, completed_steps, base_steps, config, has_extended):
            extra_steps = int(config.get("adaptive_extension_extra_steps", 50))
            max_steps += extra_steps
            has_extended = True
            if verbose:
                logger.info(
                    f"{'extend':^6} | extending finetune by {extra_steps} steps "
                    f"(new total: {max_steps})"
                )
        iter += 1

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


    if return_aux:
        with torch.no_grad():
            final_heatmap, final_logits = compute_adjusted_heatmap(
                net,
                D_batch,
                F_batch,
                heatmap_offset=heatmap_offset,
                heatmap_offset_renorm=heatmap_offset_renorm,
                heatmap_offset_sinkhorn_iterations=heatmap_offset_sinkhorn_iterations,
                logits_offset=logits_offset,
            )
            final_heatmap = final_heatmap.detach().clone()
            if final_logits is not None:
                final_logits = final_logits.detach().clone()
        aux = {
            "initial_samples": states.detach().clone(),
            "net": net,
            "final_heatmap": final_heatmap,
            "final_logits": final_logits,
        }
        return global_best_solutions, global_best_costs, gap, run_time, aux

    return global_best_solutions, global_best_costs, gap, run_time
