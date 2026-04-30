import torch
from src.backend.post_process import local_search
from src.utils.ops import qap_cost

def run_mcmc_and_improve(D_batch, F_batch, start_states, heatmap, backend, num_chains, chain_length, **kwargs):
    """
    Run MCMC sampling and local search to improve the solutions.
    
    Args:
        D_batch: (num_instances, n, n)
        F_batch: (num_instances, n, n)
        start_states: (num_instances, num_starts, n)
        heatmap: (num_instances, n, n)
    """
    num_instances, num_starts, n = start_states.shape
    num_total_samples_per_instance = num_starts * num_chains
    local_search_iter = kwargs.get("local_search_iter", 20)
    num_actions = kwargs.get("num_actions", 20)

    # --- MCMC Sampling ---
    # Expand start states for parallel chains: [B, K, n] -> [B,K,M,n] -> [B, K*M, n]
    start_states = start_states.unsqueeze(2).repeat(1, 1, num_chains, 1).view(num_instances, num_total_samples_per_instance, n)
    terminal_states = backend.mcmc_step(start_states, torch.exp(heatmap.detach()), chain_length) # [B, K*M, n]

    # --- Local Search ---
    improved_states, costs = local_search(terminal_states, D_batch, F_batch, backend, local_search_iter, num_actions)
    changed = (terminal_states != improved_states).any(dim=2)
    ratio = changed.float().mean()
    improved_states = improved_states.to(torch.int64).view(num_instances, num_starts, num_chains, n) # [B, K*M, n] -> [B, K, M, n]

    # --- Compute Loss ---
    component_scores = torch.take_along_dim(heatmap.unsqueeze(1), terminal_states.to(torch.int64).unsqueeze(-1), dim=3).squeeze(-1) # [B, K*M, n]
    total_score = component_scores.sum(dim=2)  # [B, K*M]
    advantage = (costs.detach() - costs.mean(-1, keepdim=True).detach()) / (costs.std(-1, keepdim=True).detach() + 1e-8) # [B, K*M]
    rl_loss = (advantage * total_score).mean()

    # Note that heatmap is in the log space
    entropy = - torch.sum(heatmap * torch.exp(heatmap), dim=(1,2)).mean() 
    loss = rl_loss - entropy * kwargs["entropy_weight"]

    # --- Update Start States ---
    costs_matrix = costs.view(num_instances, num_starts, num_chains)
    best_cost_over_chains, best_chain_indices = costs_matrix.min(dim=2) # [B, K]

    updated_start_states = improved_states.take_along_dim(best_chain_indices[..., None, None], dim=2).squeeze(2)

    iter_best_costs, epoch_best_sol_idx = best_cost_over_chains.min(dim=1) # [B]
    iter_best_solutions = updated_start_states[torch.arange(num_instances), epoch_best_sol_idx]

    return updated_start_states, iter_best_solutions, iter_best_costs, loss, entropy, ratio


def ar_decode_and_improve(D_batch, F_batch, heatmap, backend, num_samples, **kwargs):
    """
    Autoregressive decode and local search to improve the solutions.

    Args:
        D_batch: (num_instances, n, n)
        F_batch: (num_instances, n, n)
        start_states: (num_instances, num_starts, n)
        heatmap: (num_instances, n, n)
    """
    num_instances = D_batch.shape[0]
    local_search_iter = kwargs.get("local_search_iter", 20)
    num_actions = kwargs.get("num_actions", 20)

    # --- MCMC Sampling ---
    states, init_costs, logprobs = sequential_sampling(heatmap, num_samples, D_batch, F_batch)

    # --- Local Search ---
    improved_states, costs = local_search(states, D_batch, F_batch, backend, local_search_iter, num_actions)
    improved_states = improved_states.to(torch.int64)
    changed = (states != improved_states).any(dim=2)
    ratio = changed.float().mean()

    # --- Compute Loss ---
    advantage = (costs.detach() - costs.mean(-1, keepdim=True).detach()) / (costs.std(-1, keepdim=True).detach() + 1e-8)  # [B, S]
    rl_loss = (advantage * logprobs).mean()
    # Note that heatmap is in the log space
    entropy = -torch.sum(heatmap * torch.exp(heatmap), dim=(1, 2)).mean()
    loss = rl_loss - entropy * kwargs["entropy_weight"]

    # --- Obtain iteation best costs and solutions ---
    iter_best_costs, epoch_best_sol_idx = costs.min(dim=1)  # [B]
    iter_best_solutions = improved_states[torch.arange(num_instances), epoch_best_sol_idx]

    return iter_best_solutions, iter_best_costs, loss, entropy, ratio


def sequential_sampling(heatmap, num_samples, D_batch, F_batch):
    """
    Construct an autoregressive model from heatmap and sample permutations sequentially (row by row).

    Args:
        heatmap: (B, n, n)
        num_samples: int, number of samples to generate

    Returns:
        permutations: (B, num_samples, n)
        costs: (B, num_samples)
        logprobs: (B, num_samples)
    """
    B, n, _ = heatmap.shape
    permutations = torch.zeros(B, num_samples, n, dtype=torch.long, device=heatmap.device)
    logprobs = torch.zeros(B, num_samples, device=heatmap.device)
    used = torch.zeros(B, num_samples, n, dtype=torch.bool, device=heatmap.device)

    for i in range(n):
        row_heatmap = heatmap[:, None, i, :].expand(B, num_samples, n)  # (B, S, n)

        # Mask already used facilities
        masked_heatmap = row_heatmap.masked_fill(used, float('-inf'))  # (B, S, n)
        logp = torch.log_softmax(masked_heatmap, -1)  # (B, S, n)

        # Sample based on masked log-softmax
        sampled = torch.distributions.Categorical(logits=masked_heatmap).sample()  # (B, S)
        permutations[:, :, i] = sampled

        # Accumulate log-probs using the sampled positions
        gathered = torch.gather(logp, dim=2, index=sampled.unsqueeze(-1)).squeeze(-1)  # (B, S)
        logprobs += gathered

        used = used.scatter(2, sampled.unsqueeze(-1), True)

    costs = qap_cost(D_batch, F_batch, permutations) 

    return permutations, costs, logprobs
