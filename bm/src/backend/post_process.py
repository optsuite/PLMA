import torch

def local_search(states_batch: torch.Tensor, D_batch: torch.Tensor, F_batch: torch.Tensor, backend, local_search_iter=20, num_actions=20):
    """
    states_batch: (num_instances, num_samples, n)
    D_batch: (num_instances, n, n)
    F_batch: (num_instances, n, n)
    """
    p_batch = states_batch.clone()

    p_batch_new = backend.local_search(p_batch, D_batch, F_batch, num_actions, local_search_iter)

    final_costs = backend.compute_cost(p_batch_new, D_batch, F_batch)

    return p_batch_new, final_costs
