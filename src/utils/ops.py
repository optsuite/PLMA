import torch

def qap_cost(D, F, permutations):
    """
    Compute QAP cost for given permutations.

    Args:
        D (torch.Tensor): (B, n, n) distance matrices
        F (torch.Tensor): (B, n, n) flow matrices
        permutations (torch.Tensor): (B, S, n) permutations with S samples per instance

    Returns:
        torch.Tensor: (B, S)
    """

    F_rows_permuted = torch.take_along_dim(F.unsqueeze(1), permutations.unsqueeze(-1), dim=-2)
    F_pi = F_rows_permuted.take_along_dim(permutations.unsqueeze(-2), dim=-1) 
    return torch.einsum('bsij,bij->bs', F_pi, D)


def log_sinkhorn(log_alpha, iterations=1, temperature=1.0):
    """Sinkhorn normalization in log-space."""
    log_alpha = log_alpha / temperature
    for _ in range(iterations):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
    return log_alpha