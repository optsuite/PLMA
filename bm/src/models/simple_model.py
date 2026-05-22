import torch
from torch import nn

from src.utils.ops import log_sinkhorn


class SimpleNet(nn.Module):
    """
    Baseline model that directly optimizes an n x n assignment logit matrix.
    Output is in log-space after Sinkhorn normalization.
    """

    def __init__(self, n, iterations=1, temperature=1.0, initial_W=None):
        super().__init__()
        if initial_W is None:
            initial_W = torch.randn(n, n)
        self.W = nn.Parameter(initial_W)
        self.iterations = iterations
        self.temperature = temperature

    def forward(self, D, F):
        B = D.size(0)
        log_heatmap = log_sinkhorn(self.W, self.iterations, self.temperature)
        return log_heatmap.unsqueeze(0).expand(B, -1, -1)

    def forward_logits(self, D, F):
        B = D.size(0)
        return self.W.unsqueeze(0).expand(B, -1, -1)
