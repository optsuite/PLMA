import math
import torch
from torch import nn
from src.utils.ops import log_sinkhorn


class HeatmapGenerator(nn.Module):
    """
    heatmap generator.

    1. Compute similarity matrix through inner product of hd and hf.
    2. (Optional) Sharpen the distribution by dividing by temperature coefficient.
    3. Normalize using log-space Sinkhorn algorithm.
    """

    def __init__(self, iterations: int = 3, temperature: float = 1.0, clipping_value: float = 1.0):
        """ 
        Args:
            iterations (int): Number of Sinkhorn iterations.
            temperature (float): Temperature coefficient. Lower temperature makes the output probability distribution more "sharp", closer to hard assignment (0 or 1).
        """
        super().__init__()
        self.temperature = temperature
        self.iterations = iterations
        self.clipping_value = clipping_value

    def forward(self, hd: torch.Tensor, hf: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hd (torch.Tensor): Distance embedding, shape (B, n, d).
            hf (torch.Tensor): Flow embedding, shape (B, n, d).

        Returns:
            torch.Tensor: Final heatmap (soft assignment matrix), shape (B, n, n).
        """

        log_alpha = torch.bmm(hd, hf.transpose(-2, -1)) / math.sqrt(hd.size(-1))  # (B, n, n)
        if self.clipping_value > 0:
            log_alpha = self.clipping_value * torch.tanh(log_alpha)
        heatmap = log_sinkhorn(log_alpha, self.iterations, self.temperature)
        return heatmap
