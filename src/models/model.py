from torch import nn
from .encoder import Encoder
from .decoder import HeatmapGenerator


class QAPNet(nn.Module):
    """
    QAPNet model, which consists of an encoder and a generator for heatmaps.
    """

    def __init__(
        self,
        init_dim=32,
        embed_dim=128,
        num_heads=8,
        num_gcn_layers=8,
        num_att_layers=1,
        num_iterations=1,
        temperature=1.0,
        clipping_value=1.0,
    ):
        super(QAPNet, self).__init__()
        self.encoder = Encoder(
            initial_dim=init_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_gcn_layers=num_gcn_layers,
            num_att_layers=num_att_layers,
        )
        self.heatmap_generator = HeatmapGenerator(
            iterations=num_iterations,
            temperature=temperature,
            clipping_value=clipping_value,
        )

    def forward(self, D, F):
        """
        Args:
            D (torch.Tensor): Distance matrix, shape (B, n, n).
            F (torch.Tensor): Flow matrix, shape (B, n, n).
        Returns:
            heatmap (torch.Tensor): Final heatmap, shape (B, n, n). 
        """
        hd, hf = self.encoder(D, F)
        heatmap = self.heatmap_generator(hd, hf)
        return heatmap
