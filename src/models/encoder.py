import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# Module 1: Graph Convolutional Layer (GCNLayer)
# ==============================================================================
class GCNLayer(nn.Module):
    """
    Single hybrid GCN layer.
    Input W shape: (B, n, n)
    Input X shape: (B, n, embed_dim)
    Output H shape: (B, n, embed_dim)
    """

    def __init__(self, embed_dim):
        super(GCNLayer, self).__init__()
        self.embed_dim = embed_dim

        self.weight_alg = nn.Linear(embed_dim, embed_dim, bias=False)
        self.norm_alg = nn.LayerNorm(embed_dim)
        self.bias = nn.Parameter(torch.randn(embed_dim))
        self.norm_final = nn.LayerNorm(embed_dim)

    def forward(self, W, X):
        X_res = X

        # 1. Algebraic branch
        support_alg = self.weight_alg(X)
        W_centered = W - W.mean(dim=[1, 2], keepdim=True)
        output_alg = self.norm_alg(torch.bmm(W_centered, support_alg))

        # 2. Residual, post-normalization and activation
        H = output_alg + self.bias
        H = self.norm_final(H + X_res) 
        H = F.silu(H)

        return H


# ==============================================================================
# Module 2: Cross Attention Layer (CrossAttention)
# ==============================================================================
class CrossAttention(nn.Module):
    """
    Single cross attention module
    Input hd shape: (B, n, embed_dim)
    Input hf shape: (B, n, embed_dim)
    Output hd_out shape: (B, n, embed_dim)
    Output hf_out shape: (B, n, embed_dim)
    """

    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.attn_d_on_f = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.attn_f_on_d = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.ffn_d = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.ReLU(), nn.Linear(embed_dim * 4, embed_dim))
        self.ffn_f = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.ReLU(), nn.Linear(embed_dim * 4, embed_dim))

        self.norm_d1 = nn.LayerNorm(embed_dim)
        self.norm_d2 = nn.LayerNorm(embed_dim)
        self.norm_f1 = nn.LayerNorm(embed_dim)
        self.norm_f2 = nn.LayerNorm(embed_dim)

    def forward(self, hd, hf):

        attn_d, _ = self.attn_d_on_f(query=hd, key=hf, value=hf)
        attn_f, _ = self.attn_f_on_d(query=hf, key=hd, value=hd)

        hd = self.norm_d1(hd + attn_d)
        hf = self.norm_f1(hf + attn_f)

        hd_out = self.norm_d2(hd + self.ffn_d(hd))
        hf_out = self.norm_f2(hf + self.ffn_f(hf))

        return hd_out, hf_out


# ==============================================================================
# Module 3: Encoder
# ==============================================================================
class Encoder(nn.Module):
    """
    Encoder module for embedding and interaction of two graphs (D and F).
    Structure: GCNLayer * num_gcn_layers + CrossAttention * num_att_layers
    Input D shape: (B, n, n)
    Input F shape: (B, n, n)
    Output hd_final shape: (B, n, embed_dim)
    Output hf_final shape: (B, n, embed_dim)
    """

    def __init__(self, initial_dim, embed_dim, num_heads, num_gcn_layers=8, num_att_layers=1):
        super(Encoder, self).__init__()

        self.hd_initial = nn.Parameter(torch.randn(1, 1, initial_dim))
        self.hf_initial = nn.Parameter(torch.randn(1, 1, initial_dim))

        self.projection_d = nn.Linear(initial_dim, embed_dim)
        self.projection_f = nn.Linear(initial_dim, embed_dim)

        self.num_gcn_layers = num_gcn_layers
        self.num_att_layers = num_att_layers

        self.gcn_layers_d = nn.ModuleList()
        self.gcn_layers_f = nn.ModuleList()
        self.attention_layers = nn.ModuleList()

        for _ in range(num_gcn_layers):
            self.gcn_layers_d.append(GCNLayer(embed_dim))
            self.gcn_layers_f.append(GCNLayer(embed_dim))
        for _ in range(num_att_layers):
            self.attention_layers.append(CrossAttention(embed_dim, num_heads))

    def forward(self, D, F):
        B, n, _ = D.shape
        hd = self.projection_d(self.hd_initial).expand(B, n, -1)
        hf = self.projection_f(self.hf_initial).expand(B, n, -1)

        for i in range(self.num_gcn_layers):
            hd = self.gcn_layers_d[i](D, hd)
            hf = self.gcn_layers_f[i](F, hf)

        for i in range(self.num_att_layers):
            hd, hf = self.attention_layers[i](hd, hf)
        return hd, hf
