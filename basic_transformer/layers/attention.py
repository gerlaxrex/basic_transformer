from torch import nn
import numpy as np
import torch


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.scale_factor = torch.scalar_tensor(np.sqrt(hidden_dim))
        self.Q_w = nn.Linear(
            in_features=hidden_dim, out_features=hidden_dim, bias=False
        )
        self.K_w = nn.Linear(
            in_features=hidden_dim, out_features=hidden_dim, bias=False
        )
        self.V_w = nn.Linear(
            in_features=hidden_dim, out_features=hidden_dim, bias=False
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.tensor, mask: torch.tensor = None):
        # Create the Q, K, V matrices
        if mask is not None:
            x = torch.masked_select(x, mask)
        q = self.Q_w(x)
        k = self.K_w(x)
        v = self.V_w(x)

        # Compute the attention matrix
        attn_matrix = self.softmax(
            (q @ k.mT) / self.scale_factor
        )  # (batch, seq_length, seq_length)
        # Index values
        out = attn_matrix @ v  # (batch, seq_length, hidden_dim)
        return out
