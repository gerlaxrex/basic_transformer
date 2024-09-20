from torch import nn
import torch
from basic_transformer.layers.mha import MultiHeadAttentionLayer


class Encoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_hidden_dim: int,
        dropout: float,
        num_heads: int,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        self.layer_norm_attn = nn.LayerNorm(embedding_dim).to(self.device)
        self.dropout = nn.Dropout(dropout).to(self.device)
        self.layer_norm_linear = nn.LayerNorm(embedding_dim).to(self.device)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=mlp_hidden_dim, out_features=embedding_dim),
        ).to(self.device)
        self.multi_head_attn = MultiHeadAttentionLayer(
            input_dim=embedding_dim,
            q_proj_dim=embedding_dim,
            k_proj_dim=embedding_dim,
            v_proj_dim=embedding_dim,
            num_heads=num_heads,
            device=self.device,
        )

    def forward(self, x: torch.tensor, mask: torch.tensor = None):
        # First perform the multi-head attn
        self_attn_mask = torch.einsum("bs,bj -> bsj", mask, mask)
        attn_out = self.multi_head_attn(x, x, x, self_attn_mask)
        # Then layer norm with sum the input
        z = self.dropout(self.layer_norm_attn(x + attn_out))

        # At this point lets build the final linear layer
        mlp_out = self.mlp(z)
        mlp_norm = self.dropout(self.layer_norm_linear(z + mlp_out))

        return mlp_norm


if __name__ == "__main__":
    encoder = Encoder(embedding_dim=10, num_heads=2, dropout=0.2, mlp_hidden_dim=20)
    x = torch.rand(size=(2, 3, 10))
    mask = torch.tril(torch.ones(2, 3, 3), diagonal=0)
    encoder(x, mask)
