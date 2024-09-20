from torch import nn
import torch
from basic_transformer.layers.mha import MultiHeadAttentionLayer


class Decoder(nn.Module):
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
        # Dropout layers
        self.dropout_self_attn = nn.Dropout(dropout).to(self.device)
        self.dropout_attn = nn.Dropout(dropout).to(self.device)
        self.dropout_mlp = nn.Dropout(dropout).to(self.device)

        # Layer normalization layers
        self.layer_norm_self_attn = nn.LayerNorm(embedding_dim).to(self.device)
        self.layer_norm_attn = nn.LayerNorm(embedding_dim).to(self.device)
        self.layer_norm_linear = nn.LayerNorm(embedding_dim).to(self.device)

        # Linear layer
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(in_features=mlp_hidden_dim, out_features=embedding_dim),
        ).to(self.device)

        # MHA attention layers
        self.self_multi_head_attn = MultiHeadAttentionLayer(
            input_dim=embedding_dim,
            q_proj_dim=embedding_dim,
            k_proj_dim=embedding_dim,
            v_proj_dim=embedding_dim,
            num_heads=num_heads,
            device=self.device,
        )

        self.multi_head_attn = MultiHeadAttentionLayer(
            input_dim=embedding_dim,
            q_proj_dim=embedding_dim,
            k_proj_dim=embedding_dim,
            v_proj_dim=embedding_dim,
            num_heads=num_heads,
            device=self.device,
        )

    def forward(
        self,
        x: torch.tensor,
        encoder_inputs: torch.tensor,
        enc_mask: torch.tensor = None,
        dec_mask: torch.tensor = None,
    ):

        if dec_mask is None:
            dec_mask = torch.ones(x.shape[0], x.shape[1])

        # Create the causal mask first
        self_attn_mask = torch.einsum("bs,bj -> bsj", dec_mask, dec_mask)
        src_tgt_attn_mask = torch.einsum("bs,bj -> bsj", dec_mask, enc_mask)
        causal_mask = self_attn_mask.tril(diagonal=0).to(self.device)

        # First perform the multi-head self attention with the causal mask
        self_attn_out = self.self_multi_head_attn(x, x, x, mask=causal_mask)
        # Then layer norm with sum the input
        y = self.dropout_self_attn(self.layer_norm_self_attn(x + self_attn_out))

        # Then perform attention with k q as the encoder last output
        attn_out = self.multi_head_attn(y, encoder_inputs, encoder_inputs, mask=src_tgt_attn_mask)
        z = self.dropout_attn(self.layer_norm_attn(y + attn_out))

        # Then the final MLP layer
        mlp_out = self.mlp(z)
        mlp_out = self.dropout_mlp(self.layer_norm_linear(z + mlp_out))

        return mlp_out


if __name__ == "__main__":
    encoder = Decoder(embedding_dim=10, num_heads=2, dropout=0.2, mlp_hidden_dim=20)
    x = torch.rand(size=(2, 3, 10))
    mask = torch.tril(torch.ones(2, 3, 3), diagonal=0)
    encoder(x, mask)
