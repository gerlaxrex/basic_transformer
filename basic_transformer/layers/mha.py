from torch import nn
import numpy as np
import torch


class MultiHeadAttentionLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        q_proj_dim: int,
        k_proj_dim: int,
        v_proj_dim: int,
        num_heads: int,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device

        assert q_proj_dim % num_heads == 0 and k_proj_dim % num_heads == 0 and v_proj_dim % num_heads == 0

        self.num_heads = num_heads
        self.hidden_dim = input_dim // self.num_heads
        self.scale_factor = torch.scalar_tensor(np.sqrt(self.hidden_dim))
        self.Q_w = nn.Linear(in_features=input_dim, out_features=q_proj_dim, bias=False)
        self.K_w = nn.Linear(in_features=input_dim, out_features=k_proj_dim, bias=False)
        self.V_w = nn.Linear(in_features=input_dim, out_features=v_proj_dim, bias=False)
        self.O_w = nn.Linear(in_features=input_dim * num_heads, out_features=input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor, mask: torch.Tensor = None
    ):
        # Create the Q, K, V matrices
        B, seq_length_q, _ = q_in.shape
        _, seq_length_k, _ = k_in.shape
        q = self.Q_w(q_in)
        k = self.K_w(k_in)
        v = self.V_w(v_in)

        # At this point reshape the vector in order to have the heads
        q_h = torch.reshape(q, shape=(B, seq_length_q, self.num_heads, -1)).transpose(
            1, 2
        )  # (batch, num_heads, seq_length, hidden_dim)
        k_h = torch.reshape(k, shape=(B, seq_length_k, self.num_heads, -1)).transpose(
            1, 2
        )
        v_h = torch.reshape(v, shape=(B, seq_length_k, self.num_heads, -1)).transpose(
            1, 2
        )

        # Compute the attention matrix
        attn_matrix = q_h @ k_h.mT
        attn_matrix = torch.einsum("bhij, bhkj -> bhik", q_h, k_h)  # (B, num_heads, seq_lenght, emb_size // num_heads)
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn_matrix = attn_matrix.masked_fill(
                mask == 0, -1e9
            )  # Saturate the attention successive softmax for those tokens

        attn_matrix = self.softmax(
            attn_matrix / self.scale_factor
        )  # (batch, seq_length, seq_length)
        # Index values
        out = attn_matrix @ v_h  # (batch, seq_length, hidden_dim)

        # Reshape the matrix in order to obtain the old format
        out = out.transpose(0, 1).reshape(shape=(B, seq_length_q, -1))
        final_out = self.O_w(out)
        return final_out


if __name__ == "__main__":
    x = torch.rand(size=(2, 4, 10))
    src_mask = torch.tensor([[1,1,1,0], [1,1,0,0]])
    tgt_mask = torch.tensor([[1,1,0,0], [1,1,1,1]])
    attn_mask = torch.einsum("bs,bj -> bsj", tgt_mask, src_mask)
    mha = MultiHeadAttentionLayer(10, 10, 10, 10, 2)

    out = mha(x, x, x, mask=attn_mask)
    print(out)
