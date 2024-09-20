import torch
from torch import nn


class PositionalEncodingEmbeddings(nn.Module):
    def __init__(self,
                 d_model: int,
                 vocab_size: int,
                 dropout: float = 0.1,
                 max_len: int = 5000,
                 device: str = "cpu"):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.device = device

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

        # Normal trainable embeddings map
        self.embedding_layer = nn.Embedding(
            num_embeddings=self.vocab_size + 1,
            embedding_dim=self.d_model,
            device=self.device,
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_length]``
        """
        embedding = self.embedding_layer(x)
        pe = self.pe[:, :x.size(1)]
        x = embedding + pe
        return self.dropout(x)


if __name__ == "__main__":
    enc = PositionalEncodingEmbeddings(d_model=20,
                                       vocab_size=10,
                                       dropout=0.1,
                                       max_len=100)

    res = enc(torch.tensor([[0, 1, 2, 3], [2, 1, 0, 3]]))
