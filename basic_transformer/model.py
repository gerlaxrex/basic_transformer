import torch
from torch import nn
from torch import tensor
from basic_transformer.layers.encoder import Encoder
from basic_transformer.layers.decoder import Decoder
from basic_transformer.layers.positional_encoding import PositionalEncodingEmbeddings


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int = 1024,
        tgt_vocab_size: int = 1024,
        max_len: int = 2048,
        n_encoder_blocks: int = 3,
        n_decoder_blocks: int = 3,
        embedding_dim: int = 768,
        n_heads: int = 5,
        pad_idx: int = 0,
        device: str = "cpu",
    ):
        super().__init__()
        # Parameters
        self.src_vocab_size = src_vocab_size  # the number of tokens present in the final vocab + <sos>/<eos>/<pad>
        self.tgt_vocab_size = tgt_vocab_size
        self.max_len = max_len
        self.pad_idx = pad_idx

        self.n_encoder_blocks = (
            n_encoder_blocks  # number of transformer blocks in the encoder
        )
        self.n_decoder_blocks = (
            n_decoder_blocks  # number of transformer blocks in the decoder
        )
        self.embedding_dim = embedding_dim  # number of the embeddings dimension
        self.n_heads = n_heads  # number of MHA heads for the encoder part
        self.device = device

        # Embeddings
        self.encoder_embeddings = PositionalEncodingEmbeddings(
            d_model=self.embedding_dim,
            vocab_size=self.src_vocab_size+1,
            dropout=0.1,
            max_len=self.max_len,
            device=self.device,
        )

        self.decoder_embeddings = PositionalEncodingEmbeddings(
            d_model=self.embedding_dim,
            vocab_size=self.tgt_vocab_size + 1,
            dropout=0.1,
            max_len=self.max_len,
            device=self.device,
        )

        # Encoder, Decoder
        self.encoder_blocks = nn.ModuleList([
            Encoder(embedding_dim=self.embedding_dim,
                    num_heads=self.n_heads,
                    mlp_hidden_dim=1024,
                    device=self.device,
                    dropout=0.2)
            for _ in range(self.n_encoder_blocks)
        ])

        self.decoder_blocks = nn.ModuleList([
            Decoder(embedding_dim=self.embedding_dim,
                    num_heads=self.n_heads,
                    mlp_hidden_dim=1024,
                    device=self.device,
                    dropout=0.2)
            for _ in range(self.n_decoder_blocks)
        ])

        self.output_layer = nn.Linear(in_features=embedding_dim, out_features=self.tgt_vocab_size + 1)

    def encode(self, src: tensor, src_padding_mask: tensor) -> tensor:
        src_in = self.encoder_embeddings(src)
        x = src_in
        for enc_block in self.encoder_blocks:
            x = enc_block(x, src_padding_mask)

        return x

    def decode(self, src: tensor, tgt: tensor, src_padding_mask: tensor, tgt_padding_mask: tensor = None):
        tgt_in = self.decoder_embeddings(tgt)

        y = tgt_in
        for dec_block in self.decoder_blocks:
            y = dec_block(y, src, src_padding_mask, tgt_padding_mask)

        return y

    def forward(self,
                src: tensor,
                tgt: tensor) -> tensor:
        # Compute masks
        src_padding_mask = (src != self.pad_idx).to(float)  # (N, max_seq_len)
        tgt_padding_mask = (tgt != self.pad_idx).to(float)  # (N, max_seq_len)

        # Encoder, Decoder, output
        x = self.encode(src, src_padding_mask)
        y = self.decode(x, tgt, src_padding_mask, tgt_padding_mask)
        output_tensor = self.output_layer(y)  # (N, seq_lenght, tgt_vocab_size) -> logits for each sequence

        return output_tensor


    def greedy_decode(self,
                      src,
                      bos_idx,
                      max_len=80):
        B = src.shape[0]
        self.eval()
        src_padding_mask = (src != self.pad_idx).to(float)  # (N, max_seq_len)
        tgt = torch.tensor([[bos_idx]] * B).to(src.device)
        mem = self.encode(src, src_padding_mask)

        for _ in range(max_len - 1):
            out = self.decode(mem, tgt, src_padding_mask)
            scores = self.output_layer(out)
            next_token = torch.argmax(scores[:, -1:, :], dim=-1)
            print(next_token)
            tgt = torch.concat((tgt, next_token), dim=1)

            yield next_token





if __name__ == "__main__":
    from torch.nn.utils.rnn import pad_sequence
    import random

    transformer = Transformer(
        src_vocab_size=10,
        n_decoder_blocks=1,
        n_encoder_blocks=1,
        max_len=100,
        n_heads=2,
        tgt_vocab_size=10,
        embedding_dim=20,
        pad_idx=0,
    )

    inputs = [torch.randint(low=1, high=9, size=(random.randint(1, 10), ))
              for _ in range(4)]

    src_in = pad_sequence(inputs,
                          batch_first=True,
                          padding_value=0)

    out = transformer(src_in,
                src_in)



