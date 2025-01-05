import torch
import torch.nn as nn
from .decoder_block import DecoderBlock

class Decoder(nn.Module):
    def __init__(self, encoder_dim, vocab_size, dec_num_blocks, padding_idx, d_state, d_conv, mamba_expand, ff_expand, activation, dropout_prob=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, encoder_dim, padding_idx=padding_idx)

        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(encoder_dim, d_state, d_conv, mamba_expand, ff_expand, activation, dropout_prob) for _ in range(dec_num_blocks)]
        )
        self.layernorm = nn.LayerNorm(encoder_dim)


    def forward(self, x, enc_out):
        x = self.embedding(x)
        for block in self.decoder_blocks:
            x = block(x, enc_out)

        x = self.layernorm(x)
        return x
