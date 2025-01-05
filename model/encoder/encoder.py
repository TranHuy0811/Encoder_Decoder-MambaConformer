import torch
import torch.nn as nn
from .encoder_block import Encoder_Block
from .conv_subsample import ConvolutionSubsample

class Encoder(nn.Module):
    def __init__(self, encoder_dim, in_dim, 
                 enc_num_blocks, ff_expand, d_state, d_conv, mamba_expand, conv_expand, depth_ks, 
                 dropout_prob=0.1
        ):
        super().__init__()

        self.conv_subsample = ConvolutionSubsample(encoder_dim, in_dim, dropout_prob)
        self.encoder_blocks = nn.ModuleList(
            [Encoder_Block(encoder_dim, ff_expand, d_state, d_conv, mamba_expand, conv_expand, depth_ks, dropout_prob) for _ in range(enc_num_blocks)]
        )


    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(1) # (batch, freq, time) => (batch, channel, freq, time)
        x = self.conv_subsample(x) # (batch, channel, freq, time) => (batch, time, encoder_dim)

        for block in self.encoder_blocks:
            x = block(x)
            
        return x
        