import torch
import torch.nn as nn
from .feed_forward import EncFeedForward
from .bi_mamba import BiMamba
from .conv_module import ConvolutionModule

# Mamba Conformer Block
class Encoder_Block(nn.Module):
    def __init__(self, encoder_dim, ff_expand, d_state, d_conv, mamba_expand, conv_expand, depth_ks, dropout_prob=0.1):
        super().__init__()
        self.feed_forward1 = EncFeedForward(encoder_dim, ff_expand, dropout_prob)
        self.bi_mamba = BiMamba(encoder_dim, d_state, d_conv, mamba_expand)
        self.conv_module = ConvolutionModule(encoder_dim, conv_expand, depth_ks, dropout_prob)
        self.feed_forward2 = EncFeedForward(encoder_dim, ff_expand, dropout_prob)
        self.layernorm = nn.LayerNorm(encoder_dim)

    def compute_residual(self, x, module, residual_factor=1.0):
        return x + residual_factor * module(x)


    def forward(self, x):
        x = self.compute_residual(x, self.feed_forward1, residual_factor=0.5)
        x = self.compute_residual(x, self.bi_mamba)
        x = self.compute_residual(x, self.conv_module)
        x = self.compute_residual(x, self.feed_forward2, residual_factor=0.5)
        x = self.layernorm(x)
        return x

