import torch
import torch.nn as nn
from mamba_ssm.modules.mamba_simple import Mamba
from .feed_forward import DecFeedForward

class DecoderBlock(nn.Module):
    def __init__(self, encoder_dim, d_state, d_conv, mamba_expand, ff_expand, activation, dropout_prob=0.1):
        super().__init__()
        

        self.layernorm1 = nn.LayerNorm(encoder_dim)
        self.uni_mamba = Mamba(d_model=encoder_dim, d_state=d_state, d_conv=d_conv, expand=mamba_expand)
        self.dropout1 = nn.Dropout(p=dropout_prob)

        self.layernorm2 = nn.LayerNorm(encoder_dim)
        self.cross_mamba = Mamba(d_model=encoder_dim, d_state=d_state, d_conv=d_conv, expand=mamba_expand)
        self.dropout2 = nn.Dropout(p=dropout_prob)

        self.layernorm3 = nn.LayerNorm(encoder_dim)
        self.feed_forward = DecFeedForward(encoder_dim, ff_expand, activation, dropout_prob)
        self.dropout3 = nn.Dropout(p=dropout_prob)

    
    def forward(self, x, enc_out):
        x_prev = x
        x = self.layernorm1(x)
        x = self.uni_mamba(x)
        x = x_prev + self.dropout1(x)

        x_prev = x
        x = self.layernorm2(x)
        x = self.cross_mamba(torch.cat([enc_out, x], dim=1))[:, -x.size(1):]
        x = x_prev + self.dropout2(x)

        x_prev = x
        x = self.layernorm3(x)
        x = self.feed_forward(x)
        x = x_prev + self.dropout3(x)

        return x 
