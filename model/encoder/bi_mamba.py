import torch
import torch.nn as nn
from mamba_ssm.modules.mamba_simple import Mamba

class BiMamba(nn.Module):
    def __init__(self, encoder_dim, d_state, d_conv, mamba_expand):
        super().__init__()
        self.forward_mamba = Mamba(d_model=encoder_dim, d_state=d_state, d_conv=d_conv, expand=mamba_expand)
        self.backward_mamba = Mamba(d_model=encoder_dim, d_state=d_state, d_conv=d_conv, expand=mamba_expand)
        self.linear = nn.Linear(encoder_dim * 2, encoder_dim)

    
    def forward(self, x):
        x_forward = self.forward_mamba(x)
        x_backward = self.backward_mamba(x.flip(dims=[1]))

        x = torch.cat([x_forward, x_backward.flip(dims=[1])], dim=-1)
        x = self.linear(x)
        return x
