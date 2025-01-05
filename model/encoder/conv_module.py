import torch
import torch.nn as nn
from ..modules import Swish, GLU, Transpose

class ConvolutionModule(nn.Module):
    def __init__(self, encoder_dim, conv_expand, depth_ks, dropout_prob=0.1):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            Transpose(pair_dims=(1, 2)),
            nn.Conv1d(encoder_dim, encoder_dim * conv_expand, kernel_size=1, stride=1, padding=0),
            GLU(dim=1),
            nn.Conv1d(encoder_dim, encoder_dim, kernel_size=depth_ks, stride=1, padding=(depth_ks -1) // 2, groups=encoder_dim, bias=False),
            nn.BatchNorm1d(encoder_dim),
            Swish(),
            nn.Conv1d(encoder_dim, encoder_dim, kernel_size=1, stride=1, padding=0),
            nn.Dropout(p=dropout_prob)
        )

    
    def forward(self, x):
        return self.sequential(x).transpose(1, 2)