import torch
import torch.nn as nn
from ..modules import Swish

# Encoder Feed Forward module
class EncFeedForward(nn.Module):
    def __init__(self, encoder_dim, ff_expand, dropout_prob=0.1):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, encoder_dim * ff_expand),
            Swish(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(encoder_dim * ff_expand, encoder_dim),
            nn.Dropout(p=dropout_prob)
        )


    def forward(self, x):
        return self.sequential(x)