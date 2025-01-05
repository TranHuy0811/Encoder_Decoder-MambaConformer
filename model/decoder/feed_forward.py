import torch
import torch.nn as nn

# Decoder Feed Forward module
class DecFeedForward(nn.Module):
    def __init__(self, encoder_dim, ff_expand, activation, dropout_prob=0.1):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim * ff_expand),
            activation(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(encoder_dim * ff_expand, encoder_dim)
        )
    
    
    def forward(self, x):
        return self.sequential(x)