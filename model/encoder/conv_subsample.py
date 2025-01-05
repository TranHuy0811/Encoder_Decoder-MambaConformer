import torch
import torch.nn as nn

class ConvolutionSubsample(nn.Module):
    def __init__(self, encoder_dim, in_dim, dropout_prob=0.1):
        super().__init__()
        in_channels = 1
        out_channels = encoder_dim

        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU()
        )
        self.linear= nn.Linear(encoder_dim * (((in_dim - 1) // 2) - 1) // 2, encoder_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
    

    def forward(self, x): # (batch, channel, time, freq) => (batch, time, encoder_dim)
        x = self.sequential(x)

        batch, chanel, time, freq = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(batch, time, chanel * freq)

        x = self.linear(x)
        x = self.dropout(x)
        return x
