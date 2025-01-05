import torch
import torch.nn as nn

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    

class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        first, second = x.chunk(2, dim=self.dim)
        return first * torch.sigmoid(second)
    

class Transpose(nn.Module):
    def __init__(self, pair_dims):
        super().__init__()
        self.pair_dims = pair_dims
    
    def forward(self, x):
        return x.transpose(self.pair_dims[1], self.pair_dims[0])