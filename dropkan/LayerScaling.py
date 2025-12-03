import torch
import torch.nn as nn

class LayerScaling(nn.Module):
    def __init__(self, range=1, eps=1e-5):
        super(LayerScaling, self).__init__()
        self.range = range
        self.eps = eps

    def forward(self, x):
        min_val = x.min(dim=-1, keepdim=True)[0]
        max_val = x.max(dim=-1, keepdim=True)[0]
        x = (2 * self.range * (x - min_val) / (max_val - min_val)) - self.range
        return x