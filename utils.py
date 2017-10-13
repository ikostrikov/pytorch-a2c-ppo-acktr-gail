import torch
import torch.nn as nn


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, out_features):
        super(AddBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(out_features, 1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self.bias.t().view(1, -1)
        else:
            bias = self.bias.t().view(1, -1, 1, 1)

        return x + bias
