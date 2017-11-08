import torch
import torch.nn as nn


class ObsNorm(nn.Module):
    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        super(ObsNorm, self).__init__()
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.register_buffer('count', torch.zeros(1).double() + 1e-2)
        self.register_buffer('sum', torch.zeros(shape).double())
        self.register_buffer('sum_sqr', torch.zeros(shape).double() + 1e-2)

        self.register_buffer('mean', torch.zeros(shape),)
        self.register_buffer('std', torch.ones(shape))

    def update(self, x):
        self.count += x.size(0)
        self.sum += x.sum(0, keepdim=True).double()
        self.sum_sqr += x.pow(2).sum(0, keepdim=True).double()

        self.mean = self.sum / self.count
        self.std = (self.sum_sqr / self.count - self.mean.pow(2)).clamp(1e-2, 1e9).sqrt()

        self.mean = self.mean.float()
        self.std = self.std.float()

    def __call__(self, x):
        if self.demean:
            x = x - self.mean
        if self.destd:
            x = x / self.std
        if self.clip:
            x = x.clamp(-self.clip, self.clip)
        return x
