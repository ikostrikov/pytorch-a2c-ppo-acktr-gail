import random

import torch

class ObsNorm(object):
    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.count = torch.zeros(1).double() + 1e-2
        self.sum = torch.zeros(shape).double()
        self.sum_sqr = torch.zeros(shape).double() + 1e-2

        self.mean = torch.zeros(shape)
        self.std = torch.ones(shape)

    def cuda(self):
        self.count = self.count.cuda()
        self.sum = self.sum.cuda()
        self.sum_sqr = self.sum_sqr.cuda()

        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.count = self.count.cpu()
        self.sum = self.sum.cpu()
        self.sum_sqr = self.sum_sqr.cpu()

        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

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
