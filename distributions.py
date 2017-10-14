import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import AddBias


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs, bias=False)
        self.ab = AddBias(num_outputs)

    def forward(self, x):
        x = self.ab(self.linear(x))
        return x

    def sample(self, x, deterministic):
        x = self(x)

        probs = F.softmax(x)
        if deterministic is False:
            action = probs.multinomial()
        else:
            action = probs.max(1)[1]
        return action

    def evaluate_actions(self, x, actions):
        x = self(x)

        log_probs = F.log_softmax(x)
        probs = F.softmax(x)

        action_log_probs = log_probs.gather(1, actions)

        dist_entropy = -(log_probs * probs).sum(-1).mean()
        return action_log_probs, dist_entropy


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()
        self.fc_mean = nn.Linear(64, num_outputs, bias=False)
        self.ab_mean = AddBias(num_outputs)
        self.ab_logstd = AddBias(num_outputs)

    def forward(self, x):
        x = self.fc_mean(x)
        x = self.ab_mean(x)
        action_mean = x

        #  An ugly hack for my KFAC implementation.
        zeros = Variable(torch.zeros(x.size()), volatile=x.volatile)
        if x.is_cuda:
            zeros = zeros.cuda()

        x = self.ab_logstd(zeros)
        action_logstd = x
        return action_mean, action_logstd

    def sample(self, x, deterministic):
        action_mean, action_logstd = self(x)

        action_std = action_logstd.exp()

        noise = Variable(torch.randn(action_std.size()))
        if action_std.is_cuda:
            noise = noise.cuda()

        if deterministic is False:
            action = action_mean + action_std * noise
        else:
            action = action_mean
        return action

    def evaluate_actions(self, x, actions):
        action_mean, action_logstd = self(x)

        action_std = action_logstd.exp()

        action_log_probs = -0.5 * ((actions - action_mean) / action_std).pow(2) - 0.5 * math.log(2 * math.pi) - action_logstd
        action_log_probs = action_log_probs.sum(1, keepdim=True)
        dist_entropy = 0.5 + math.log(2 * math.pi) + action_log_probs
        dist_entropy = dist_entropy.sum(-1).mean()

        return action_log_probs, dist_entropy
