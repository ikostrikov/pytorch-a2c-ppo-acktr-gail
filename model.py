import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from running_stat import ObsNorm


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)


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


class CNNPolicy(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(CNNPolicy, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 8, stride=4, bias=False)
        self.ab1 = AddBias(32)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, bias=False)
        self.ab2 = AddBias(64)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1, bias=False)
        self.ab3 = AddBias(32)

        self.linear1 = nn.Linear(32 * 7 * 7, 512, bias=False)
        self.ab_fc1 = AddBias(512)

        self.critic_linear = nn.Linear(512, 1, bias=False)
        self.ab_fc2 = AddBias(1)

        num_outputs = action_space.n
        self.actor_linear = nn.Linear(512, num_outputs, bias=False)
        self.ab_fc3 = AddBias(num_outputs)

        self.apply(weights_init)

        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.linear1.weight.data.mul_(relu_gain)

        self.train()

    def forward(self, inputs):
        x = self.conv1(inputs / 255.0)
        x = self.ab1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.ab2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.ab3(x)
        x = F.relu(x)

        x = x.view(-1, 32 * 7 * 7)
        x = self.linear1(x)
        x = self.ab_fc1(x)
        x = F.relu(x)

        return self.ab_fc2(self.critic_linear(x)), self.ab_fc3(
            self.actor_linear(x))

    def act(self, inputs, deterministic=False):
        value, logits = self(inputs)
        probs = F.softmax(logits)
        if deterministic is False:
            action = probs.multinomial()
        else:
            action = probs.max(1)[1]
        return value, action

    def evaluate_actions(self, inputs, actions):
        assert inputs.dim() == 4, "Expect to have inputs in num_processes * num_steps x ... format"

        values, logits = self(inputs)

        log_probs = F.log_softmax(logits)
        probs = F.softmax(logits)

        action_log_probs = log_probs.gather(1, actions)

        dist_entropy = -(log_probs * probs).sum(-1).mean()

        return values, action_log_probs, dist_entropy


def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class MLPPolicy(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(MLPPolicy, self).__init__()

        self.obs_filter = ObsNorm((1, num_inputs), clip=5)
        self.action_space = action_space

        self.a_fc1 = nn.Linear(num_inputs, 64, bias=False)
        self.a_ab1 = AddBias(64)
        self.a_fc2 = nn.Linear(64, 64, bias=False)
        self.a_ab2 = AddBias(64)
        self.a_fc_mean = nn.Linear(64, action_space.shape[0], bias=False)
        self.a_ab_mean = AddBias(action_space.shape[0])
        self.a_ab_logstd = AddBias(action_space.shape[0])

        self.v_fc1 = nn.Linear(num_inputs, 64, bias=False)
        self.v_ab1 = AddBias(64)
        self.v_fc2 = nn.Linear(64, 64, bias=False)
        self.v_ab2 = AddBias(64)
        self.v_fc3 = nn.Linear(64, 1, bias=False)
        self.v_ab3 = AddBias(1)

        self.apply(weights_init_mlp)

        tanh_gain = nn.init.calculate_gain('tanh')
        #self.a_fc1.weight.data.mul_(tanh_gain)
        #self.a_fc2.weight.data.mul_(tanh_gain)
        self.a_fc_mean.weight.data.mul_(0.01)
        #self.v_fc1.weight.data.mul_(tanh_gain)
        #self.v_fc2.weight.data.mul_(tanh_gain)

        self.train()

    def cuda(self, **args):
        super(MLPPolicy, self).cuda(**args)
        self.obs_filter.cuda()

    def cpu(self, **args):
        super(MLPPolicy, self).cpu(**args)
        self.obs_filter.cpu()

    def forward(self, inputs):
        inputs.data = self.obs_filter(inputs.data)

        x = self.v_fc1(inputs)
        x = self.v_ab1(x)
        x = F.tanh(x)

        x = self.v_fc2(x)
        x = self.v_ab2(x)
        x = F.tanh(x)

        x = self.v_fc3(x)
        x = self.v_ab3(x)
        value = x

        x = self.a_fc1(inputs)
        x = self.a_ab1(x)
        x = F.tanh(x)

        x = self.a_fc2(x)
        x = self.a_ab2(x)
        x = F.tanh(x)

        x = self.a_fc_mean(x)
        x = self.a_ab_mean(x)
        action_mean = x

        #  An ugly hack for my KFAC implementation.
        zeros = Variable(torch.zeros(x.size()), volatile=x.volatile)
        if x.is_cuda:
            zeros = zeros.cuda()

        x = self.a_ab_logstd(zeros)
        action_logstd = x

        return value, action_mean, action_logstd

    def act(self, inputs, deterministic=False):
        value, action_mean, action_logstd = self(inputs)

        action_std = action_logstd.exp()

        noise = Variable(torch.randn(action_std.size()))
        if action_std.is_cuda:
            noise = noise.cuda()

        if deterministic is False:
            action = action_mean + action_std * noise
        else:
            action = action_mean
        return value, action

    def evaluate_actions(self, inputs, actions):
        assert inputs.dim() == 2, "Expect to have inputs in num_processes * num_steps x ... format"

        value, action_mean, action_logstd = self(inputs)

        action_std = action_logstd.exp()

        action_log_probs = -0.5 * ((actions - action_mean) / action_std).pow(2) - 0.5 * math.log(2 * math.pi) - action_logstd
        action_log_probs = action_log_probs.sum(1, keepdim=True)
        dist_entropy = 0.5 + math.log(2 * math.pi) + action_log_probs
        dist_entropy = dist_entropy.sum(-1).mean()

        return value, action_log_probs, dist_entropy
