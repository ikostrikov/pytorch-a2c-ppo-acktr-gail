import glob
import os

import torch
import torch.nn as nn

from a2c_ppo_acktr.envs import VecNormalize


# Get a render function
def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)


# State entropy maximization with random encoders for efficient exploration (RE3)
class CNNEmbeddingNetwork(nn.Module):
    def __init__(self, kwargs):
        super(CNNEmbeddingNetwork, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(kwargs['in_channels'], 32, (8, 8), stride=(4, 4)), nn.ReLU(),
            nn.Conv2d(32, 64, (4, 4), stride=(2, 2)), nn.ReLU(),
            nn.Conv2d(64, 32, (3, 3), stride=(1, 1)), nn.ReLU(), nn.Flatten(),
            nn.Linear(32 * 7 * 7, kwargs['embedding_size']))

    def forward(self, ob):
        x = self.main(ob)

        return x

class MLPEmbeddingNetwork(nn.Module):
    def __init__(self, kwargs):
        super(MLPEmbeddingNetwork, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(kwargs['input_dim'], 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, kwargs['embedding_size'])
        )

    def forward(self, ob):
        x = self.main(ob)

        return x

class SEM:
    def __init__(self,
                 ob_space,
                 action_space,
                 device,
                 num_updates
                 ):
        self.device = device
        self.num_updates = num_updates
        if action_space.__class__.__name__ == "Discrete":
            self.embedding_network = CNNEmbeddingNetwork(
                kwargs={'in_channels': ob_space.shape[0], 'embedding_size': 128})
        elif action_space.__class__.__name__ == 'Box':
            self.embedding_network = MLPEmbeddingNetwork(
                kwargs={'input_dim': ob_space.shape[0], 'embedding_size': 128})
        else:
            raise NotImplementedError('Please check the supported environments!')

        self.embedding_network.to(self.device)

        # fixed and random encoder
        for p in self.embedding_network.parameters():
            p.requires_grad = False

    def compute_intrinsic_rewards(self, obs_buffer, update_step, k=5):
        size = obs_buffer.size()
        obs = obs_buffer[:size[0] - 1]
        intrinsic_rewards = torch.zeros(size=(size[0] - 1, size[1], 1))

        for process in range(size[1]):
            encoded_obs = self.embedding_network(obs[:, process].to(self.device))
            for step in range(size[0] - 1):
                dist = torch.norm(encoded_obs[step] - encoded_obs, p=2, dim=1)
                H_step = torch.log(dist.sort().values[k + 1] + 1.)
                intrinsic_rewards[step, process, 0] = H_step

        return intrinsic_rewards * (1. - update_step / self.num_updates)