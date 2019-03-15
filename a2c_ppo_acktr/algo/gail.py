import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

from baselines.common.running_mean_std import RunningMeanStd


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, device):
        super(Discriminator, self).__init__()

        self.device = device

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)).to(device)

        self.trunk.train()

        self.optimizer = torch.optim.Adam(self.trunk.parameters())

        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())

    def compute_grad_pen(self,
                         expert_state,
                         expert_action,
                         policy_state,
                         policy_action,
                         lambda_=10):
        alpha = torch.rand(expert_state.size(0), 1)
        expert_data = torch.cat([expert_state, expert_action], dim=1)
        policy_data = torch.cat([policy_state, policy_action], dim=1)

        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * policy_data
        mixup_data.requires_grad = True

        disc = self.trunk(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def update(self, expert_loader, rollouts, obsfilt=None):
        self.train()

        policy_data_generator = rollouts.feed_forward_generator(
            None, mini_batch_size=expert_loader.batch_size)

        loss = 0
        n = 0
        for expert_batch, policy_batch in zip(expert_loader,
                                              policy_data_generator):
            policy_state, policy_action = policy_batch[0], policy_batch[2]
            policy_d = self.trunk(
                torch.cat([policy_state, policy_action], dim=1))

            expert_state, expert_action = expert_batch
            expert_state = obsfilt(expert_state.numpy(), update=False)
            expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_action = expert_action.to(self.device)
            expert_d = self.trunk(
                torch.cat([expert_state, expert_action], dim=1))

            expert_loss = F.binary_cross_entropy_with_logits(
                expert_d,
                torch.ones(expert_d.size()).to(self.device))
            policy_loss = F.binary_cross_entropy_with_logits(
                policy_d,
                torch.zeros(policy_d.size()).to(self.device))

            gail_loss = expert_loss + policy_loss
            grad_pen = self.compute_grad_pen(expert_state, expert_action,
                                             policy_state, policy_action)

            loss += (gail_loss + grad_pen).item()
            n += 1

            self.optimizer.zero_grad()
            (gail_loss + grad_pen).backward()
            self.optimizer.step()
        return loss / n

    def predict_reward(self, state, action, gamma, masks, update_rms=True):
        with torch.no_grad():
            self.eval()
            d = self.trunk(torch.cat([state, action], dim=1))
            s = torch.sigmoid(d)
            reward = s.log() - (1 - s).log()
            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

            return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)


def get_expert_traj_loaders(file_name,
                            batch_size,
                            num_train_traj=4,
                            num_valid_traj=4,
                            subsamp_freq=20):
    with h5py.File(file_name, 'r') as f:
        dataset_size = f['obs_B_T_Do'].shape[0]  # full dataset size

        states = f['obs_B_T_Do'][:dataset_size, ...][...]
        actions = f['a_B_T_Da'][:dataset_size, ...][...]
        rewards = f['r_B_T'][:dataset_size, ...][...]
        lens = f['len_B'][:dataset_size, ...][...]

    # Stack everything together
    perm = np.random.permutation(np.arange(dataset_size))
    train_random_idxs = perm[:num_train_traj]
    valid_random_idxs = perm[num_train_traj:num_train_traj + num_valid_traj]

    start_times = np.random.randint(0, subsamp_freq, size=lens.shape[0])

    def make_tensor(idxs):
        xs, ys = [], []
        for i in idxs:
            l = lens[i]
            for j in range(start_times[i], l, subsamp_freq):
                state = states[i, j].reshape(1, -1)
                action = actions[i, j].reshape(1, -1)
                xs.append(state)
                ys.append(action)
        x = np.concatenate(xs, axis=0)
        x = torch.from_numpy(x).float()
        y = np.concatenate(ys, axis=0)
        y = torch.from_numpy(y).float()
        return x, y

    train_x, train_y = make_tensor(train_random_idxs)
    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)

    valid_x, valid_y = make_tensor(valid_random_idxs)
    valid_dataset = torch.utils.data.TensorDataset(valid_x, valid_y)

    kwargs = {'num_workers': 0}
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **kwargs)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **kwargs)

    return train_loader, valid_loader
