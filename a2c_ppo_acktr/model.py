import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):

    def __init__(self,
                 obs_shape,
                 action_space,
                 base=None,
                 base_kwargs=None,
                 navi=False):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], **base_kwargs)
        # print(self.base.state_dict().keys())








        ######FIXME TEMPORARY for gibson experiments

        # pretrained_dict = torch.load(
        #     "./single_cube_cnn_ppo-v1.pt", map_location=torch.device('cpu'))

        # model_dict = self.base.state_dict()
        # keys = list(pretrained_dict.keys())
        # for key in keys:
        #     pretrained_dict[key.replace("cnn",
        #                                 "main")] = pretrained_dict.pop(key)
        # pretrained_dict_filtered = {
        #     k: v for k, v in pretrained_dict.items() if k in model_dict
        # }
        # model_dict.update(pretrained_dict_filtered)
        # self.base.load_state_dict(model_dict)
        # self.base.train()
        # print("=== loaded pretrained CNN ====")

        ######FIXME END






        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            net_outputs = self.base.output_size
            if navi:
                net_outputs = 256 * 10
            self.dist = Categorical(net_outputs, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)






            ######FIXME TEMPORARY for gibson experiments

            # model_dict = self.dist.state_dict()
            # pretrained_dict_filtered = {
            #     k: v for k, v in pretrained_dict.items() if k in model_dict
            # }
            # model_dict.update(pretrained_dict_filtered)
            # self.dist.load_state_dict(model_dict)
            # self.dist.train()
            # print("=== loaded pretrained DiagGauss ====")

            ######FIXME END







        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class RandomPolicy(Policy):

    def __init__(self,
                 obs_shape,
                 action_space,
                 base=None,
                 base_kwargs=None,
                 navi=False):
        super(RandomPolicy, self).__init__(obs_shape, action_space, base,
                                           base_kwargs, navi)
        self.action_space = action_space

    @property
    def is_recurrent(self):
        pass

    @property
    def recurrent_hidden_state_size(self):
        return torch.tensor(10)

    def forward(self, inputs, rnn_hxs, masks):
        pass

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        return torch.tensor([10]), torch.tensor([[
            np.random.choice(self.action_space.n)
        ]]), torch.tensor([1]), torch.tensor([range(10)])

    def get_value(self, inputs, rnn_hxs, masks):
        return torch.tensor(-1)

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        return None


class NNBase(nn.Module):

    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx], hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):

    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        # show(make_grid((inputs/255.0).view(4,3,84,84)))
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):

    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class NaviBase(NNBase):

    def __init__(self,
                 num_inputs,
                 recurrent=False,
                 num_streets=4,
                 hidden_size=256,
                 total_hidden_size=(256 * 10)):
        if recurrent:
            raise NotImplementedError("recurrent policy not done yet")
        super(NaviBase, self).__init__(recurrent, hidden_size, hidden_size)
        self.num_streets = num_streets
        init_cnn = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init
                                  .constant_(x, 0),
                                  nn.init.calculate_gain('relu'))
        init_dense = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                    constant_(x, 0), np.sqrt(2))

        self.img_embed = nn.Sequential(
            init_cnn(nn.Conv2d(3, 32, 3, stride=2)), nn.ReLU(),
            init_cnn(nn.Conv2d(32, 64, 5, stride=2)), nn.ReLU(),
            init_cnn(nn.Conv2d(64, 32, 5, stride=2)), nn.ReLU(), Flatten(),
            init_cnn(nn.Linear(32 * 8 * 8, hidden_size)), nn.ReLU())

        # NeED to look if different activation functions

        self.coord_embed = nn.Sequential(
            init_dense(nn.Linear(2, 64)), nn.Tanh(),
            init_dense(nn.Linear(64, hidden_size)), nn.Tanh())

        self.number_embed = nn.Sequential(
            init_dense(nn.Linear(10, 64)), nn.Tanh())

        self.street_embed = nn.Sequential(
            init_dense(nn.Linear(self.num_streets, hidden_size)), nn.Tanh())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(total_hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        image = inputs[:, :3, :, :]
        rel_gps = inputs[:, 3, 0, :2]
        abs_gps = inputs[:, 3, 0, 2:4]

        vis_street_names = inputs[:, 3, 1, :2 * self.num_streets]
        vis_house_numbers = torch.cat(
            [inputs[:, 3, 2, :84], inputs[:, 3, 3, :36]], dim=1)
        goal_house_numbers = inputs[:, 3, 4, :40]
        goal_street_name = inputs[:, 3, 4, 40:40 + self.num_streets]

        img_e = self.img_embed(image)
        rel_gps_e = self.coord_embed(rel_gps)
        abs_gps_e = self.coord_embed(abs_gps)

        goal_hn_e = torch.tensor([])
        vis_hn_e = torch.tensor([])
        vis_sn_e = torch.tensor([])

        if torch.cuda.is_available():
            goal_hn_e = goal_hn_e.cuda()
            vis_hn_e = vis_hn_e.cuda()
            vis_sn_e = vis_sn_e.cuda()

        for i in range(4):
            goal_hn_embed = self.number_embed(
                goal_house_numbers[:, i * 10:(i + 1) * 10])
            goal_hn_e = torch.cat((goal_hn_e, goal_hn_embed), dim=1)

        goal_sn_e = self.street_embed(goal_street_name)

        for j in range(3):
            offset = j * 40
            for i in range(4):
                vis_hn_embed = self.number_embed(
                    vis_house_numbers[:, offset + (i * 10):offset +
                                      ((i + 1) * 10)])
                vis_hn_e = torch.cat((vis_hn_e, vis_hn_embed), dim=1)

        for i in range(2):
            vis_sn_embed = self.street_embed(
                vis_street_names[:, i * self.num_streets:(i + 1) *
                                 self.num_streets])
            vis_sn_e = torch.cat((vis_sn_e, vis_sn_embed), dim=1)

        x = torch.cat((img_e, rel_gps_e, abs_gps_e, goal_hn_e, goal_sn_e,
                       vis_hn_e, vis_sn_e),
                      dim=1)
        return self.critic_linear(x), x, rnn_hxs
