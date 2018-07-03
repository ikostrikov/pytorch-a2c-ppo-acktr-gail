import torch
import torch.nn as nn

from distributions import Categorical, DiagGaussian
from utils import init, init_normc_, initializer, Flatten

import numpy as np

class OptionCritic(nn.Module):
	def __init__(self, base_net, action_head, value_head, termination_head, policy_over_options, args):
		super(OptionCritic, self).__init__()
		self.base_net = base_net
		self.action_head = action_head
		self.value_head = value_head
		self.termination_head = termination_head
		self.policy_over_options = policy_over_options

		self.state_size = self.base_net.state_size

		self.value_loss_coef = args.value_loss_coef
		self.entropy_coef = args.entropy_coef
		self.termination_coef = args.termination_coef
		self.max_grad_norm = args.max_grad_norm
		self.lr = args.lr
		self.alpha = args.alpha
		self.delib = args.delib
		self.eps_thresh = torch.FloatTensor([args.eps_thresh])

		self.num_threads = args.num_processes
		self.num_options = args.num_options
		self.num_steps = args.num_steps

		self.current_options = torch.LongTensor(self.num_threads).random_(0, self.num_options).unsqueeze(dim=1)
		self.options_history = torch.LongTensor([])

		self.terminations = torch.zeros(self.num_threads)
		self.terminations_history = torch.FloatTensor([])

	def act(self, inputs, states, masks, deterministic=False):
		actor_features = self.base_net.main(inputs)

		self.terminations = self.termination_head(actor_features).gather(1,
		                                                                 self.current_options).squeeze()  # dimension: num_threads x 1
		rand_num = torch.rand(1)
		self.terminations = torch.where(self.terminations > rand_num,
		                                torch.zeros(self.num_threads),
		                                torch.ones(self.num_threads))

		self.select_new_option(actor_features)

		action_prob = self.action_head(actor_features)  # dimension: num_threads x num_options x num_actions
		dist = torch.distributions.Categorical(action_prob)

		if deterministic:
			action = dist.mode().squeeze()  # dimension: num_threads x num_options
		else:
			action = dist.sample().squeeze()  # dimension: num_threads x num_options
		action_log_prob = dist.log_probs(action).squeeze()  # dimension: num_threads x num_options

		action = action.gather(1, self.current_options)  # dimension: num_threads x 1
		action_log_prob = action_log_prob.gather(1, self.current_options)  # dimension: num_threads x 1

		value = self.get_value(inputs)  # dimension: num_threads x 1

		return value, action, action_log_prob, states

	def select_new_option(self, inputs):
		dist = self.policy_over_options(inputs) # dimension: num_threads x num_options
		dist = torch.distributions.Categorical(dist)

		dist_options = dist.sample().squeeze()[self.terminations == 1]
		random_options = torch.LongTensor(self.num_threads).random_(0, self.num_options).squeeze()[self.terminations == 1]
		random_numbers = torch.rand(self.num_threads).squeeze()[self.terminations == 1]

		new_options = torch.where(random_numbers > self.eps_thresh.expand_as(random_numbers), dist_options, random_options)

		old_options = self.current_options.squeeze().clone()
		self.current_options = self.current_options.squeeze()
		self.current_options[self.terminations == 1] = new_options

		self.terminations[old_options == self.current_options] = 0
		self.current_options = self.current_options.unsqueeze(-1)
		self.options_history = torch.cat((self.options_history, self.current_options), dim=0)
		self.terminations_history = torch.cat((self.terminations_history, self.terminations), dim=0)

	def get_value(self, inputs):
		actor_features = self.base_net.main(inputs)
		value = self.value_head(actor_features)  # dimension: num_threads x num_options or (num_steps * num_threads) x num_options
		if value.shape[0] == self.num_steps * self.num_threads:
			indices = self.options_history
		else:
			indices = self.current_options
		count = torch.arange(value.shape[0]).long()
		indices = indices.squeeze()
		value = value[count, indices].unsqueeze(-1)
		return value  # dimension: num_threads x 1 or (num_steps * num_threads) x 1

	def evaluate_actions(self, inputs, states, masks, action):
		value = self.get_value(inputs)

		actor_features = self.base_net.main(inputs)
		action_prob = self.action_head(actor_features)  # dimension: num_threads x num_options x num_actions

		# TODO: get the right option for each thread so that instead of having a 80x8x5 (N_STEPxP, O, A) we will have (N_STEPxP, A)
		count = torch.arange(self.num_steps * self.num_threads).long()
		indices = self.options_history.squeeze()
		action_prob = action_prob[count, indices, :]
		dist = torch.distributions.Categorical(action_prob)
		f_log_probs = lambda actions: dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)
		action_log_prob = f_log_probs(action)  # dimension: num_threads x num_options

		dist_entropy = dist.entropy().mean()

		# The expectancy of the value given state and policy over options
		options_value = self.get_options_value(actor_features)  # dimension: num_threads x 1

		return value, action_log_prob, dist_entropy, options_value

	def get_options_value(self, actor_features):
		value = self.value_head(actor_features).gather(1, self.options_history)  # dimension: num_steps * num_threads x num_options
		value = self.policy_over_options(actor_features) * value
		return torch.sum(value, dim=1).unsqueeze(1)

	def act_enjoy(self, inputs, states, masks, deterministic=False):
		num_threads = 1

		actor_features = self.base_net.main(inputs)

		count = torch.arange(num_threads).long()
		self.current_options = self.current_options[count]
		self.terminations = self.termination_head(actor_features)[count, self.current_options].squeeze()  # dimension: num_threads x 1
		rand_num = torch.rand(1)

		if self.terminations > rand_num:
			dist = self.policy_over_options(actor_features) # dimension: num_threads x num_options
			dist = torch.distributions.Categorical(dist)

			dist_options = dist.sample().squeeze()
			random_options = torch.LongTensor(num_threads).random_(0, self.num_options).squeeze()
			random_numbers = torch.rand(num_threads)

			self.current_options = torch.where(random_numbers > self.eps_thresh.expand_as(random_numbers),
			                                   dist_options, random_options)

		action_prob = self.action_head(actor_features).squeeze()[self.current_options]  # dimension: num_threads x num_options x num_actions
		dist = torch.distributions.Categorical(action_prob)

		if deterministic:
			action = dist.mode().squeeze()  # dimension: num_threads x num_options
		else:
			action = dist.sample().squeeze()  # dimension: num_threads x num_options

		return action.expand([1,1])


class Policy(nn.Module):
	def __init__(self, obs_shape, action_space, recurrent_policy):
		super(Policy, self).__init__()
		if len(obs_shape) == 3:
			self.base = CNNBase(obs_shape[0], recurrent_policy)
		elif len(obs_shape) == 1:
			assert not recurrent_policy, \
				"Recurrent policy is not implemented for the MLP controller"
			self.base = MLPBase(obs_shape[0])
		else:
			raise NotImplementedError

		if action_space.__class__.__name__ == "Discrete":
			num_outputs = action_space.n
			self.dist = Categorical(self.base.output_size, num_outputs)
		elif action_space.__class__.__name__ == "Box":
			num_outputs = action_space.shape[0]
			self.dist = DiagGaussian(self.base.output_size, num_outputs)
		else:
			raise NotImplementedError

		self.state_size = self.base.state_size

	def forward(self, inputs, states, masks):
		raise NotImplementedError

	def act(self, inputs, states, masks, deterministic=False):
		value, actor_features, states = self.base(inputs, states, masks)
		dist = self.dist(actor_features)

		if deterministic:
			action = dist.mode()
		else:
			action = dist.sample()

		action_log_probs = dist.log_probs(action)

		return value, action, action_log_probs, states

	def act_enjoy(self, inputs, states, masks, deterministic=False):
		_, action, _, _ = self.act(inputs, states, masks, deterministic=deterministic)

		return action

	def get_value(self, inputs, states, masks):
		value, _, _ = self.base(inputs, states, masks)
		return value

	def evaluate_actions(self, inputs, states, masks, action):
		value, actor_features, states = self.base(inputs, states, masks)
		dist = self.dist(actor_features)

		action_log_probs = dist.log_probs(action)
		dist_entropy = dist.entropy().mean()

		return value, action_log_probs, dist_entropy, states


class CNNBase(nn.Module):
	def __init__(self, num_inputs, use_gru):
		super(CNNBase, self).__init__()

		self.main = nn.Sequential(
				initializer(nn.Conv2d(num_inputs, 32, kernel_size=(4, 4), stride=2)),
				nn.ReLU(),
				initializer(nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2)),
				nn.ReLU(),
				Flatten(),
				initializer(nn.Linear(64, 48)),
				nn.ReLU(),
				initializer(nn.Linear(48, 32)),
				nn.ReLU()
		)

		if use_gru:
			self.gru = nn.GRUCell(512, 512)
			nn.init.orthogonal_(self.gru.weight_ih.data)
			nn.init.orthogonal_(self.gru.weight_hh.data)
			self.gru.bias_ih.data.fill_(0)
			self.gru.bias_hh.data.fill_(0)

		init_ = lambda m: init(m,
		                       nn.init.orthogonal_,
		                       lambda x: nn.init.constant_(x, 0))

		self.critic_linear = init_(nn.Linear(32, 1))

		self.train()

	@property
	def state_size(self):
		if hasattr(self, 'gru'):
			return 512
		else:
			return 1

	@property
	def output_size(self):
		return 32

	def forward(self, inputs, states, masks):
		x = self.main(inputs / 255.0)

		if hasattr(self, 'gru'):
			if inputs.size(0) == states.size(0):
				x = states = self.gru(x, states * masks)
			else:
				x = x.view(-1, states.size(0), x.size(1))
				masks = masks.view(-1, states.size(0), 1)
				outputs = []
				for i in range(x.size(0)):
					hx = states = self.gru(x[i], states * masks[i])
					outputs.append(hx)
				x = torch.cat(outputs, 0)

		return self.critic_linear(x), x, states


class MLPBase(nn.Module):
	def __init__(self, num_inputs):
		super(MLPBase, self).__init__()

		init_ = lambda m: init(m,
		                       init_normc_,
		                       lambda x: nn.init.constant_(x, 0))

		self.actor = nn.Sequential(
				init_(nn.Linear(num_inputs, 64)),
				nn.Tanh(),
				init_(nn.Linear(64, 64)),
				nn.Tanh()
		)

		self.critic = nn.Sequential(
				init_(nn.Linear(num_inputs, 64)),
				nn.Tanh(),
				init_(nn.Linear(64, 64)),
				nn.Tanh()
		)

		self.critic_linear = init_(nn.Linear(64, 1))

		self.train()

	@property
	def state_size(self):
		return 1

	@property
	def output_size(self):
		return 64

	def forward(self, inputs, states, masks):
		hidden_critic = self.critic(inputs)
		hidden_actor = self.actor(inputs)

		return self.critic_linear(hidden_critic), hidden_actor, states
