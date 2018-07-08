import torch
import torch.nn as nn
import torch.optim as optim

from model import CNNBase, OptionCritic

import numpy as np


class View(nn.Module):
	def __init__(self, shape):
		super(View, self).__init__()
		self.shape = shape

	def forward(self, x):
		return x.view((-1, ) + self.shape)


class A2OC(object):
	def __init__(self, envs, args):

		self.args = args

		self.num_threads = args.num_processes
		self.num_options = args.num_options
		self.num_steps = args.num_steps
		self.num_actions = envs.action_space.n

		self.envs = envs

		obs_shape = envs.observation_space.shape
		obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

		self.base = CNNBase(obs_shape[0], False)
		self.options_action_head = nn.Sequential(nn.Linear(self.base.output_size, self.num_actions * self.num_options),
		                                         View((self.num_options, self.num_actions)),
		                                         nn.Softmax(2))
		self.options_termination_head = nn.Sequential(nn.Linear(self.base.output_size, self.num_options),
		                                              nn.Sigmoid())
		self.options_value_head = nn.Linear(self.base.output_size, self.num_options)
		self.policy_over_options = nn.Sequential(nn.Linear(self.base.output_size, self.num_options),
		                                         nn.Softmax(1))

		self.actor_critic = OptionCritic(self.base,
		                                 self.options_action_head,
		                                 self.options_value_head,
		                                 self.options_termination_head,
		                                 self.policy_over_options,
		                                 args)

		if args.cuda:
			self.actor_critic.cuda()

		self.value_loss_coef = args.value_loss_coef
		self.entropy_coef = args.entropy_coef
		self.termination_loss_coef = args.termination_loss_coef
		self.max_grad_norm = args.max_grad_norm

		self.state_size = self.actor_critic.state_size

		self.optimizer = optim.RMSprop(self.actor_critic.parameters(), args.lr, eps=args.eps,
		                               alpha=args.alpha)

	def act(self, inputs, states, masks, deterministic=False):
		value, action, action_log_prob, states = self.actor_critic.act(inputs, states, masks, deterministic=deterministic)

		cpu_actions = action.squeeze(1).cpu().numpy()

		# Observe reward and next obs
		obs, reward, done, info = self.envs.step(cpu_actions)

		reward = np.add(reward, self.actor_critic.terminations * self.actor_critic.delib)
		reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()

		if self.args.cuda:
			reward = reward.cuda()

		return value, action, action_log_prob, states, obs, reward, done, info

	def evaluate_actions(self, inputs, states, masks, action):
		return self.actor_critic.evaluate_actions(inputs, states, masks, action)

	def update(self, rollouts):
		obs_shape = rollouts.observations.size()[2:]
		action_shape = rollouts.actions.size()[-1]
		num_steps, num_processes, _ = rollouts.rewards.size()

		values, action_log_probs, dist_entropy, options_value = self.actor_critic.evaluate_actions(
				rollouts.observations[:-1].view(-1, *obs_shape),
				rollouts.states[0].view(-1, self.actor_critic.state_size),
				rollouts.masks[:-1].view(-1, 1),
				rollouts.actions.view(-1, action_shape))

		values = values.view(num_steps, num_processes, 1)   # n_steps x n_processes x 1
		action_log_probs = action_log_probs.view(num_steps, num_processes, 1)   # n_steps x n_processes x 1

		advantages = rollouts.returns[:-1] - values     # n_steps x n_processes x 1
		value_loss = advantages.pow(2).mean()   # no dimension, just scalar

		action_loss = (advantages.detach() * action_log_probs).mean()  # no dimension, just scalar

		options_value = options_value.view(num_steps * num_processes)     # n_steps * n_processes
		values = values.view(num_steps * num_processes)   # n_steps x n_processes x 1

		V = options_value.max() * (1 - self.args.options_epsilon) + (self.args.options_epsilon * options_value.mean())
		V = V.detach()

		termination_loss = ((values - V + self.actor_critic.delib).detach() * self.actor_critic.terminations_history).mean()  # n_steps x n_processes x 1
		self.optimizer.zero_grad()
		(value_loss * self.value_loss_coef - action_loss -
		 dist_entropy * self.entropy_coef + termination_loss * self.termination_loss_coef).backward()

		nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

		self.optimizer.step()

		self.actor_critic.options_history = torch.tensor([]).long()
		self.actor_critic.terminations_history = torch.tensor([])

		return value_loss.item(), action_loss.item(), termination_loss, dist_entropy.item()

	def get_value(self, inputs, states, masks):
		return self.actor_critic.get_value(inputs)
