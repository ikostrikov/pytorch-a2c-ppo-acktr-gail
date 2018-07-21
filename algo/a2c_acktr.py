import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from model import Policy
from .kfac import KFACOptimizer


class A2C_ACKTR(object):
	def __init__(self,
	             obs_shape,
	             action_space,
	             recurrent_policy,
	             value_loss_coef,
	             entropy_coef,
	             envs,
	             lr=None,
	             eps=None,
	             alpha=None,
	             max_grad_norm=None,
	             acktr=False,
	             cuda=False):

		self.actor_critic = Policy(obs_shape, action_space, recurrent_policy)
		if cuda:
			self.actor_critic.cuda()

		self.acktr = acktr

		self.envs = envs

		self.value_loss_coef = value_loss_coef
		self.entropy_coef = entropy_coef

		self.max_grad_norm = max_grad_norm

		if acktr:
			self.optimizer = KFACOptimizer(self.actor_critic)
		else:
			self.optimizer = optim.RMSprop(
					self.actor_critic.parameters(), lr, eps=eps, alpha=alpha)

		self.state_size = self.actor_critic.state_size

	def act(self, inputs, states, masks, deterministic=False):

		value, action, action_log_prob, states = self.actor_critic.act(inputs,
		                                                               states,
		                                                               masks,
		                                                               deterministic=deterministic)

		cpu_actions = action.squeeze(1).cpu().numpy()

		# Observe reward and next obs
		obs, reward, done, info = self.envs.step(cpu_actions)
		reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()

		return value, action, action_log_prob, states, obs, reward, done, info

	def get_value(self, inputs, states, masks):
		return self.actor_critic.get_value(inputs, states, masks)

	def evaluate_actions(self, inputs, states, masks, action):
		return self.actor_critic.evaluate_actions(inputs, states, masks, action)

	def update(self, rollouts):
		obs_shape = rollouts.observations.size()[2:]
		action_shape = rollouts.actions.size()[-1]
		num_steps, num_processes, _ = rollouts.rewards.size()

		values, action_log_probs, dist_entropy, states = self.evaluate_actions(
				rollouts.observations[:-1].view(-1, *obs_shape),
				rollouts.states[0].view(-1, self.state_size),
				rollouts.masks[:-1].view(-1, 1),
				rollouts.actions.view(-1, action_shape))

		values = values.view(num_steps, num_processes, 1)
		action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

		advantages = rollouts.returns[:-1] - values
		value_loss = advantages.pow(2).mean()

		action_loss = -(advantages.detach() * action_log_probs).mean()

		if self.acktr and self.optimizer.steps % self.optimizer.Ts == 0:
			# Sampled fisher, see Martens 2014
			self.actor_critic.zero_grad()
			pg_fisher_loss = -action_log_probs.mean()

			value_noise = torch.randn(values.size())
			if values.is_cuda:
				value_noise = value_noise.cuda()

			sample_values = values + value_noise
			vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()

			fisher_loss = pg_fisher_loss + vf_fisher_loss
			self.optimizer.acc_stats = True
			fisher_loss.backward(retain_graph=True)
			self.optimizer.acc_stats = False

		self.optimizer.zero_grad()
		(value_loss * self.value_loss_coef + action_loss -
		 dist_entropy * self.entropy_coef).backward()

		if self.acktr == False:
			nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
			                         self.max_grad_norm)

		self.optimizer.step()

		return value_loss.item(), action_loss.item(), 0, dist_entropy.item()
