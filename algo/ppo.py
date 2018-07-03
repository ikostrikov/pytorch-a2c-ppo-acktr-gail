import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from model import Policy


class PPO(object):
	def __init__(self,
	             obs_shape,
	             action_space,
	             recurrent_policy,
	             clip_param,
	             ppo_epoch,
	             num_mini_batch,
	             value_loss_coef,
	             entropy_coef,
	             envs,
	             lr=None,
	             eps=None,
	             max_grad_norm=None,
	             cuda=False):
		self.actor_critic = Policy(obs_shape, action_space, recurrent_policy)

		if cuda:
			self.actor_critic.cuda()

		self.envs = envs

		self.clip_param = clip_param
		self.ppo_epoch = ppo_epoch
		self.num_mini_batch = num_mini_batch

		self.value_loss_coef = value_loss_coef
		self.entropy_coef = entropy_coef

		self.max_grad_norm = max_grad_norm

		self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr, eps=eps)

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
		return self.actor_critic.evaluate_actions(inputs, states, masks)

	def update(self, rollouts):
		advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
		advantages = (advantages - advantages.mean()) / (
			advantages.std() + 1e-5)

		value_loss_epoch = 0
		action_loss_epoch = 0
		dist_entropy_epoch = 0

		for e in range(self.ppo_epoch):
			if hasattr(self.actor_critic.base, 'gru'):
				data_generator = rollouts.recurrent_generator(
						advantages, self.num_mini_batch)
			else:
				data_generator = rollouts.feed_forward_generator(
						advantages, self.num_mini_batch)

			for sample in data_generator:
				observations_batch, states_batch, actions_batch, \
				return_batch, masks_batch, old_action_log_probs_batch, \
				adv_targ = sample

				# Reshape to do in a single forward pass for all steps
				values, action_log_probs, dist_entropy, states = self.actor_critic.evaluate_actions(
						observations_batch, states_batch,
						masks_batch, actions_batch)

				ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
				surr1 = ratio * adv_targ
				surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
				                    1.0 + self.clip_param) * adv_targ
				action_loss = -torch.min(surr1, surr2).mean()

				value_loss = (return_batch - values).pow(2).mean()

				self.optimizer.zero_grad()
				(value_loss * self.value_loss_coef + action_loss -
				 dist_entropy * self.entropy_coef).backward()
				nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
				                         self.max_grad_norm)
				self.optimizer.step()

				value_loss_epoch += value_loss.item()
				action_loss_epoch += action_loss.item()
				dist_entropy_epoch += dist_entropy.item()

		num_updates = self.ppo_epoch * self.num_mini_batch

		value_loss_epoch /= num_updates
		action_loss_epoch /= num_updates
		dist_entropy_epoch /= num_updates

		return value_loss_epoch, action_loss_epoch, 0, dist_entropy_epoch
