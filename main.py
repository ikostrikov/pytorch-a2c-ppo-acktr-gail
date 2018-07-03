import copy
import glob
import os
import time

import torch
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize

import algo
from arguments import get_args
from envs import make_env
from storage import RolloutStorage
from visualize import visdom_plot

args = get_args()

assert args.algo in ['a2c', 'a2oc', 'ppo', 'acktr']
if args.recurrent_policy:
	assert args.algo in ['a2c', 'ppo'], \
		'Recurrent policy is not implemented for ACKTR and A2OC'

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)

log_dir = os.path.join(args.log_dir_base_path, args.env_name, args.algo)

try:
	os.makedirs(log_dir)
except OSError:
	files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
	for f in files:
		os.remove(f)

def main():
	print("#######")
	print(
			"WARNING: All rewards are clipped or normalized so you need to use a monitor (see envs.py) or visdom plot to get true rewards")
	print("#######")

	torch.set_num_threads(1)

	if args.vis:
		from visdom import Visdom
		viz = Visdom(port=args.port)
		win = None

	envs = [make_env(args.env_name, args.seed, i, log_dir, args.add_timestep)
	        for i in range(args.num_processes)]

	if args.num_processes > 1:
		envs = SubprocVecEnv(envs)
	else:
		envs = DummyVecEnv(envs)

	if len(envs.observation_space.shape) == 1:
		envs = VecNormalize(envs)

	obs_shape = envs.observation_space.shape
	obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

	if args.algo == 'a2c':
		agent = algo.A2C_ACKTR(obs_shape, envs.action_space, args.recurrent_policy, args.value_loss_coef,
		                       args.entropy_coef, envs, lr=args.lr,
		                       eps=args.eps, alpha=args.alpha,
		                       max_grad_norm=args.max_grad_norm, cuda=args.cuda)

	elif args.algo == 'a2oc':
		agent = algo.A2OC(envs, args)

	elif args.algo == 'ppo':
		agent = algo.PPO(obs_shape, envs.action_space, args.recurrent_policy, args.clip_param, args.ppo_epoch, args.num_mini_batch,
		                 args.value_loss_coef, args.entropy_coef, envs, lr=args.lr,
		                 eps=args.eps,
		                 max_grad_norm=args.max_grad_norm, cuda=args.cuda)

	elif args.algo == 'acktr':
		agent = algo.A2C_ACKTR(obs_shape, envs.action_space, args.recurrent_policy, args.value_loss_coef,
		                       args.entropy_coef, acktr=True)

	else:
		raise ValueError('args.algo does not match any expected algorithm')

	rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, agent.envs.action_space, agent.state_size)
	current_obs = torch.zeros(args.num_processes, *obs_shape)

	def update_current_obs(obs):
		shape_dim0 = agent.envs.observation_space.shape[0]
		obs = torch.from_numpy(obs).float()
		if args.num_stack > 1:
			current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
		current_obs[:, -shape_dim0:] = obs

	obs = agent.envs.reset()
	update_current_obs(obs)

	rollouts.observations[0].copy_(current_obs)

	# These variables are used to compute average rewards for all processes.
	episode_rewards = torch.zeros([args.num_processes, 1])
	final_rewards = torch.zeros([args.num_processes, 1])

	if args.cuda:
		current_obs = current_obs.cuda()
		rollouts.cuda()

	start = time.time()
	for j in range(num_updates):
		for step in range(args.num_steps):
			# Sample actions
			with torch.no_grad():
				value, action, action_log_prob, states, obs, reward, done, _ = agent.act(
						rollouts.observations[step],
						rollouts.states[step],
						rollouts.masks[step])

			# cpu_actions = action.squeeze(1).cpu().numpy()
			#
			# # Observe reward and next obs
			# obs, reward, done, info = envs.step(cpu_actions)
			# reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()

			episode_rewards += reward

			# If done then clean the history of observations.
			masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
			final_rewards *= masks
			final_rewards += (1 - masks) * episode_rewards
			episode_rewards *= masks

			if args.cuda:
				masks = masks.cuda()

			if current_obs.dim() == 4:
				current_obs *= masks.unsqueeze(2).unsqueeze(2)
			else:
				current_obs *= masks

			update_current_obs(obs)
			rollouts.insert(current_obs, states, action, action_log_prob, value, reward, masks)

		with torch.no_grad():
			next_value = agent.get_value(rollouts.observations[-1],
			                             rollouts.states[-1],
			                             rollouts.masks[-1]).detach()

		rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

		value_loss, action_loss, termination_loss, dist_entropy = agent.update(rollouts)

		rollouts.after_update()

		if j % args.save_interval == 0 and args.save_dir != "":
			save_path = os.path.join(args.save_dir, args.algo)
			try:
				os.makedirs(save_path)
			except OSError:
				pass

			# A really ugly way to save a model to CPU
			save_model = agent.actor_critic
			if args.cuda:
				save_model = copy.deepcopy(agent.actor_critic).cpu()

			save_model = [save_model,
			              hasattr(agent.envs, 'ob_rms') and agent.envs.ob_rms or None]

			torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

		if j % args.log_interval == 0:
			end = time.time()
			total_num_steps = (j + 1) * args.num_processes * args.num_steps
			print(
				"Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}, termination_loss: {:.5f}".
					format(j, total_num_steps,
				           int(total_num_steps / (end - start)),
				           final_rewards.mean(),
				           final_rewards.median(),
				           final_rewards.min(),
				           final_rewards.max(), dist_entropy,
				           value_loss, action_loss, termination_loss))
		if args.vis and j % args.vis_interval == 0:
			try:
				# Sometimes monitor doesn't properly flush the outputs
				win = visdom_plot(viz, win, log_dir, args.env_name,
				                  args.algo, args.num_frames)
			except IOError:
				pass


if __name__ == "__main__":
	main()
