import argparse
import os
# workaround to unpickle olf model files
import sys

import numpy as np
import torch
from tqdm import tqdm

from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize

sys.path.append('a2c_ppo_acktr')

parser = argparse.ArgumentParser(description='RL')
parser.add_argument(
    '--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    help='log interval, one log per n updates (default: 10)')
parser.add_argument(
    '--env-name',
    default='PongNoFrameskip-v4',
    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument(
    '--load-dir',
    default='./trained_models/',
    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument(
    '--non-det',
    action='store_true',
    default=False,
    help='whether to use a non-deterministic policy')
parser.add_argument(
    '--save-gail-expert',
    action='store_true',
    default=False,
    help='whether to save gail expert trajectory using loaded policy')
parser.add_argument(
    '--gail-expert-dir',
    default='./gail_experts/',
    help='directory to save gail expert trajectory (default: ./gail_experts/)')
parser.add_argument(
    '--gail-expert-traj-num',
    type=int,
    default=53,
    help='number of gail expert trajectories (default: 53)')
args = parser.parse_args()

args.det = not args.non_det

if args.env_name.find('Kuka') > -1 and not args.save_gail_expert:
    # NOTE: fix Kuka env not rendering bug
    env = make_vec_envs(
        args.env_name,
        args.seed + 1000,
        1,
        None,
        None,
        device='cpu',
        allow_early_resets=False,
        renders=True)
else:
    env = make_vec_envs(
        args.env_name,
        args.seed + 1000,
        1,
        None,
        None,
        device='cpu',
        allow_early_resets=False)

# Get a render function
# if args.save_gail_expert:
#     render_func = None  # no render
# else:
render_func = get_render_func(env)

# We need to use the same statistics for normalization as used in training
actor_critic, ob_rms = \
            torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))
actor_critic = actor_critic.cpu()

vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

recurrent_hidden_states = torch.zeros(1,
                                      actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

obs = env.reset()

if render_func is not None:
    render_func('human')

if args.env_name.find('Bullet') > -1:
    import pybullet as p

    torsoId = -1
    for i in range(p.getNumBodies()):
        if (p.getBodyInfo(i)[0].decode() == "torso"):
            torsoId = i

# generate expert trajectory using loaded policy, no rendering
if args.save_gail_expert:
    print('Generating gail expert trajectory')
    # init placeholders
    if env.observation_space.__class__.__name__ == 'Discrete':
        state_dim = env.observation_space.n
    else:
        state_dim = env.observation_space.shape[0]
    if env.action_space.__class__.__name__ == 'Discrete':
        action_dim = env.action_space.n
    else:
        action_dim = env.action_space.shape[0]

    traj_num = 0
    max_traj_num = args.gail_expert_traj_num

    states = []
    actions = []
    rewards = []
    lengths = []
    
    length = 0
    traj_states = []
    traj_actions = []
    traj_rewards = []

    # set up pbar
    pbar = tqdm(total=max_traj_num)

# evaluate loaded policy, with rendering
while True:
    if args.save_gail_expert:
        traj_states.append(obs)
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=args.det)

    # Obser reward and next obs
    obs, reward, done, _ = env.step(action)

    masks.fill_(0.0 if done else 1.0)

    if args.save_gail_expert:
        # traj_states.append(obs) # TODO
        traj_actions.append(action)
        traj_rewards.append(reward)
        length += 1

        if done:
            # store trajectory
            # states.append(torch.cat(traj_states))
            # actions.append(torch.cat(traj_actions))
            # rewards.append(torch.cat(traj_rewards))
            # lengths.append(length)
            states.append(torch.cat(traj_states).unsqueeze(0))
            actions.append(torch.cat(traj_actions).unsqueeze(0))
            rewards.append(torch.cat(traj_rewards).unsqueeze(0))
            lengths.append(length)

            # reset buffer
            length = 0
            traj_states = []
            traj_actions = []
            traj_rewards = []

            traj_num += 1
            pbar.update(1)
            obs = env.reset()
        
        if traj_num >= max_traj_num:
            break

    if args.env_name.find('Bullet') > -1:
        if torsoId > -1:
            distance = 5
            yaw = 0
            humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
            p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

    if render_func is not None:
        render_func('human')

if args.save_gail_expert:
    # convert to torch
    # states_tensor = torch.zeros(max_traj_num, max(lengths), state_dim)
    # actions_tensor = torch.zeros(max_traj_num, max(lengths), action_dim)
    # rewards_tensor = torch.zeros(max_traj_num, max(lengths))
    # lengths_tensor = torch.tensor(lengths, dtype=torch.int64)
    states_tensor = torch.cat(states)
    actions_tensor = torch.cat(actions)
    rewards_tensor = torch.cat(rewards)
    lengths_tensor = torch.tensor(lengths, dtype=torch.int64)

    # import ipdb; ipdb.set_trace()
    # for i in range(max_traj_num):
    #     states_tensor[i, :states[i].shape[0], :] = states[i]
    #     actions_tensor[i, :actions[i].shape[0], :] = actions[i]
    #     rewards_tensor[i, :rewards[i].shape[0]] = rewards[i].squeeze()

    # import ipdb; ipdb.set_trace()
    # save expert traj
    torch.save({'states': states_tensor, 'actions': actions_tensor, 'rewards': rewards_tensor, 'lengths': lengths_tensor},
                os.path.join(args.gail_expert_dir, 'trajs_' + args.env_name.split('-')[0].lower() + '.pt'))

    pbar.close()
    print('Expert trajectory saved!')

