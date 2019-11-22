import argparse
import os
# workaround to unpickle olf model files
import sys

import numpy as np
import torch

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
args = parser.parse_args()

args.det = not args.non_det

env = make_vec_envs(
    args.env_name,
    args.seed + 1000,
    1,
    None,
    None,
    device='cpu',
    allow_early_resets=False)

# Get a render function
if args.save_gail_expert:
    render_func = None  # no render
else:
    render_func = get_render_func(env)

# We need to use the same statistics for normalization as used in training
actor_critic, ob_rms = \
            torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))

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
    # TODO
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
    max_traj_num = 53

    states = []
    actions = []
    rewards = []
    lengths = []
    
    length = 0
    traj_states = []
    traj_actions = []
    traj_rewards = []

    while traj_num < max_traj_num:
        traj_states.append(obs)
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = actor_critic.act(
                obs, recurrent_hidden_states, masks, deterministic=args.det)

        # Obser reward and next obs
        obs, reward, done, _ = env.step(action)

        masks.fill_(0.0 if done else 1.0)

        traj_actions.append(action)
        traj_rewards.append(reward)
        length += 1

        if done:
            # store trajectory
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

    # convert to torch
    states = torch.cat(states)
    actions = torch.cat(actions)
    rewards = torch.cat(rewards)
    lengths = torch.tensor(lengths, dtype=torch.int64)

    # save expert traj
    torch.save({'states': states, 'actions': actions, 'rewards': rewards, 'lengths': lengths},
               os.path.join(args.gail_expert_dir, 'trajs_' + args.env_name.split('-')[0].lower() + '.pt'))

# evaluate loaded policy, with rendering
else:
    while True:
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = actor_critic.act(
                obs, recurrent_hidden_states, masks, deterministic=args.det)

        # Obser reward and next obs
        obs, reward, done, _ = env.step(action)

        masks.fill_(0.0 if done else 1.0)

        if args.env_name.find('Bullet') > -1:
            if torsoId > -1:
                distance = 5
                yaw = 0
                humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
                p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

        if render_func is not None:
            render_func('human')
