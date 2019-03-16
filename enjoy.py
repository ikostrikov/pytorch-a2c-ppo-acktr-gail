import sys
path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if path in sys.path:
    sys.path.remove(path)

import argparse
import os

import numpy as np
import torch

from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize

from utils import LoggerCsv
# workaround to unpickle olf model files
import sys
sys.path.append('a2c_ppo_acktr')

parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='log interval, one log per n updates (default: 10)')
parser.add_argument('--env-name', default='CellrobotEnvCPG4-v0',
                    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument('--load-dir', default='./trained_models/',
                    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument('--load-file-dir', default='log-files/Mar_16_PPO_RL_Exp7/No_1_CellrobotEnvCPG4-v0_PPO-2019-03-16_11:18:50/model/ppo/CellrobotEnvCPG4-v0_0.pt'   )
parser.add_argument('--result-dir', default=None   )
parser.add_argument('--num-enjoy',type=int, default=1   )
parser.add_argument('--add-timestep', action='store_true', default=False,
                    help='add timestep to observations')
parser.add_argument('--non-det', action='store_true', default=False,
                    help='whether to use a non-deterministic policy')
args = parser.parse_args()
num_enjoy = 1

if args.result_dir is None:
    result_dir = 'tmp'
    os.makedirs(result_dir,exist_ok=True)
else:
    result_dir = args.result_dir
logger = LoggerCsv(result_dir, csvname='log_data')


args.det = not args.non_det

env = make_vec_envs(args.env_name, args.seed + 1000, 1,
                            None, None,  device='cpu',
                            allow_early_resets=False)

# Get a render function
render_func = get_render_func(env)

# We need to use the same statistics for normalization as used in training
if args.load_file_dir is not None:
    model_path = args.load_file_dir
else:
    model_path = os.path.join(args.load_dir, args.env_name + ".pt")

actor_critic, ob_rms =   torch.load(model_path)

vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

if render_func is not None:
    render_func('human')

obs = env.reset()

if args.env_name.find('Bullet') > -1:
    import pybullet as p

    torsoId = -1
    for i in range(p.getNumBodies()):
        if (p.getBodyInfo(i)[0].decode() == "torso"):
            torsoId = i

velocity_base_lists = []
command_lists = []
reward_lists = []
num_episodes_lists =[]
obs_lists = []
num_episodes = 0

while True:
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=args.det)

    # Obser reward and next obs
    obs, reward, done, log_info = env.step(action)


    masks.fill_(0.0 if done else 1.0)

    if done:
        num_episodes += 1
    if logger is not None:
        velocity_base_lists.append(log_info[0]['velocity_base'])
        command_lists.append(log_info[0]['commands'])
        reward_lists.append(log_info[0]['rewards'])
        num_episodes_lists.append(num_episodes)
        obs_lists.append(log_info[0]['obs'])
    if num_episodes == num_enjoy:
        break

    if args.env_name.find('Bullet') > -1:
        if torsoId > -1:
            distance = 5
            yaw = 0
            humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
            p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

    if render_func is not None:
        render_func('human')

if logger is not None:
    velocity_base = np.array(velocity_base_lists, dtype=np.float64)
    commands = np.array(command_lists, dtype=np.float64)
    rewards = np.array(reward_lists, dtype=np.float64)
    num_episodes_lists = np.array(num_episodes_lists, dtype=np.float64).reshape((-1,1))
    obs_lists = np.array(obs_lists, dtype=np.float64)

    data = np.concatenate((num_episodes_lists , velocity_base, commands,  obs_lists, rewards  ), axis=1)

    trajectory = {}
    for j in range(data.shape[0]):
        for i in range(data.shape[1]):
            trajectory[i] = data[j][i]
        logger.log(trajectory)
        logger.write(display=False)
