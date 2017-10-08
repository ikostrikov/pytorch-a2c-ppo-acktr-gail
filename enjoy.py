import argparse
import os

import torch
from torch.autograd import Variable

from envs import make_env


parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--algo', default='a2c',
                    help='algorithm to use: a2c | ppo | acktr')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-stack', type=int, default=4,
                    help='number of frames to stack (default: 4)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='log interval, one log per n updates (default: 10)')
parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument('--load-dir', default='./trained_models/',
                    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument('--log-dir', default='/tmp/gym/',
                    help='directory to save agent logs (default: /tmp/gym)')
args = parser.parse_args()

try:
    os.makedirs(args.log_dir)
except OSError:
    pass

env = make_env(args.env_name, args.seed, 0, args.log_dir)()

actor_critic = torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))
actor_critic.eval()

obs_shape = env.observation_space.shape
obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])
current_state = torch.zeros(1, *obs_shape)


def update_current_state(state):
    shape_dim0 = env.observation_space.shape[0]
    state = torch.from_numpy(state).float()
    if args.num_stack > 1:
        current_state[:, :-shape_dim0] = current_state[:, shape_dim0:]
    current_state[:, -shape_dim0:] = state

env.render('human')
state = env.reset()
update_current_state(state)

if args.env_name.find('Bullet') > -1:
    import pybullet as p

    torsoId = -1
    for i in range(p.getNumBodies()):
        if (p.getBodyInfo(i)[0].decode() == "torso"):
            torsoId = i

while True:
    value, action = actor_critic.act(Variable(current_state, volatile=True),
                                        deterministic=True)
    cpu_actions = action.data.cpu().numpy()

    # Obser reward and next state
    state, _, done, _ = env.step(cpu_actions[0])

    if args.env_name.find('Bullet') > -1:
        if torsoId > -1:
            distance = 5
            yaw = 0
            humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
            p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

    env.render('human')

    if done:
        state = env.reset()
        actor_critic = torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))
        actor_critic.eval()

    update_current_state(state)
