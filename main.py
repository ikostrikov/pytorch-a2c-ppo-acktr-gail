import gym
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from envs import make_env
from model import ActorCritic
from torch.autograd import Variable
import argparse

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=7e-4,
                    help='learning rate (default: 7e-4)')
parser.add_argument('--eps', type=float, default=1e-5,
                    help='RMSprop optimizer epsilon (default: 1e-5)')
parser.add_argument('--alpha', type=float, default=0.99,
                    help='RMSprop optimizer apha (default: 0.99)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=16,
                    help='how many training CPU processes to use (default: 16)')
parser.add_argument('--num-steps', type=int, default=5,
                    help='number of forward steps in A2C (default: 5)')
parser.add_argument('--num-stack', type=int, default=4,
                    help='number of frames to stack (default: 4)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='log interval, one log per n updates (default: 10)')
parser.add_argument('--num-frames', type=int, default=10e6,
                    help='number of frames to train (default: 10e6)')
parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument('--log-dir', default='/tmp/gym/',
                    help='directory to save agent logs (default: /tmp/gym)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

try:
    os.makedirs(args.log_dir)
except OSError:
    pass

def main():
    print("#######")
    print("WARNING: All rewards are clipped so you need to use a monitor (see envs.py) to get true rewards")
    print("#######")

    os.environ['OMP_NUM_THREADS'] = '1'

    envs = SubprocVecEnv([make_env(args.env_name, args.seed, i, args.log_dir)
                            for i in range(args.num_processes)])

    actor_critic = ActorCritic(envs.observation_space.shape[0] * args.num_stack, envs.action_space)

    if args.cuda:
        actor_critic.cuda()

    optimizer = optim.RMSprop(actor_critic.parameters(), args.lr, eps=args.eps, alpha=args.alpha)
    #optimizer = KFACOptimizer(actor_critic, damping=1e-2, kl_clip=0.01, stat_decay=0.99)

    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0] * args.num_stack, obs_shape[1], obs_shape[2])

    states = torch.zeros(args.num_steps + 1, args.num_processes, *obs_shape)
    current_state = torch.zeros(args.num_processes, *obs_shape)
    counts = 0

    def update_current_state(state):
        state = torch.from_numpy(np.stack(state)).float()
        current_state[:, :-1] = current_state[:, 1:]
        current_state[:, -1] = state

    state = envs.reset()
    update_current_state(state)

    rewards = torch.zeros(args.num_steps, args.num_processes, 1)
    returns = torch.zeros(args.num_steps + 1, args.num_processes, 1)

    actions = torch.LongTensor(args.num_steps, args.num_processes)
    masks = torch.zeros(args.num_steps, args.num_processes, 1)

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([args.num_processes, 1])
    final_rewards = torch.zeros([args.num_processes, 1])

    if args.cuda:
        states = states.cuda()
        current_state = current_state.cuda()
        rewards = rewards.cuda()
        returns = returns.cuda()
        actions = actions.cuda()
        masks = masks.cuda()

    for j in range(num_updates):
        for step in range(args.num_steps):
            # Sample actions
            _, logits = actor_critic(Variable(states[step], volatile=True))
            probs = F.softmax(logits)
            log_probs = F.log_softmax(logits).data
            actions[step] = probs.multinomial().data

            cpu_actions = actions[step].cpu()
            cpu_actions = cpu_actions.numpy()

            # Obser reward and next state
            state, reward, done, info = envs.step(cpu_actions)

            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            episode_rewards += reward

            np_masks = np.array([0.0 if done_ else 1.0 for done_ in done])

            # If done then clean the history of observations.
            pt_masks = torch.from_numpy(np_masks.reshape(np_masks.shape[0], 1, 1, 1)).float()
            if args.cuda:
                pt_masks = pt_masks.cuda()
            current_state *= pt_masks

            update_current_state(state)
            states[step + 1].copy_(current_state)
            rewards[step].copy_(reward)
            masks[step].copy_(torch.from_numpy(np_masks).unsqueeze(1))

            final_rewards *= masks[step].cpu()
            final_rewards += (1 - masks[step].cpu()) * episode_rewards

            episode_rewards *= masks[step].cpu()

        # Reshape to do in a single forward pass for all steps
        values, logits = actor_critic(Variable(states.view(-1, *states.size()[-3:])))
        log_probs = F.log_softmax(logits)
        probs = F.softmax(logits)

        # Unreshape
        logits_size = (args.num_steps + 1, args.num_processes, logits.size(-1))

        log_probs = F.log_softmax(logits).view(logits_size)[:-1]
        probs = F.softmax(logits).view(logits_size)[:-1]

        values = values.view(args.num_steps + 1, args.num_processes, 1)
        logits = logits.view(logits_size)[:-1]

        action_log_probs = log_probs.gather(2, Variable(actions.unsqueeze(2)))

        dist_entropy = -(log_probs * probs).sum(-1).mean()

        returns[-1] = values[-1].data

        for step in reversed(range(args.num_steps)):
            returns[step] = returns[step + 1] * \
                args.gamma * masks[step] + rewards[step]

        value_loss = (values[:-1] - Variable(returns[:-1])).pow(2).mean()

        advantages = returns[:-1] - values[:-1].data
        action_loss = -(Variable(advantages) * action_log_probs).mean()

        optimizer.zero_grad()
        (value_loss * args.value_loss_coef + action_loss - dist_entropy * args.entropy_coef).backward()

        nn.utils.clip_grad_norm(actor_critic.parameters(), args.max_grad_norm)
        optimizer.step()

        states[0].copy_(states[-1])

        if j % args.log_interval == 0:
            print("Updates {}, num frames {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".format(
                j, j * args.num_processes * args.num_steps, final_rewards.mean(), final_rewards.median(), final_rewards.min(), final_rewards.max(), -dist_entropy.data[0], value_loss.data[0], action_loss.data[0]))

if __name__ == "__main__":
    main()
