import glob
import os

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from arguments import get_args
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from envs import make_env
from kfac import KFACOptimizer
from model import ActorCritic
from storage import RolloutStorage
from vizualize_atari import visdom_plot

args = get_args()

assert args.algo in ['a2c', 'ppo', 'acktr']
if args.algo == 'ppo':
    assert args.num_processes * args.num_steps % args.batch_size == 0

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

try:
    os.makedirs(args.log_dir)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.json'))
    for f in files:
        os.remove(f)


def main():
    print("#######")
    print("WARNING: All rewards are clipped so you need to use a monitor (see envs.py) or visdom plot to get true rewards")
    print("#######")

    os.environ['OMP_NUM_THREADS'] = '1'

    if args.vis:
        from visdom import Visdom
        viz = Visdom()
        win = None

    envs = SubprocVecEnv([
        make_env(args.env_name, args.seed, i, args.log_dir)
        for i in range(args.num_processes)
    ])

    actor_critic = ActorCritic(envs.observation_space.shape[0] * args.num_stack, envs.action_space)
    if args.algo == 'ppo':
        actor_critic = nn.DataParallel(actor_critic)

    if args.cuda:
        actor_critic.cuda()

    if args.algo == 'a2c':
        optimizer = optim.RMSprop(actor_critic.parameters(), args.lr, eps=args.eps, alpha=args.alpha)
    elif args.algo == 'ppo':
        optimizer = optim.Adam(actor_critic.parameters(), args.lr, eps=args.eps)
    elif args.algo == 'acktr':
        optimizer = KFACOptimizer(actor_critic)

    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

    rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, envs.action_space.n)

    current_state = torch.zeros(args.num_processes, *obs_shape)
    def update_current_state(state):
        state = torch.from_numpy(np.stack(state)).float()
        current_state[:, :-1] = current_state[:, 1:]
        current_state[:, -1] = state

    state = envs.reset()
    update_current_state(state)

    rollouts.states[0].copy_(current_state)

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([args.num_processes, 1])
    final_rewards = torch.zeros([args.num_processes, 1])

    if args.cuda:
        current_state = current_state.cuda()
        rollouts.cuda()

    for j in range(num_updates):
        for step in range(args.num_steps):
            # Sample actions
            value, action, action_log_probs = actor_critic.act(Variable(rollouts.states[step], volatile=True))
            cpu_actions = action.data.cpu().numpy()

            # Obser reward and next state
            state, reward, done, info = envs.step(cpu_actions)

            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            episode_rewards += reward

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            if args.cuda:
                masks = masks.cuda()
            current_state *= masks.unsqueeze(2).unsqueeze(2)

            update_current_state(state)
            rollouts.insert(step, current_state, action.data, value.data, action_log_probs.data, reward, masks)

        next_value = actor_critic(Variable(rollouts.states[-1], volatile=True))[0].data

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        if args.algo in ['a2c', 'acktr']:
            # Reshape to do in a single forward pass for all steps
            values, logits = actor_critic(Variable(rollouts.states[:-1].view(-1, *rollouts.states.size()[-3:])))
            log_probs = F.log_softmax(logits)

            # Unreshape
            logits_size = (args.num_steps, args.num_processes, logits.size(-1))

            log_probs = F.log_softmax(logits).view(logits_size)
            probs = F.softmax(logits).view(logits_size)

            values = values.view(args.num_steps, args.num_processes, 1)
            logits = logits.view(logits_size)

            action_log_probs = log_probs.gather(2, Variable(rollouts.actions))

            dist_entropy = -(log_probs * probs).sum(-1).mean()

            advantages = Variable(rollouts.returns[:-1]) - values
            value_loss = advantages.pow(2).mean()

            action_loss = -(Variable(advantages.data) * action_log_probs).mean()

            if args.algo == 'acktr' and optimizer.steps % optimizer.Ts == 0:
                # Sampled fisher, see Martens 2014
                actor_critic.zero_grad()
                pg_fisher_loss = -action_log_probs.mean()

                value_noise = Variable(torch.randn(values[:-1].size()))
                if args.cuda:
                    value_noise = value_noise.cuda()

                sample_values = values[:-1] + value_noise
                vf_fisher_loss = - (values[:-1] - Variable(sample_values.data)).pow(2).mean()

                fisher_loss = pg_fisher_loss + vf_fisher_loss
                optimizer.acc_stats = True
                fisher_loss.backward(retain_graph=True)
                optimizer.acc_stats = False

            optimizer.zero_grad()
            (value_loss * args.value_loss_coef + action_loss - dist_entropy * args.entropy_coef).backward()

            if args.algo == 'a2c':
                nn.utils.clip_grad_norm(actor_critic.parameters(), args.max_grad_norm)

            optimizer.step()
        elif args.algo == 'ppo':
            advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
            advantages = (advantages - advantages.mean()) / advantages.std()
            for _ in range(args.ppo_epoch):
                sampler = BatchSampler(SubsetRandomSampler(range(args.num_processes * args.num_steps)), args.batch_size * args.num_processes, drop_last=False)
                for indices in sampler:
                    states_batch = rollouts.states[:-1].view(-1, *rollouts.states.size()[-3:])[indices]
                    actions_batch = rollouts.actions.view(-1, 1)[indices]
                    return_batch = rollouts.returns[:-1].view(-1, 1)[indices]

                    # Reshape to do in a single forward pass for all steps
                    values, logits = actor_critic(Variable(states_batch))
                    log_probs = F.log_softmax(logits)
                    action_log_probs = log_probs.gather(1, Variable(actions_batch))

                    old_action_log_probs = rollouts.action_log_probs.view(-1, rollouts.action_log_probs.size(-1))[indices]

                    ratio = torch.exp(action_log_probs - Variable(old_action_log_probs))
                    adv_targ = Variable(advantages.view(-1, 1)[indices])
                    surr1 = ratio * adv_targ
                    surr2 = ratio.clamp(1.0 - args.clip_param, 1.0 + args.clip_param) * adv_targ
                    action_loss = -torch.min(surr1, surr2).mean() # PPO's pessimistic surrogate (L^CLIP)

                    probs = F.softmax(logits)

                    dist_entropy = -(log_probs * probs).sum(-1).mean()

                    value_loss = (Variable(return_batch) - values).pow(2).mean()

                    optimizer.zero_grad()
                    (value_loss + action_loss - dist_entropy * args.entropy_coef).backward()
                    optimizer.step()

        rollouts.states[0].copy_(rollouts.states[-1])

        if j % args.log_interval == 0:
            print("Updates {}, num frames {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
                format(j, j * args.num_processes * args.num_steps,
                       final_rewards.mean(),
                       final_rewards.median(),
                       final_rewards.min(),
                       final_rewards.max(), -dist_entropy.data[0],
                       value_loss.data[0], action_loss.data[0]))

        if j % args.vis_interval == 0:
            win = visdom_plot(viz, win, args.log_dir, args.env_name, args.algo)


if __name__ == "__main__":
    main()
