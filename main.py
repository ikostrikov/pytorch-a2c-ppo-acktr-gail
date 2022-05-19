import os
import time
from pytz import timezone
from collections import deque
from datetime import datetime

import numpy as np
import torch

from a2c_ppo_acktr import algo, utils

# from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from torch.utils.tensorboard import SummaryWriter
from evaluation import evaluate


def main(cfg: dict):
    id = cfg["id"]
    seed = cfg["seed"]
    gamma = cfg["gamma"]
    env_name = cfg["env_name"]
    algo_name = cfg["algorithm"]
    num_processes = cfg["num_processes"]
    num_steps = cfg["train"]["num_steps"]
    cfg_algo = cfg["train"]["algorithm_params"]
    save_path = os.path.join("./checkpoints", algo_name, id, str(seed))

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
    #     torch.backends.cudnn.benchmark = False
    #     torch.backends.cudnn.deterministic = True

    now_str = datetime.now(timezone("EST")).strftime("%d_%m_%Y/%H:%M:%S")
    log_dir = f"./logs/{env_name}/{id}/{seed}/{now_str}"
    eval_log_dir = log_dir + "/eval"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(eval_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    envs = make_vec_envs(
        env_name=env_name,
        seed=seed,
        num_processes=num_processes,
        gamma=gamma,
        log_dir=log_dir,
        device=device,
        num_frame_stack=cfg["num_frame_stack"],
        allow_early_resets=False,
    )

    actor_critic = Policy(
        obs_shape=envs.observation_space.shape,
        action_space=envs.action_space,
        base_kwargs={"recurrent": cfg["recurrent_policy"]},
    )
    actor_critic.to(device)

    # if args.algo == "a2c":
    #     agent = algo.A2C_ACKTR(
    #         actor_critic,
    #         args.value_loss_coef,
    #         args.entropy_coef,
    #         lr=args.lr,
    #         eps=args.eps,
    #         alpha=args.alpha,
    #         max_grad_norm=args.max_grad_norm,
    #     )
    if algo_name == "ppo":
        agent = algo.PPO(
            actor_critic=actor_critic,
            **cfg_algo
            # actor_critic,
            # args.clip_param,
            # args.ppo_epoch,
            # args.num_mini_batch,
            # args.value_loss_coef,
            # args.entropy_coef,
            # lr=args.lr,
            # eps=args.eps,
            # max_grad_norm=args.max_grad_norm,
        )
    # elif args.algo == "acktr":
    #     agent = algo.A2C_ACKTR(
    #         actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True
    #     )
    else:
        raise NotImplementedError(f"The algorithm:{algo_name} is not implemented")

    # TODO: Gail Stuff
    # if args.gail:
    #     assert len(envs.observation_space.shape) == 1
    #     discr = gail.Discriminator(
    #         envs.observation_space.shape[0] + envs.action_space.shape[0], 100, device
    #     )
    #     file_name = os.path.join(
    #         args.gail_experts_dir,
    #         "trajs_{}.pt".format(env_name.split("-")[0].lower()),
    #     )

    #     expert_dataset = gail.ExpertDataset(
    #         file_name, num_trajectories=4, subsample_frequency=20
    #     )
    #     drop_last = len(expert_dataset) > args.gail_batch_size
    #     gail_train_loader = torch.utils.data.DataLoader(
    #         dataset=expert_dataset,
    #         batch_size=args.gail_batch_size,
    #         shuffle=True,
    #         drop_last=drop_last,
    #     )

    rollouts = RolloutStorage(
        num_steps=num_steps,
        num_processes=num_processes,
        obs_shape=envs.observation_space.shape,
        action_space=envs.action_space,
        recurrent_hidden_state_size=actor_critic.recurrent_hidden_state_size,
    )

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=cfg['train']['episode_rewards_queue_len'])

    start = time.time()
    top_mean_train_reward = float("-inf")
    num_updates = int(cfg["train"]["num_env_steps"]) // num_steps // num_processes
    for j in range(num_updates):

        if cfg["train"]["use_linear_lr_decay"]:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                optimizer=agent.optimizer,
                epoch=j,
                total_num_epochs=num_updates,
                initial_lr=agent.optimizer.lr
                if algo_name == "acktr"
                else cfg_algo["lr"],
            )

        for step in range(num_steps):
            # Sample actions
            with torch.no_grad():
                (
                    value,
                    action,
                    action_log_prob,
                    recurrent_hidden_states,
                ) = actor_critic.act(
                    inputs=rollouts.obs[step],
                    rnn_hxs=rollouts.recurrent_hidden_states[step],
                    masks=rollouts.masks[step],
                )

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(actions=action)

            for info in infos:
                if "episode" in info.keys():
                    episode_rewards.append(info["episode"]["r"])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if "bad_transition" in info.keys() else [1.0] for info in infos]
            )
            rollouts.insert(
                obs=obs,
                recurrent_hidden_states=recurrent_hidden_states,
                actions=action,
                action_log_probs=action_log_prob,
                value_preds=value,
                rewards=reward,
                masks=masks,
                bad_masks=bad_masks,
            )

        with torch.no_grad():
            next_value = actor_critic.get_value(
                inputs=rollouts.obs[-1],
                rnn_hxs=rollouts.recurrent_hidden_states[-1],
                masks=rollouts.masks[-1],
            ).detach()

        # TODO: Gail Stuff
        # if args.gail:
        #     if j >= 10:
        #         envs.venv.eval()

        #     gail_epoch = args.gail_epoch
        #     if j < 10:
        #         gail_epoch = 100  # Warm up
        #     for _ in range(gail_epoch):
        #         discr.update(
        #             gail_train_loader, rollouts, utils.get_vec_normalize(envs)._obfilt
        #         )

        #     for step in range(args.num_steps):
        #         rollouts.rewards[step] = discr.predict_reward(
        #             rollouts.obs[step],
        #             rollouts.actions[step],
        #             gamma,
        #             rollouts.masks[step],
        #         )

        rollouts.compute_returns(
            next_value=next_value, gamma=gamma, **cfg["train"]["compute_returns"]
        )

        value_loss, action_loss, dist_entropy = agent.update(rollouts=rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if j % cfg["train"]["save_interval"] == 0 or j == num_updates - 1:
            os.makedirs(save_path, exist_ok=True)
            utils.save_model(save_path=save_path, agent=agent, epoch=j, is_best=False)

            mean_train_reward = np.mean(episode_rewards)
            if top_mean_train_reward < mean_train_reward:
                top_mean_train_reward = mean_train_reward
                utils.save_model(
                    save_path=save_path, agent=agent, epoch=j, is_best=True
                )

        if (
            j % cfg["log_interval"] == 0
            and len(episode_rewards) > 1
            or j == num_updates - 1
        ):
            total_num_steps = (j + 1) * num_processes * num_steps
            end = time.time()
            utils.log_train_tb(
                writer=writer,
                total_num_steps=total_num_steps,
                start=start,
                end=end,
                episode_rewards=episode_rewards,
                dist_entropy=dist_entropy,
                value_loss=value_loss,
                action_loss=action_loss,
                use_print=True,
            )

        if (
            cfg["eval_interval"] is not None
            and len(episode_rewards) > 1
            and j % cfg["eval_interval"] == 0
            or j == num_updates - 1
        ):
            total_num_steps = (j + 1) * num_processes * num_steps
            obs_rms = utils.get_vec_normalize(envs).obs_rms
            mean_eval_reward = evaluate(
                actor_critic=actor_critic,
                obs_rms=obs_rms,
                env_name=env_name,
                seed=seed,
                num_processes=num_processes,
                eval_log_dir=eval_log_dir,
                device=device,
                num_frame_stack=cfg["num_frame_stack"],
            )
            writer.add_scalar(
                tag="Mean Eval Reward",
                scalar_value=mean_eval_reward,
                global_step=total_num_steps,
            )


if __name__ == "__main__":
    cfg = utils.get_config()

    if cfg["device_id"] is not None:
        with torch.cuda.device(cfg["device_id"]):
            main(cfg)
    else:
        main(cfg)
