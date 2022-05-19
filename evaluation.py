import numpy as np
import torch

from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs


def evaluate(
    actor_critic,
    obs_rms,
    env_name,
    seed,
    num_processes,
    eval_log_dir,
    device,
    num_frame_stack,
):
    eval_envs = make_vec_envs(
        env_name=env_name,
        seed=seed + 1000,
        num_processes=num_processes,
        gamma=None,
        log_dir=eval_log_dir,
        device=device,
        allow_early_resets=True,
        num_frame_stack=num_frame_stack,
    )

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device
    )
    eval_masks = torch.zeros(num_processes, 1, device=device)

    while len(eval_episode_rewards) < 10:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs, eval_recurrent_hidden_states, eval_masks, deterministic=True
            )

        # Obser reward and next obs
        obs, _, done, infos = eval_envs.step(action)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device,
        )

        for info in infos:
            if "episode" in info.keys():
                eval_episode_rewards.append(info["episode"]["r"])

    eval_envs.close()
    mean_eval_reward = np.mean(eval_episode_rewards)
    print(
        " Evaluation using {} episodes: mean reward {:.5f}\n".format(
            len(eval_episode_rewards), mean_eval_reward
        )
    )

    return mean_eval_reward
