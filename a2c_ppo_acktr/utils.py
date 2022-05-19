import os
import yaml
import torch
import argparse
import numpy as np
import torch.nn as nn

from a2c_ppo_acktr.envs import VecNormalize


def log_train_tb(
    writer,
    total_num_steps,
    start,
    end,
    episode_rewards,
    dist_entropy,
    value_loss,
    action_loss,
    use_print,
):
    num_rewards = len(episode_rewards)
    mean_rewards = np.mean(episode_rewards)
    median_rewards = np.median(episode_rewards)
    min_rewards = np.min(episode_rewards)
    max_rewards = np.max(episode_rewards)
    writer.add_scalar(
        tag="FPS",
        scalar_value=int(total_num_steps / (end - start)),
        global_step=total_num_steps,
    )
    writer.add_scalar(
        tag="Mean Reward Of Last 10 Episode Rewards",
        scalar_value=mean_rewards,
        global_step=total_num_steps,
    )
    writer.add_scalar(
        tag="Median Reward Of Last 10 Episode Rewards",
        scalar_value=median_rewards,
        global_step=total_num_steps,
    )
    writer.add_scalar(
        tag="Min Reward Of Last 10 Episode Rewards",
        scalar_value=min_rewards,
        global_step=total_num_steps,
    )
    writer.add_scalar(
        tag="Max Reward Of Last 10 Episode Rewards",
        scalar_value=max_rewards,
        global_step=total_num_steps,
    )
    writer.add_scalar(
        tag="Distribution Entropy At Num Step",
        scalar_value=dist_entropy,
        global_step=total_num_steps,
    )
    writer.add_scalar(
        tag="Value Loss At Num Step",
        scalar_value=value_loss,
        global_step=total_num_steps,
    )
    writer.add_scalar(
        tag="Action Loss At Num Step",
        scalar_value=action_loss,
        global_step=total_num_steps,
    )

    if use_print:
        print(
            "Num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".format(
                total_num_steps,
                int(total_num_steps / (end - start)),
                num_rewards,
                mean_rewards,
                median_rewards,
                min_rewards,
                max_rewards,
                dist_entropy,
                value_loss,
                action_loss,
            )
        )


def save_model(save_path, agent, epoch, is_best: bool = False):
    os.makedirs(save_path, exist_ok=True)

    file_name = "checkpoint.pt"

    if is_best:
        file_name = "best_" + file_name

    torch.save(
        {"epoch": epoch, "model_state_dict": agent.actor_critic.state_dict()},
        os.path.join(save_path, file_name),
    )


def get_config():
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="./configs/clean.yml",
        help="Configuration file to use",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)

    if not cfg["id"]:
        raise ValueError('"id" should not be none in config yml')

    return cfg


# Get a render function
def get_render_func(venv):
    if hasattr(venv, "envs"):
        return venv.envs[0].render
    elif hasattr(venv, "venv"):
        return get_render_func(venv.venv)
    elif hasattr(venv, "env"):
        return get_render_func(venv.env)

    return None


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, "venv"):
        return get_vec_normalize(venv.venv)

    return None


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
