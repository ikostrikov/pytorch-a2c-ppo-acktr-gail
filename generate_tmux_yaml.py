import argparse

import yaml

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument(
    '--num-seeds',
    type=int,
    default=4,
    help='number of random seeds to generate')
parser.add_argument(
    '--env-names',
    default="PongNoFrameskip-v4",
    help='environment name separated by semicolons')
args = parser.parse_args()

ppo_mujoco_template = "python main.py --env-name {0} --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --tau 0.95 --num-env-steps 1000000 --use-linear-lr-decay --no-cuda --log-dir /tmp/gym/{1}/{1}-{2} --seed {2} --use-proper-time-limits"

ppo_atari_template = "env CUDA_VISIBLE_DEVICES={2} python main.py --env-name {0} --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01 --log-dir /tmp/gym/{1}/{1}-{2} --seed {2}"

template = ppo_atari_template

config = {"session_name": "run-all", "windows": []}

for i in range(args.num_seeds):
    panes_list = []
    for env_name in args.env_names.split(';'):
        panes_list.append(
            template.format(env_name,
                            env_name.split('-')[0].lower(), i))

    config["windows"].append({
        "window_name": "seed-{}".format(i),
        "panes": panes_list
    })

yaml.dump(config, open("run_all.yaml", "w"), default_flow_style=False)
