import argparse
import os
import sys

import h5py
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(
        'Converts expert trajectories from h5 to pt format.')
    parser.add_argument(
        '--h5-file',
        default='trajs_halfcheetah.h5',
        help='input h5 file',
        type=str)
    parser.add_argument(
        '--pt-file',
        default=None,
        help='output pt file, by default replaces file extension with pt',
        type=str)
    args = parser.parse_args()

    if args.pt_file is None:
        args.pt_file = os.path.splitext(args.h5_file)[0] + '.pt'

    with h5py.File(args.h5_file, 'r') as f:
        dataset_size = f['obs_B_T_Do'].shape[0]  # full dataset size

        states = f['obs_B_T_Do'][:dataset_size, ...][...]
        actions = f['a_B_T_Da'][:dataset_size, ...][...]
        rewards = f['r_B_T'][:dataset_size, ...][...]
        lens = f['len_B'][:dataset_size, ...][...]

        states = torch.from_numpy(states).float()
        actions = torch.from_numpy(actions).float()
        rewards = torch.from_numpy(rewards).float()
        lens = torch.from_numpy(lens).long()

    data = {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'lengths': lens
    }

    torch.save(data, args.pt_file)


if __name__ == '__main__':
    main()
