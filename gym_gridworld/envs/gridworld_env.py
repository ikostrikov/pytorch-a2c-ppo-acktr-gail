import gym
import sys
import os
import copy
from gym import spaces
from gym.utils import seeding
import numpy as np

EMPTY = BLACK = 0
WALL = GRAY = 1
MINE = RED = 2
TARGET = GREEN = 3
AGENT = BLUE = 4
SUCCESS = PINK = 6
YELLOW = 7
COLORS = {BLACK: [0.0, 0.0, 0.0], GRAY: [0.5, 0.5, 0.5],
          BLUE: [0.0, 0.0, 1.0], GREEN: [0.0, 1.0, 0.0],
          RED: [1.0, 0.0, 0.0], PINK: [1.0, 0.0, 1.0],
          YELLOW: [1.0, 1.0, 0.0]}

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3


class GridworldEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    num_env = 0

    def __init__(self, plan, stochastic=False):

        self.stochastic = stochastic
        self.actions = [UP, DOWN, LEFT, RIGHT]
        self.inv_actions = [0, 1, 2, 3]
        self.action_space = spaces.Discrete(4)
        self.action_pos_dict = {UP: [-1, 0], DOWN: [1, 0], LEFT: [0, -1], RIGHT: [0, 1]}
        self.img_shape = [256, 256, 3]  # observation space shape

        # initialize system state
        this_file_path = os.path.dirname(os.path.realpath(__file__))
        grid_map_path = os.path.join(this_file_path, 'plan{}{}.txt'.format(plan, 's' if stochastic else ''))
        self.start_grid_map = self._read_grid_map(grid_map_path)  # initial grid map
        self.current_grid_map = copy.deepcopy(self.start_grid_map)  # current grid map
        if stochastic:
            self._init_stochastic()
        self.grid_map_shape = (1, ) + self.start_grid_map.shape
        self.observation_space = spaces.Box(low=0, high=6, shape=self.grid_map_shape)

        # agent state: start, target, current state
        self.agent_start_state, self.agent_target_state = self._get_agent_start_and_target_state()

        # set other parameters
        self.restart_once_done = False  # restart or not once done

        GridworldEnv.num_env += 1
        self.this_fig_num = GridworldEnv.num_env
        self.viewer = None

    def _seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        return seed1

    def _step(self, action):
        """ return next observation, reward, finished, success """
        action = int(action)
        info = {'success': True}
        done = False
        reward = 0.0
        nxt_agent_state = (self.agent_state[0] + self.action_pos_dict[action][0],
                           self.agent_state[1] + self.action_pos_dict[action][1])

        next_state_out_of_map = (nxt_agent_state[0] < 0 or nxt_agent_state[0] >= self.grid_map_shape[1]) or \
                                (nxt_agent_state[1] < 0 or nxt_agent_state[1] >= self.grid_map_shape[2])
        if next_state_out_of_map:
            info['success'] = False
            return self.current_grid_map, reward, False, info

        # successful behavior
        target_position = self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]]

        if target_position == EMPTY:
            self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]] = AGENT
        elif target_position == WALL:
            info['success'] = False
            reward = -0.1
            return self.current_grid_map, reward, False, info
        elif target_position == TARGET:
            done = True
            reward = 1
            # self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]] = SUCCESS
        elif target_position == MINE:
            done = True
            reward = -1
            # self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]] = MINE

        # if done and self.restart_once_done:
        #     self._reset()
        #     return self.current_grid_map, reward, done, info

        self.current_grid_map[self.agent_state[0], self.agent_state[1]] = EMPTY
        self.agent_state = copy.deepcopy(nxt_agent_state)

        return self.current_grid_map, reward, done, info

    def _reset(self):
        self.current_grid_map = copy.deepcopy(self.start_grid_map)
        if self.stochastic:
            self._init_stochastic()
        self.agent_state, self.agent_target_state = self._get_agent_start_and_target_state()
        return self.current_grid_map

    def _init_stochastic(self):
        self.select_stochastic(self.current_grid_map, TARGET)
        self.select_stochastic(self.current_grid_map, AGENT)
        if MINE in self.current_grid_map:
            self.select_stochastic(self.current_grid_map, MINE)

    def _read_grid_map(self, grid_map_path):
        grid_map = open(grid_map_path, 'r').readlines()
        grid_map_array = []
        for k1 in grid_map:
            k1s = k1.split(' ')
            tmp_arr = []
            for k2 in k1s:
                try:
                    tmp_arr.append(int(k2))
                except:
                    pass
            grid_map_array.append(tmp_arr)
        grid_map_array = np.array(grid_map_array, dtype=int)
        return grid_map_array

    def select_stochastic(self, grid_map_array, type):
        target_idxs = np.nonzero(grid_map_array == type)
        selected_target = np.random.randint(len(target_idxs[0]))
        new_target_idxs = [[], []]
        new_target_idxs[0] = np.delete(target_idxs[0], selected_target)
        new_target_idxs[1] = np.delete(target_idxs[1], selected_target)
        grid_map_array[new_target_idxs] = EMPTY

    def _get_agent_start_and_target_state(self):
        start_state = np.where(self.current_grid_map == AGENT)
        target_state = np.where(self.current_grid_map == TARGET)

        start_or_target_not_found = not(start_state[0] and target_state[0])
        if start_or_target_not_found:
            sys.exit('Start or target state not specified')
        start_state = (start_state[0][0], start_state[1][0])
        target_state = (target_state[0][0], target_state[1][0])

        return start_state, target_state

    def _gridmap_to_image(self, img_shape=None):
        if img_shape is None:
            img_shape = self.img_shape
        observation = np.random.randn(*img_shape) * 0.0
        gs0 = int(observation.shape[0] / self.current_grid_map.shape[0])
        gs1 = int(observation.shape[1] / self.current_grid_map.shape[1])
        for i in range(self.current_grid_map.shape[0]):
            for j in range(self.current_grid_map.shape[1]):
                for k in range(3):
                    this_value = COLORS[self.current_grid_map[i, j]][k]
                    observation[i * gs0:(i + 1) * gs0, j * gs1:(j + 1) * gs1, k] = this_value
        return (255*observation).astype(np.uint8)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        img = self._gridmap_to_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)

    def change_start_state(self, sp):
        """ change agent start state
            Input: sp: new start state
         """
        if self.agent_start_state[0] == sp[0] and self.agent_start_state[1] == sp[1]:
            self._reset()
            return True
        elif self.start_grid_map[sp[0], sp[1]] != EMPTY:
            raise ValueError('Cannot move start position to occupied cell')
        else:
            s_pos = copy.deepcopy(self.agent_start_state)
            self.start_grid_map[s_pos[0], s_pos[1]] = EMPTY
            self.start_grid_map[sp[0], sp[1]] = AGENT
            self.current_grid_map = copy.deepcopy(self.start_grid_map)
            self.agent_start_state = [sp[0], sp[1]]
            self.agent_state = copy.deepcopy(self.agent_start_state)
            self._reset()
            self._render()
        return True

    def change_target_state(self, tg):
        if self.agent_target_state[0] == tg[0] and self.agent_target_state[1] == tg[1]:
            self._reset()
            return True
        elif self.start_grid_map[tg[0], tg[1]] != EMPTY:
            raise ValueError('Cannot move target position to occupied cell')
        else:
            t_pos = copy.deepcopy(self.agent_target_state)
            self.start_grid_map[t_pos[0], t_pos[1]] = EMPTY
            self.start_grid_map[tg[0], tg[1]] = TARGET
            self.current_grid_map = copy.deepcopy(self.start_grid_map)
            self.agent_target_state = [tg[0], tg[1]]
            self.agent_state = copy.deepcopy(self.agent_start_state)
            self._reset()
            self._render()
        return True

    @staticmethod
    def get_action_meaning():
        return ['UP', 'DOWN', 'LEFT', 'RIGHT']
