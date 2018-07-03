import os
from itertools import cycle

from .gridworld_env import GridworldEnv


class SequentialGridworldEnv(GridworldEnv):
    def __init__(self, plans, stochastic=False):
        super(SequentialGridworldEnv, self).__init__(plans[0], stochastic)
        self.plans = plans
        self.current_plan_it = cycle(self.plans)
        self.current_plan = 1
        self.path = os.path.dirname(os.path.realpath(__file__))
        self.steps = 0
        self.max_steps = 100

    def _reset(self, success=False):
        if not success:
            self.current_plan_it = cycle(self.plans)
            next(self.current_plan_it)
            self.current_plan = next(self.current_plan_it)
        self.steps = 0
        self.current_plan = next(self.current_plan_it)
        grid_map_path = os.path.join(self.path, 'plan{}{}.txt'.format(self.current_plan,
                                                                      's' if self.stochastic else ''))
        self.start_grid_map = self._read_grid_map(grid_map_path)  # initial grid map
        super(SequentialGridworldEnv, self)._reset()
        return self.current_grid_map

    def _step(self, action):
        self.steps += 1
        state, reward, done, info = super(SequentialGridworldEnv, self)._step(action)
        info['plan'] = self.current_plan
        if self.steps == self.max_steps:
            done = True
            reward = -1
        if done:
            if reward == 1:
                done = False
                state = self._reset(success=not done)
            elif reward == -1:
                self.current_plan_it = cycle(self.plans)
                self.current_plan = 1
        return state, reward, done, info
