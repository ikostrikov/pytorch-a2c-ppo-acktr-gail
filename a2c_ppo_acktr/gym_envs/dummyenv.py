import gym
from gym import spaces
import numpy as np


class DummyEnv(gym.Env):

    def __init__(self):
        self.correct_actions = [0, 1, 2, 3]
        self.state = 0

        self.action_space = spaces.Discrete(4)  # up, down, left, right

        # we need to merge everything into 1 tensor, not a dict
        self.observation_space = spaces.Box(low=-0, high=1, shape=(4, 84, 84))

        super().__init__()

    def step(self, action):
        assert 0 <= action < 4

        done = False
        reward = -1

        if action == self.correct_actions[self.state]:
            self.state += 1
            reward = 1
            if self.state == 4:
                done = True

        return self.observe(), reward, done, {}

    def observe(self):
        img = np.zeros((3, 84, 84),
                       dtype=np.float32)  # already transposed for pytorch

        coord_holder = np.zeros((1, 84, 84), dtype=np.float32)
        coord_holder[0, 0, :2] = [4, 4]  # goal
        coord_holder[0, 0, 2:4] = [4 - self.state, 4 - self.state]  # rel_goal

        out = np.concatenate((img, coord_holder), axis=0)

        return out

    def reset(self):
        self.state = 0
        return self.observe()


if __name__ == '__main__':

    import a2c_ppo_acktr  # to get the gym envs
    env = gym.make("Dummy-Stateful-v0")

    for _ in range(10):
        env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, rew, done, misc = env.step(action)
            print(action, env.unwrapped.state, rew)

        print("===")
