from gym import register

register(
    id='Dummy-Stateful-v0',
    entry_point='a2c_ppo_acktr.gym_envs:DummyEnv',
    max_episode_steps=10,
    kwargs={})
