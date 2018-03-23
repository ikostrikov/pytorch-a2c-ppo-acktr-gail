from gym.envs.registration import register

for i in range(10):
    for stochastic in [True, False]:
        register(
            id='Gridworld{}{}-v0'.format(i, 'Stochastic' if stochastic else ''),
            kwargs={'plan': i, 'stochastic': stochastic},
            entry_point='gym_gridworld.envs:GridworldEnv',
        )

register(
    id='SequentialGridworld{}-v0'.format('Stochastic'),
    kwargs={'plans': list(range(1, 6)), 'stochastic': True},
    entry_point='gym_gridworld.envs:SequentialGridworldEnv',
)