from setuptools import find_packages, setup

setup(
    name='a2c-ppo-acktr',
    packages=find_packages(),
    version='0.0.1',
    install_requires=['gym', 'matplotlib', 'pybullet', 'stable-baselines3'])
