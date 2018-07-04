# pytorch-a2c-a2oc-ppo-acktr

## Update 07/04/2018: Added a2oc!

## WARNING: Currently the termination loss calculation for a2oc is ignored thus the options are random and the policy over options network doesn't learn or converge. This will hopefully be fixed soon! Also, get_options_value doesn't return the correct value either... it returns exactly the value of "value" and then they cancel each other out when calculating termination loss, again leading to a bad result. Point is - termination loss is broken and therefore you shouldn't run a2oc!


## Update 10/06/2017: added enjoy.py and a link to pretrained models!
## Update 09/27/2017: now supports both Atari and MuJoCo/Roboschool!

This is a PyTorch implementation of
* Advantage Actor Critic (A2C), a synchronous deterministic version of [A3C](https://arxiv.org/pdf/1602.01783v1.pdf)
* Proximal Policy Optimization [PPO](https://arxiv.org/pdf/1707.06347.pdf)
* Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation [ACKTR](https://arxiv.org/abs/1708.05144)

Also see the OpenAI posts: [A2C/ACKTR](https://blog.openai.com/baselines-acktr-a2c/) and [PPO](https://blog.openai.com/openai-baselines-ppo/) for more information.

This implementation is inspired by the OpenAI baselines for [A2C](https://github.com/openai/baselines/tree/master/baselines/a2c), [ACKTR](https://github.com/openai/baselines/tree/master/baselines/acktr) and [PPO](https://github.com/openai/baselines/tree/master/baselines/ppo1). It uses the same hyper parameters and the model since they were well tuned for Atari games.

Please use this bibtex if you want to cite this repository in your publications:

    @misc{pytorchrl,
      author = {Kostrikov, Ilya},
      title = {PyTorch Implementations of Reinforcement Learning Algorithms},
      year = {2018},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/ikostrikov/pytorch-a2c-ppo-acktr}},
    }

## Supported (and tested) environments (via [OpenAI Gym](https://gym.openai.com))
* [Atari Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment)
* [MuJoCo](http://mujoco.org)
* [PyBullet](http://pybullet.org) (including Racecar, Minitaur and Kuka)
* [DeepMind Control Suite](https://github.com/deepmind/dm_control) (via [dm_control2gym](https://github.com/martinseilair/dm_control2gym))

I highly recommend PyBullet as a free open source alternative to MuJoCo for continuous control tasks.

All environments are operated using exactly the same Gym interface. See their documentations for a comprehensive list.

To use the DeepMind Control Suite environments, set the flag `--env-name dm.<domain_name>.<task_name>`, where `domain_name` and `task_name` are the name of a domain (e.g. `hopper`) and a task within that domain (e.g. `stand`) from the DeepMind Control Suite. Refer to their repo and their [tech report](https://arxiv.org/abs/1801.00690) for a full list of available domains and tasks. Other than setting the task, the API for interacting with the environment is exactly the same as for all the Gym environments thanks to [dm_control2gym](https://github.com/martinseilair/dm_control2gym).

## Requirements

* Python 3 (it might work with Python 2, but I didn't test it)
* [PyTorch](http://pytorch.org/)
* [Visdom](https://github.com/facebookresearch/visdom)
* [OpenAI baselines](https://github.com/openai/baselines)

In order to install requirements, follow:

```bash
# PyTorch
conda install pytorch torchvision -c soumith

# Baselines for Atari preprocessing
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .

# Other requirements
pip install -r requirements.txt
```

## Contributions

Contributions are very welcome. If you know how to make this code better, please open an issue. If you want to submit a pull request, please open an issue first. Also see a todo list below.

Also I'm searching for volunteers to run all experiments on Atari and MuJoCo (with multiple random seeds).

## Disclaimer

It's extremely difficult to reproduce results for Reinforcement Learning methods. See ["Deep Reinforcement Learning that Matters"](https://arxiv.org/abs/1709.06560) for more information. I tried to reproduce OpenAI results as closely as possible. However, majors differences in performance can be caused even by minor differences in TensorFlow and PyTorch libraries.

### TODO
* Fix termination loss calculation for a2oc. Right now it's being ignored.
* Clean the code because a ton of stuff is commented out or deprecated, especially in algo/a2oc.py
* Improve this README file. Rearrange images.
* Improve performance of KFAC, see kfac.py for more information.
* Run evaluation for all games and algorithms.

## Training

Start a `Visdom` server with `python -m visdom.server`, it will serve `http://localhost:8097/` by default.

### Atari
#### A2C

```bash
python main.py --algo a2c --env-name "PongNoFrameskip-v4"
```

#### PPO

```bash
python main.py --env-name "PongNoFrameskip-v4" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 8 --num-steps 128 --num-mini-batch 4 --vis-interval 1 --log-interval 1
```

#### ACKTR

```bash
python main.py --algo a2c --env-name "PongNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20
```

### MuJoCo

I **highly** recommend to use --add-timestep argument with some mujoco environments (for example, Reacher) despite it's not a default option with OpenAI implementations.

#### A2OC (default algorithm is a2oc so no --algo a2oc is required)

```bash
python main.py --env-name "Reacher-v2" --num-stack 1 --num-frames 1000000
```

#### PPO

```bash
python main.py --env-name "Reacher-v2" --algo ppo --use-gae --vis-interval 1  --log-interval 1 --num-stack 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 1 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --tau 0.95 --num-frames 1000000
```

#### ACKTR

ACKTR requires some modifications to be made specifically for MuJoCo. But at the moment, I want to keep this code as unified as possible. Thus, I'm going for better ways to integrate it into the codebase.

## Enjoy

Load a pretrained model from [my Google Drive](https://drive.google.com/open?id=0Bw49qC_cgohKS3k2OWpyMWdzYkk).

Also pretrained models for other games are available on request. Send me an email or create an issue, and I will upload it.

Disclaimer: I might have used different hyper-parameters to train these models.

### Atari

```bash
python enjoy.py --algo a2oc --load-dir trained_models/a2c --env-name "PongNoFrameskip-v4" --num-stack 4
```

### MuJoCo

```bash
python enjoy.py --load-dir trained_models/ppo --env-name "Reacher-v2" --num-stack 1
```

## Results

### A2C

![BreakoutNoFrameskip-v4](imgs/a2c_breakout.png)

![SeaquestNoFrameskip-v4](imgs/a2c_seaquest.png)

![QbertNoFrameskip-v4](imgs/a2c_qbert.png)

![beamriderNoFrameskip-v4](imgs/a2c_beamrider.png)

### PPO


![BreakoutNoFrameskip-v4](imgs/ppo_halfcheetah.png)

![SeaquestNoFrameskip-v4](imgs/ppo_hopper.png)

![QbertNoFrameskip-v4](imgs/ppo_reacher.png)

![beamriderNoFrameskip-v4](imgs/ppo_walker.png)


### ACKTR

![BreakoutNoFrameskip-v4](imgs/acktr_breakout.png)

![SeaquestNoFrameskip-v4](imgs/acktr_seaquest.png)

![QbertNoFrameskip-v4](imgs/acktr_qbert.png)

![beamriderNoFrameskip-v4](imgs/acktr_beamrider.png)
