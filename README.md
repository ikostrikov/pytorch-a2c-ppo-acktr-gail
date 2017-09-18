# pytorch-a2c-ppo-acktr

This is a PyTorch implementation of
* Advantage Actor Critic (A2C), a synchronous deterministic version of [A3C](https://arxiv.org/pdf/1602.01783v1.pdf)
* Proximal Policy Optimization [PPO](https://arxiv.org/pdf/1707.06347.pdf)
* Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation [ACKTR](https://arxiv.org/abs/1708.05144)

Also see the OpenAI posts: [A2C/ACKTR](https://blog.openai.com/baselines-acktr-a2c/) and [PPO](https://blog.openai.com/openai-baselines-ppo/) for more information.

This implementation is inspired by the OpenAI baselines for [A2C](https://github.com/openai/baselines/tree/master/baselines/a2c), [ACKTR](https://github.com/openai/baselines/tree/master/baselines/acktr) and [PPO](https://github.com/openai/baselines/tree/master/baselines/ppo1). It uses the same hyper parameters and the model since they were well tuned for Atari games.

## Contributions

Contributions are very welcome. If you know how to make this code better, don't hesitate to send a pull request. Also see a todo list below.

### TODO
* Add MuJoCo and continuous actions
* Improve performance of KFAC, see kfac.py for more information
* Run evaluation for all games and algorithms

## Usage

### A2C

```
python main.py --env-name "PongNoFrameskip-v4"
```

### PPO

```
python main.py --env-name "PongNoFrameskip-v4" --algo ppo --use-gae --num-processes 8 --num-steps 256 --vis-interval 1 --log-interval 1
```

### ACKTR

```
python main.py --env-name "PongNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20
```

## Results

### A2C

![BreakoutNoFrameskip-v4](imgs/breakout.png)

![SeaquestNoFrameskip-v4](imgs/seaquest.png)

![QbertNoFrameskip-v4](imgs/qbert.png)

![beamriderNoFrameskip-v4](imgs/beamrider.png)

### PPO

Coming soon.

### ACKTR

Coming soon.
