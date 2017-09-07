# pytorch-a2c

This is a PyTorch implementation of Advantage Actor Critic (A2C), a synchronous deterministic version of A3C ["Asynchronous Methods for Deep Reinforcement Learning"](https://arxiv.org/pdf/1602.01783v1.pdf). Also see [the OpenAI post](https://blog.openai.com/baselines-acktr-a2c/) (section A2C and A3C) for more information.

This implementation is inspired by the [OpenAI A2C baseline](https://github.com/openai/baselines/tree/master/baselines/a2c). It uses the same hyper parameters and the model since they were well tuned for Atari games.

## Contibutions

Contributions are very welcome. If you know how to make this code better, don't hesitate to send a pull request.

## Usage
```
python main.py --env-name "PongNoFrameskip-v4"
```

## Results

Coming soon.
