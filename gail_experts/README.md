## Data

Download from 
https://drive.google.com/open?id=1Ipu5k99nwewVDG1yFetUxqtwVlgBg5su

and store in this folder.

## Convert to pytorch

```bash
python convert_to_pytorch.py --h5-file trajs_halfcheetah.h5
```

## Run

```bash
python main.py --env-name "HalfCheetah-v2" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 10000000 --use-linear-lr-decay --use-proper-time-limits --gail
```
