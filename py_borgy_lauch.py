import os

debug = 0

wandb = 'pupper1'

SEEDS = list(range(3))
HIDDEN_SIZES = [64, 256]
HIDDEN_SIZES = [128]
# FRAME_STACCS = [1,4]
FRAME_STACCS = [1]
NUM_PROCESSES = [1,4]

if debug:
    wandb = 'temp'

for seed in SEEDS:
  for hidden_size in HIDDEN_SIZES:
    for frame_stacc in FRAME_STACCS:
      for num_processes in NUM_PROCESSES:
        command = ("borgy submit -i images.borgy.elementai.net/fgolemo/gym:v4 " 
                "--mem 32 --gpu-mem 12 --gpu 1 --cuda-version 10.1 -H -- bash -c "
                " 'cd /root && export PATH=/mnt/home/optimass/miniconda3/bin/:$PATH " 
                "&& export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/home/optimass/.mujoco/mujoco200/bin " 
                "&& cd ~/StanfordQuadruped/ && pip install -e . " 
                "&& pip install wandb "
                "&& cd ~/pytorch-a2c-ppo-acktr-gail && ls . && python main.py "
                "--custom-gym stanford_quad "
                "--env-name Pupper-Walk-Relative-ScaledDown-Headless-v0 "
                "--algo ppo "
                "--use-gae "
                "--log-interval 1 "
                "--num-steps 2048 "
                "--num-processes {num_processes} "
                "--lr 3e-4 "
                "--entropy-coef 0 "
                "--value-loss-coef 0.5 "
                "--ppo-epoch 10 "
                "--num-mini-batch 32 "
                "--gamma 0.99 "
                "--gae-lambda 0.95 "
                "--frame-stacc {frame_stacc} "
                "--num-env-steps 2000000 "
                "--use-linear-lr-decay "
                "--use-proper-time-limits "
                "--save-interval 10 "
                "--hidden_size {hidden_size} "
                "--seed {seed} "
                "--wandb {wandb} "
                "> ~/borgy/pupper.log "
                "'".format(seed=seed, hidden_size=hidden_size, 
                           frame_stacc=frame_stacc, wandb=wandb,
                           num_processes=num_processes)
        )

        print(command)
        os.system(command)

        if debug:
            exit()


# (export SEEDBASE=0; export EXP=0; for ((i = 0; i < 1; i++)); do borgy submit -i 
# images.borgy.elementai.net/fgolemo/gym:v4 --mem 32 --gpu-mem 12 --gpu 1 --cuda-version 10.2 -H 
# -- zsh -c "cd /root && source ./.zshrc 
# && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/home/$USER/.mujoco/mujoco200/bin 
# && cd ~/StanfordQuadruped/ && pip install -e . 
# && cd ~/pytorch-a2c-ppo-acktr-gail && python main.py --custom-gym 'stanford_quad' --env-name 'XXXXXXXXXXXXXXX' --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --frame-stacc 1 --num-env-steps 1000000 --use-linear-lr-decay --use-proper-time-limits --comet fgolemo/pupper-walk/ZfKpzyaedH6ajYSiKmvaSwyCs --comet-tags v2,lr3,nostacc --save-interval 10 --seed $((i+SEEDBASE)) > ~/borgy/pupperwalk-exp${EXP}-$((i+SEEDBASE)).log"; done)