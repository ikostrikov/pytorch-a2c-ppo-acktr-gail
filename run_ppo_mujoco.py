#!/home/kostrikov/.linuxbrew/bin/fish

set envs  "Reacher-v1" "HalfCheetah-v1" "Hopper-v1" "Walker2d-v1"
set seed 42
for x in (seq 4)
  python main.py --env-name "$envs[$x]" --seed $seed --algo ppo --use-gae --vis-interval 1  --log-interval 1 --num-stack 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --ppo-epoch 10 --batch-size 64 --log-dir "/tmp/gym/$x" --gamma 0.99 --tau 0.95&
  set seed (math $seed + 1)
end
