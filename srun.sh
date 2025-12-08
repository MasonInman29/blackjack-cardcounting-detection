#!/bin/bash

account_name="yangli1-lab"
# account_name="bweng-lab"

stage_1_model_path="play_network_supervised_v7_best.pth"
# stage_2_model_path="rl_model_stage2_v4.pth"
stage_2_model_path="play_network_supervised_v7_best.pth"
stage_3_model_path="rl_model_stage3.path"

# srun --account="$account_name" --time=24:00:00 --nodes=1 --cpus-per-task=16 --mem=64G --partition=nova --gres=gpu:a100:1 \
#   python train_play_supervised.py

# srun --account="$account_name" --time=24:00:00 --nodes=1 --cpus-per-task=16 --mem=64G --partition=nova --gres=gpu:a100:1 \
#   python train_rl_model_dqn.py \
#   --stage 2 \
#   --load_path "$stage_1_model_path" \
#   --save_path "$stage_2_model_path" \
#   --plot_path "training_progress_stage2_v4.png"



srun --account="$account_name" --time=24:00:00 --nodes=1 --cpus-per-task=16 --mem=64G --partition=nova --gres=gpu:a100:1 \
  python train_rl_model_dqn.py \
  --stage 3 \
  --load_path "$stage_2_model_path" \
  --save_path "$stage_3_model_path" \
  --plot_path "training_progress_stage3.png"