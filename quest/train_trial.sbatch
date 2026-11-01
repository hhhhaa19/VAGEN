#!/bin/bash
#SBATCH -A p53316
#SBATCH -p gengpu
#SBATCH --gres=gpu:4
#SBATCH --constraint="quest12|quest13"
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH -t 02:00:00
#SBATCH -J train-trial
#SBATCH -o /projects/p53316/logs/%x_%j.log
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd /projects/p53316/hhhhaa19/VAGEN
PROJECT_NAME=vagen-trial EXPERIMENT_NAME=trial_bi_level_gae_sr_qwen25vl3b \
  bash /projects/p53316/run.sh bash examples/train/sokoban/train_bi_level_gae_sr_qwen25vl3b.sh \
  trainer.total_training_steps=10 trainer.test_freq=5 trainer.save_freq=10
echo "== EXIT CODE $?"
