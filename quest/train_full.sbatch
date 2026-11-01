#!/bin/bash
#SBATCH -A p53316
#SBATCH -p gengpu
#SBATCH --gres=gpu:4
#SBATCH --constraint="quest12|quest13"
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH -t 24:00:00
#SBATCH -J vagen-full
#SBATCH -o /projects/p53316/logs/%x_%j.log
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd /projects/p53316/hhhhaa19/VAGEN
bash /projects/p53316/run.sh bash examples/train/sokoban/train_bi_level_gae_sr_qwen25vl3b.sh \
  trainer.max_actor_ckpt_to_keep=6 trainer.max_critic_ckpt_to_keep=1
echo "== EXIT CODE $?"
