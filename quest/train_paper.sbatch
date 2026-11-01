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
#SBATCH -J vagen-paper
#SBATCH -o /projects/p53316/logs/%x_%j.log
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd /projects/p53316/hhhhaa19/VAGEN
git rev-parse --abbrev-ref HEAD; git status -s
export STATE_ESTIMATION_REWARD=0.5 TRANSITION_PREDICTION_REWARD=0.5 STATE_REWARD_SCORE_BASE=0
export EXPERIMENT_NAME=sokoban_bi_level_gae_paper_config_qwen25vl3b
D=examples/train/sokoban
bash /projects/p53316/run.sh bash $D/train_bi_level_gae_sr_qwen25vl3b.sh \
  data.train_files=$PWD/$D/train_sokoban_vision_sr_paper.yaml \
  data.val_files=$PWD/$D/val_sokoban_vision_sr_paper.yaml \
  trainer.max_actor_ckpt_to_keep=null trainer.save_best_actor=false
echo "== EXIT CODE $?"
