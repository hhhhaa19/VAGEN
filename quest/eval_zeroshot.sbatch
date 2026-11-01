#!/bin/bash
#SBATCH -A p53316
#SBATCH -p gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -J eval-zeroshot
#SBATCH -o /projects/p53316/logs/%x_%j.log
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd /projects/p53316/hhhhaa19/VAGEN
PORT=$((30000 + RANDOM % 1000)) MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct \
  bash /projects/p53316/run.sh bash examples/evaluate/sokoban/sglang/eval_qwen25_vl_3b.sh
echo "== EXIT CODE $?"
