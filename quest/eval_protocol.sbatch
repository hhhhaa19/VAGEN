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
#SBATCH -J vagen-eval-protocol
#SBATCH -o /projects/p53316/logs/%x_%j.log
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd /projects/p53316/hhhhaa19/VAGEN
export STATE_ESTIMATION_REWARD=0.5 TRANSITION_PREDICTION_REWARD=0.5 STATE_REWARD_SCORE_BASE=0
export STATE_REWARD_SCORER=exact
D=examples/train/sokoban
CKPT=$PWD/exps/vagen-experiments/sokoban_bi_level_gae_paper_full_qwen25vl3b/verl_checkpoints/global_step_400/actor/huggingface
SAMPLE="actor_rollout_ref.rollout.val_kwargs.do_sample=True actor_rollout_ref.rollout.val_kwargs.temperature=0.7 actor_rollout_ref.rollout.val_kwargs.top_p=0.95"
GREEDY="actor_rollout_ref.rollout.val_kwargs.do_sample=False actor_rollout_ref.rollout.val_kwargs.temperature=0 actor_rollout_ref.rollout.val_kwargs.top_p=1.0"
run() {
  name=$1; val=$2; shift 2
  echo "== RUN $name"
  EXPERIMENT_NAME=eval_step400_$name bash /projects/p53316/run.sh bash $D/train_bi_level_gae_sr_qwen25vl3b.sh \
    data.train_files=$PWD/$D/train_sokoban_vision_sr_paper.yaml \
    data.val_files=$PWD/$D/$val \
    actor_rollout_ref.model.path=$CKPT \
    trainer.val_only=True trainer.val_before_train=True trainer.resume_mode=disable \
    trainer.logger='[console]' \
    "$@"
  echo "== RUN $name EXIT $?"
}
run seen_sample val_sokoban_vision_sr_seen128.yaml $SAMPLE
run seen_greedy val_sokoban_vision_sr_seen128.yaml $GREEDY
run unseen_sample val_sokoban_vision_sr_paper.yaml $SAMPLE
run unseen_greedy val_sokoban_vision_sr_paper.yaml $GREEDY
