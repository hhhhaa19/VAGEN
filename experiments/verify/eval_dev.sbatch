#!/bin/bash
# B, G, R and D1 on the dev set, greedy decoding, always on H100 (quest13) so the settings compare.
# Same command as quest/eval_protocol.sbatch (quest-jobs branch); only the harness differs.
# Submit from your own clone: cd /projects/p53316/bowu2026-cpu/VAGEN && sbatch experiments/verify/eval_dev.sbatch
# Pick settings with SETTINGS="R G" sbatch ...; default runs all four.
# Pick the model with CKPT=<.../huggingface> TAG=<short name> sbatch ...; default is main step 400.
# Pick the levels with VAL=<yaml in examples/train/sokoban>: *_dev (default) to tune, the test file once;
# the *_5turn files are the same levels with max_turns 5, for checkpoints trained with 5 turns.
#SBATCH -A p53316
#SBATCH -p gengpu
#SBATCH --gres=gpu:h100:4
#SBATCH --constraint=quest13
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH -t 02:00:00
#SBATCH -J verify-dev
#SBATCH -o /projects/p53316/logs/%x_%j.log
set -u
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
V=$(pwd)
[ -f "$V/experiments/verify/eval_dev.sbatch" ] || { echo "submit from the VAGEN clone root" >&2; exit 1; }
export STATE_ESTIMATION_REWARD=0.5 TRANSITION_PREDICTION_REWARD=0.5 STATE_REWARD_SCORE_BASE=0
export STATE_REWARD_SCORER=exact
D=examples/train/sokoban
VAL=${VAL:-val_sokoban_vision_sr_paper_dev.yaml}
CKPT=${CKPT:-/projects/p53316/hhhhaa19/VAGEN/exps/vagen-experiments/sokoban_bi_level_gae_paper_full_qwen25vl3b/verl_checkpoints/global_step_400/actor/huggingface}
TAG=${TAG:-step400}
[ -r "$CKPT/config.json" ] || { echo "cannot read $CKPT" >&2; exit 1; }
for w in "$CKPT"/*.safetensors; do [ -r "$w" ] || { echo "cannot read $w" >&2; exit 1; }; done
[ -f "$V/$D/$VAL" ] || { echo "no $D/$VAL" >&2; exit 1; }
echo "ckpt=$CKPT tag=$TAG val=$VAL commit=$(git -C "$V" rev-parse --short HEAD)"
# The newprompt checkpoint needs prompt.patch applied (2); every other checkpoint needs 0.
echo "prompt check: $(grep -c "AFTER all actions" "$V/vagen/envs/sokoban/utils/prompt.py") AFTER-all-actions rules"
GREEDY="actor_rollout_ref.rollout.val_kwargs.do_sample=False actor_rollout_ref.rollout.val_kwargs.temperature=0 actor_rollout_ref.rollout.val_kwargs.top_p=1.0"
declare -A HARNESS=(
  [B]=concat
  [G]=vagen.harness.verify:GenericNoteHarness
  [R]=vagen.harness.verify:NoRepeatHarness
  [D1]=vagen.harness.verify:VerifyHarness
)
for name in ${SETTINGS:-B G R D1}; do
  out=$V/exps/verify-dev/${TAG}/${VAL%.yaml}_$name
  mkdir -p "$out"
  rm -f "$out/verify.jsonl"
  echo "== RUN $name -> $out"
  VERIFY_LOG=$out/verify.jsonl EXPERIMENT_NAME=dev_${TAG}_$name EXPERIMENT_DIR=$out \
    bash /projects/p53316/run.sh bash $D/train_bi_level_gae_sr_qwen25vl3b.sh \
    data.train_files=$V/$D/train_sokoban_vision_sr_paper.yaml \
    data.val_files=$V/$D/$VAL \
    actor_rollout_ref.model.path=$CKPT \
    trainer.harness=${HARNESS[$name]} \
    trainer.val_only=True trainer.val_before_train=True trainer.resume_mode=disable \
    trainer.logger='[console]' \
    $GREEDY
  echo "== RUN $name EXIT $?"
done
