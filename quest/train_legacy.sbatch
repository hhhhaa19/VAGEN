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
#SBATCH -J vagen-legacy
#SBATCH -o /projects/p53316/logs/%x_%j.log
# vagen-legacy VAGEN-Full Sokoban: scripts/examples/vagen_full/sokoban/run_tmux.sh as an
# sbatch job, training flags unchanged. Only the tmux sessions, paths and ports differ.
#   STEPS=2 NAME=smoke sbatch -t 01:30:00 train_legacy.sbatch   short pipeline check
#   JUDGE=openai sbatch train_legacy.sbatch                     needs OPENAI_API_KEY set
set -uo pipefail
P=/projects/p53316
JOBS=$P/hhhhaa19/jobs
LEGACY=$P/hhhhaa19/VAGEN-legacy-check
MAIN=$P/hhhhaa19/VAGEN
SCRIPT_DIR=$LEGACY/scripts/examples/vagen_full/sokoban
ENVBIN=$P/envs/vagen-legacy/bin
JUDGE=${JUDGE:-local}
STEPS=${STEPS:-300}
NAME=${NAME:-baseline}_$JUDGE
RUN_DIR=$P/hhhhaa19/legacy_runs/$NAME
PORT=$((5000 + SLURM_JOB_ID % 1000))
JUDGE_PORT=$((8000 + SLURM_JOB_ID % 1000))
mkdir -p "$RUN_DIR/data"
cd "$RUN_DIR"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
git -C "$LEGACY" rev-parse HEAD; git -C "$LEGACY" status -s
git -C $P/hhhhaa19/verl-legacy rev-parse HEAD

export PATH=$ENVBIN:$PATH
export HF_HOME=$P/cache/hf
export WANDB_ENTITY=cs496_vagen WANDB_DIR=$P/logs/wandb
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONHASHSEED=0
[ "$STEPS" -lt 20 ] && export WANDB_MODE=offline

PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done; }
trap cleanup EXIT

case "$JUDGE" in
  local)
    # Same judge as our main runs. The legacy client code is untouched: it reads these
    # two variables, and the server answers with Qwen whatever model name is requested.
    export OPENAI_BASE_URL=http://127.0.0.1:$JUDGE_PORT/v1 OPENAI_API_KEY=local
    ( cd "$MAIN" && MODEL=Qwen/Qwen3-4B-Instruct-2507 PORT=$JUDGE_PORT TP=4 MEM=0.10 \
        bash $P/run.sh bash scripts/launch_judge.sh ) > judge.log 2>&1 &
    PIDS+=($!)
    for _ in $(seq 1 90); do
        curl -sf -m 3 "http://127.0.0.1:$JUDGE_PORT/health" >/dev/null 2>&1 && break
        kill -0 "${PIDS[0]}" 2>/dev/null || { tail -n 60 judge.log; exit 1; }
        sleep 10
    done
    curl -sf -m 3 "http://127.0.0.1:$JUDGE_PORT/health" >/dev/null || { tail -n 60 judge.log; exit 1; }
    ;;
  openai)
    [ -n "${OPENAI_API_KEY:-}" ] || { echo "OPENAI_API_KEY is not set"; exit 1; }
    ;;
  *) echo "JUDGE must be local or openai"; exit 1 ;;
esac

python -m vagen.server.server server.port=$PORT use_state_reward=True > server.log 2>&1 &
PIDS+=($!)
sleep 10

python -m vagen.env.create_dataset \
    --yaml_path "$SCRIPT_DIR/env_config.yaml" \
    --train_path "data/train.parquet" \
    --test_path "data/test.parquet" || exit 1

python3 -m vagen.trainer.main_ppo \
    algorithm.adv_estimator=bi_level_gae \
    algorithm.high_level_gamma=0.95 \
    data.train_files=data/train.parquet \
    data.val_files=data/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=200 \
    data.max_trajectory_length=2400 \
    data.image_key=images \
    data.truncation=left \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=mse \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.temperature=0.7 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    'trainer.logger=["console","wandb"]' \
    trainer.project_name=vagen-legacy \
    trainer.experiment_name=legacy_$NAME \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=150 \
    trainer.test_freq=20 \
    trainer.total_training_steps=$STEPS \
    rollout_manager.max_turns=3 \
    rollout_manager.window_size=5 \
    rollout_manager.use_multi_turn_reward=True \
    rollout_manager.use_loss_mask=True \
    rollout_manager.use_gae_mask=True \
    trainer.val_before_train=True \
    trainer.val_generations_to_log_to_wandb=8 \
    rollout_manager.n_trajectory=1 \
    rollout_manager.use_service=True \
    rollout_manager.timeout=300 \
    rollout_manager.base_url=http://localhost:$PORT \
    2>&1 | tee train.log
rc=${PIPESTATUS[0]}; echo "== EXIT CODE $rc"; exit $rc
