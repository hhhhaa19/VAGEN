#!/bin/bash
#SBATCH -A p53316
#SBATCH -p gengpu
#SBATCH --gres=gpu:4
#SBATCH --constraint="quest12|quest13"
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH -t 03:00:00
#SBATCH -J vagen-legacy-eval
#SBATCH -o /projects/p53316/logs/%x_%j.log
# Validation only, on checkpoints of the legacy baseline (train_legacy.sbatch). Flags are
# those of the training run; only val_only, the checkpoint and the validation file change.
#   1. step 300 on legacy test.parquet: the 128 training levels, should give about 0.742
#   2. step 300 and step 150 on 256 maps that no training room equals (make_heldout.py)
set -uo pipefail
P=/projects/p53316
JOBS=$P/hhhhaa19/jobs
LEGACY=$P/hhhhaa19/VAGEN-legacy-check
MAIN=$P/hhhhaa19/VAGEN
ENVBIN=$P/envs/vagen-legacy/bin
SRC=$P/hhhhaa19/legacy_runs/baseline_local
CKPT=$SRC/checkpoints/vagen-legacy/legacy_baseline_local
RUN_DIR=$P/hhhhaa19/legacy_runs/heldout_eval
PORT=$((5000 + SLURM_JOB_ID % 1000))
JUDGE_PORT=$((8000 + SLURM_JOB_ID % 1000))
mkdir -p "$RUN_DIR/data"
cd "$RUN_DIR"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
git -C "$LEGACY" rev-parse HEAD; git -C "$LEGACY" status -s

export PATH=$ENVBIN:$PATH
export HF_HOME=$P/cache/hf
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONHASHSEED=0
export WANDB_MODE=disabled

PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done; }
trap cleanup EXIT

# Same local judge as the training run.
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

python -m vagen.server.server server.port=$PORT use_state_reward=True > server.log 2>&1 &
PIDS+=($!)
sleep 10

cp "$SRC/data/train.parquet" "$SRC/data/test.parquet" data/
if [ ! -f data/heldout.parquet ]; then
    ( cd "$LEGACY" && python "$JOBS/make_heldout.py" "$RUN_DIR/data/train.parquet" \
        "$RUN_DIR/data/test.parquet" "$RUN_DIR/data/heldout.parquet" 256 20000 ) || exit 1
fi

validate() {  # <checkpoint step> <validation parquet> <tag>
    python3 -m vagen.trainer.main_ppo \
        algorithm.adv_estimator=bi_level_gae \
        algorithm.high_level_gamma=0.95 \
        data.train_files=data/train.parquet \
        data.val_files=data/$2 \
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
        'trainer.logger=["console"]' \
        trainer.project_name=vagen-legacy \
        trainer.experiment_name=legacy_eval_$3 \
        trainer.n_gpus_per_node=4 \
        trainer.nnodes=1 \
        trainer.total_training_steps=300 \
        trainer.default_local_dir=$CKPT \
        trainer.resume_mode=$CKPT/global_step_$1 \
        trainer.val_before_train=True \
        trainer.val_only=True \
        trainer.val_generations_to_log_to_wandb=0 \
        rollout_manager.max_turns=3 \
        rollout_manager.window_size=5 \
        rollout_manager.use_multi_turn_reward=True \
        rollout_manager.use_loss_mask=True \
        rollout_manager.use_gae_mask=True \
        rollout_manager.n_trajectory=1 \
        rollout_manager.use_service=True \
        rollout_manager.timeout=300 \
        rollout_manager.base_url=http://localhost:$PORT \
        > val_$3.log 2>&1
    local rc=$?
    echo "== $3 (step $1 on $2) exit $rc"
    grep -oE "val/success/[^:]*: ?[0-9.]+" val_$3.log | tail -1
}

validate 300 test.parquet step300_seen
validate 300 heldout.parquet step300_heldout
validate 150 heldout.parquet step150_heldout
echo "== EXIT CODE 0"
