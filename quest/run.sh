#!/bin/bash
module purge >/dev/null 2>&1
module load singularityce/4.3.1-gcc-8.5.0
P=/projects/p53316
SIF=$P/containers/sglang-0.5.13-cu130.sif
export SINGULARITYENV_HF_HOME=$P/cache/hf
export SINGULARITYENV_PIP_CACHE_DIR=$P/cache/pip
export SINGULARITYENV_WANDB_DIR=$P/logs/wandb
export SINGULARITYENV_WANDB_ENTITY=cs496_vagen
export SINGULARITYENV_FLASHINFER_WORKSPACE_BASE=$P/cache
export SINGULARITYENV_TRITON_CACHE_DIR=$P/cache/triton
ROOT=$(git -C "$PWD" rev-parse --show-toplevel 2>/dev/null); ROOT=${ROOT#/gpfs}
[ -n "$ROOT" ] && [ -d "$ROOT/vagen" ] && export SINGULARITYENV_PYTHONPATH="$ROOT/verl:$ROOT"
[ $# -eq 0 ] && set -- bash
exec singularity exec --nv -B $P --pwd "$PWD" $SIF bash -c 'source /projects/p53316/envs/vagen-sglang/bin/activate && exec "$@"' bash "$@"
