#!/bin/bash
# Build the vagen-legacy environment (VAGEN 9cfe7f0 + JamesKrW/verl vagen-legacy).
# Runs on the login node: package installs only, no GPU.
set -euo pipefail
P=/projects/p53316
ENV=$P/envs/vagen-legacy
VERL=$P/hhhhaa19/verl-legacy
VAGEN=$P/hhhhaa19/VAGEN-legacy-check
export PIP_CACHE_DIR=$P/cache/pip
export CONDA_PKGS_DIRS=$P/cache/conda-pkgs
module purge
module load mamba/24.3.0

[ -d "$ENV" ] || mamba create -y -p "$ENV" python=3.10
PIP="$ENV/bin/python -m pip"
$PIP install -U pip

echo "== torch"
$PIP install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

echo "== verl (vagen-legacy)"
[ -d "$VERL" ] || git clone https://github.com/JamesKrW/verl.git "$VERL"
git -C "$VERL" checkout vagen-legacy
git -C "$VERL" log -1 --format='verl %H %ad'
$PIP install -e "$VERL"

echo "== flash-attn 2.7.4.post1 (prebuilt wheel)"
$PIP install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

echo "== vagen deps (scripts/install.sh, minus flash-attn)"
$PIP install qwen-vl-utils mathruler matplotlib flask gym gym-sokoban gymnasium "gymnasium[toy-text]" together openai
$PIP install -e "$VAGEN"

echo "== versions"
$ENV/bin/python -c "import torch, vllm, transformers, tensordict, verl, vagen, openai; print('torch', torch.__version__, 'vllm', vllm.__version__, 'transformers', transformers.__version__, 'tensordict', tensordict.__version__, 'openai', openai.__version__)"
$ENV/bin/python -c "import flash_attn; print('flash_attn', flash_attn.__version__)"
echo "== DONE"
