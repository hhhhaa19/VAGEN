#!/bin/bash
# Continue install_legacy.sh after verl failed on vllm 0.8.2's pin xgrammar==0.1.16,
# which is no longer on PyPI. xgrammar only backs guided decoding, which VAGEN does not
# use, so install the nearest release (0.1.17) and every other pinned dependency as is.
set -euo pipefail
P=/projects/p53316
ENV=$P/envs/vagen-legacy
VERL=$P/hhhhaa19/verl-legacy
VAGEN=$P/hhhhaa19/VAGEN-legacy-check
export PIP_CACHE_DIR=$P/cache/pip
PY=$ENV/bin/python
PIP="$PY -m pip"

reqs() {  # requirements of an installed or indexed distribution, minus the given names
    $PY - "$@" <<'PYEOF'
import json, re, sys, urllib.request
name, version, *skip = sys.argv[1:]
data = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json"))
for r in data["info"]["requires_dist"] or []:
    if "extra ==" in r:
        continue
    base = re.split(r"[ ;<>=!~\[(]", r, 1)[0].lower()
    if base not in skip:
        print(r)
PYEOF
}

echo "== vllm 0.8.2 (no deps) + its requirements except xgrammar"
$PIP install --no-deps vllm==0.8.2
reqs vllm 0.8.2 xgrammar > /tmp/vllm_reqs_$$.txt
cat /tmp/vllm_reqs_$$.txt
$PIP install -r /tmp/vllm_reqs_$$.txt
$PIP install xgrammar==0.1.17

echo "== verl (no deps) + its requirements except vllm"
git -C "$VERL" log -1 --format='verl %H %ad'
$PIP install --no-deps -e "$VERL"
$PY - "$VERL" > /tmp/verl_reqs_$$.txt <<'PYEOF'
import ast, sys
src = open(sys.argv[1] + "/setup.py").read()
tree = ast.parse(src)
for node in ast.walk(tree):
    if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "install_requires" for t in node.targets):
        for item in ast.literal_eval(node.value):
            if not item.lower().startswith("vllm"):
                print(item)
PYEOF
cat /tmp/verl_reqs_$$.txt
$PIP install -r /tmp/verl_reqs_$$.txt

echo "== flash-attn 2.7.4.post1 (prebuilt wheel)"
$PIP install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

echo "== vagen deps (scripts/install.sh, minus flash-attn)"
$PIP install qwen-vl-utils mathruler matplotlib flask gym gym-sokoban gymnasium "gymnasium[toy-text]" together openai
$PIP install --no-deps -e "$VAGEN"

echo "== check"
$PIP check || true
$PY -c "import torch, vllm, transformers, tensordict, verl, vagen, openai, xgrammar; print('torch', torch.__version__, 'vllm', vllm.__version__, 'transformers', transformers.__version__, 'tensordict', tensordict.__version__, 'openai', openai.__version__, 'xgrammar', xgrammar.__version__)"
$PY -c "import flash_attn; print('flash_attn', flash_attn.__version__)"
echo "== DONE"
