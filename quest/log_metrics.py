"""Per-window training and validation metrics from VAGEN console logs."""
import re, sys
from collections import defaultdict

TRAIN = {"ent": "actor/entropy", "len": "response_length/mean", "clip": "response_length/clip_ratio",
         "score": "custom_metrics/train/episode_score/mean", "turns": "custom_metrics/train/episode_turns/mean",
         "agn": "actor/grad_norm", "cgn": "critic/grad_norm", "ev": "critic/vf_explained_var",
         "pkl": "actor/ppo_kl", "pclip": "actor/pg_clipfrac", "klp": "actor/reward_kl_penalty"}
VAL = {"succ": "val-aux/sokoban/traj_success/mean@1", "se": "val-aux/sokoban/state_estimation_reward/mean@1",
       "tp": "val-aux/sokoban/transition_prediction_reward/mean@1", "fmt": "val-aux/sokoban/format_correct_rate/mean@1",
       "tok": "val-aux/sokoban/model_tokens_per_turn/mean@1"}
num = re.compile(r"(?:np\.\w+\()?(-?[\d.]+(?:e-?\d+)?)")

def parse(path):
    train, val = {}, {}
    for line in open(path, errors="ignore"):
        m = re.search(r"\bstep:(\d+) ", line)
        if not m: continue
        s = int(m[1]); kv = {}
        for tok in line.split():
            k, _, v = tok.partition(":")
            n = num.match(v)
            if n: kv[k] = float(n[1])
        train[s] = {a: kv.get(k) for a, k in TRAIN.items()}
        if VAL["succ"] in kv: val[s] = {a: kv.get(k) for a, k in VAL.items()}
    return train, val

for path in sys.argv[1:]:
    train, val = parse(path)
    print("##", path.split("/")[-1])
    print("step | " + " ".join(f"{a:>6}" for a in VAL) + " || " + " ".join(f"{a:>6}" for a in TRAIN))
    for s in sorted(val):
        win = [train[t] for t in range(max(1, s - 19), s + 1) if t in train]
        avg = {a: (sum(w[a] for w in win if w[a] is not None) / max(1, sum(w[a] is not None for w in win))) if win else None for a in TRAIN}
        f = lambda x: "  -   " if x is None else (f"{x:6.3f}" if abs(x) < 100 else f"{x:6.0f}")
        print(f"{s:4d} | " + " ".join(f(val[s][a]) for a in VAL) + " || " + " ".join(f(avg[a]) for a in TRAIN))
