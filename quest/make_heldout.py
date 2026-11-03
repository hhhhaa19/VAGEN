"""A legacy Sokoban test set of maps the legacy training set never contains.

Legacy test.parquet reuses the first training seeds. This draws seeds after the legacy
candidate range instead, keeps the difficulty filter of legacy generate_seeds, and drops
any seed whose room (walls, boxes, targets, player) equals a training room, as the main
branch's map partition does.

Usage: python make_heldout.py <train.parquet> <test.parquet> <out.parquet> [size] [first_seed]
"""
import hashlib
import multiprocessing as mp
import sys

import numpy as np
import pandas as pd

from vagen.env.sokoban.env import SokobanEnv
from vagen.env.sokoban.env_config import SokobanEnvConfig
from vagen.env.sokoban.utils import process_seed

CONFIG = None


def init(env_config):
    global CONFIG
    CONFIG = SokobanEnvConfig(**env_config)


def room_hash(seed):
    env = SokobanEnv(CONFIG)
    env.reset(seed=seed)
    room = np.concatenate([np.asarray(env.env.room_fixed, "<i8"), np.asarray(env.env.room_state, "<i8")])
    return seed, hashlib.sha256(room.tobytes()).hexdigest()


def valid(seed):
    _, actions = process_seed(seed, CONFIG, CONFIG.min_actions_to_succeed)
    return seed, bool(actions)


if __name__ == "__main__":
    train_path, test_path, out_path = sys.argv[1:4]
    size = int(sys.argv[4]) if len(sys.argv) > 4 else 256
    first = int(sys.argv[5]) if len(sys.argv) > 5 else 20000
    train, test = pd.read_parquet(train_path), pd.read_parquet(test_path)
    env_config = dict(test.iloc[0]["extra_info"]["env_config"])
    with mp.Pool(mp.cpu_count(), initializer=init, initargs=(env_config,)) as pool:
        seeds = [int(x["seed"]) for x in train["extra_info"]]
        seen = {h for _, h in pool.imap(room_hash, seeds, chunksize=64)}
        chosen, duplicates, seed = [], 0, first
        while len(chosen) < size:
            batch = range(seed, seed + 4 * size)
            ok = [s for s, v in pool.imap(valid, batch, chunksize=16) if v]
            for s, h in pool.imap(room_hash, ok, chunksize=16):
                if h in seen:
                    duplicates += 1
                elif len(chosen) < size:
                    chosen.append(s)
            seed += 4 * size
    rows = []
    for s in chosen:
        row = test.iloc[0].to_dict()
        row["extra_info"] = {**row["extra_info"], "seed": s}
        rows.append(row)
    pd.DataFrame(rows).to_parquet(out_path)
    print(f"{len(seen)} distinct training rooms; held-out seeds {chosen[0]}..{chosen[-1]}; "
          f"{duplicates} candidates dropped as training duplicates; wrote {len(chosen)} to {out_path}")
