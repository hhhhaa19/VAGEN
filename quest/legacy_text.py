"""Agent descriptions the legacy judge scored, from the run's wandb example tables.

Each table file holds every row logged so far, so only the last file of each kind is read.
Row ids are a logging counter, not training steps; rows are split into five equal parts
of the run.

Usage: python3 legacy_text.py <wandb run>/files/media/table
"""
import glob
import json
import sys
from collections import Counter, defaultdict

KINDS = ["correct_grounding", "incorrect_grounding", "correct_worldmodeling",
         "incorrect_worldmodeling", "parse_failed"]
MARK = "Input Text to Parse:"

parts = defaultdict(lambda: defaultdict(list))
last = 0
for kind in KINDS:
    path = max(glob.glob(f"{sys.argv[1]}/{kind}_examples_*.table.json"),
               key=lambda p: int(p.split("_")[-2]))
    for row in json.load(open(path))["data"]:
        last = max(last, row[0])
        for i in range(8):
            prompt = row[3 + 5 * i]
            if isinstance(prompt, str) and MARK in prompt:
                text = prompt.split(MARK)[-1].split("Objects to Look For")[0].strip()
                parts[row[0]][kind].append(text)

for part in range(5):
    print(f"== part {part + 1}/5 of the run")
    for kind in KINDS:
        texts = [t for rid, by in parts.items() if min(rid * 5 // (last + 1), 4) == part
                 for t in by[kind]]
        if texts:
            top, count = Counter(texts).most_common(1)[0]
            print(f"  {kind:24s} n={len(texts):3d} distinct={len(set(texts)):3d} "
                  f"top={count:3d}x: {top[:110]}")
