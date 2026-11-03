"""Judge state-estimation score per turn, split by whether the perception parses."""
import json, sys
from diag_wm import parse, well_formed, tag

for path in sys.argv[1:]:
    groups = {"all parse": [], "mixed": [], "none parse": []}
    for line in open(path):
        r = json.loads(line)
        turns = r["output"].split("\nassistant\n")
        ok = [tag(t, "perception") is not None and well_formed(parse(tag(t, "perception"))) for t in turns]
        per_turn = r["state_estimation_reward"] / (0.5 * len(turns))
        key = "all parse" if all(ok) else "none parse" if not any(ok) else "mixed"
        groups[key].append(per_turn)
    step = path.split("/")[-1].split(".")[0]
    print(f"{path.split(chr(47))[-3][:28]:28s} step {step:>3}: " + "  ".join(
        f"{k} n={len(v):3d} judge={sum(v)/len(v):.2f}" if v else f"{k} n=  0" for k, v in groups.items()))
