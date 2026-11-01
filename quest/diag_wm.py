"""Offline diagnostic of perception/prediction text in saved validation rollouts.

Usage: python3 diag_wm.py <validation .jsonl>
Parsing follows vagen/envs/sokoban/state_reward_spec.py:exact_relations (copied, no numpy).
"""
import json, re, sys
from collections import Counter

VERT = (("above", "above"), ("below", "below"), ("same row", "same"))
HORZ = (("left", "left"), ("right", "right"), ("same column", "same"))
OBJ = re.compile(
    r"(?:the\s+|a\s+|another\s+)?(?P<object>box|target)(?:\s*\d+)?\b(?P<clause>.*?)"
    r"(?=(?:,?\s+and\s+)?(?:the\s+|a\s+|another\s+)?(?:box|target)(?:\s*\d+)?\b|$)")

def one(clause, vocab):
    found = {v for p, v in vocab if p in clause}
    return next(iter(found)) if len(found) == 1 else None

def parse(text):
    t = re.sub(r"\s+", " ", text).strip().lower()
    return {m.group("object"): (one(m.group("clause"), VERT), one(m.group("clause"), HORZ))
            for m in OBJ.finditer(t)}

def well_formed(rel):
    return set(rel) == {"box", "target"} and all(None not in v for v in rel.values())

def tag(turn, name):
    m = re.search(rf"<{name}>(.*?)</{name}>", turn, re.S)
    return m.group(1) if m else None

if __name__ == "__main__":
    rows = [json.loads(l) for l in open(sys.argv[1])]
    stats = {True: Counter(), False: Counter()}
    for r in rows:
        ok = bool(r["traj_success"])
        turns = r["output"].split("\nassistant\n")
        s = stats[ok]
        s["episodes"] += 1
        prev_pred = None
        preds = []
        for turn in turns:
            per, pred = tag(turn, "perception"), tag(turn, "prediction")
            s["turns"] += 1
            if re.search(r"[^\x00-\x7f]", turn):
                s["turn_non_ascii"] += 1
            if per is not None and "behind" in per.lower():
                s["perception_behind"] += 1
            if per is not None and well_formed(parse(per)):
                s["perception_wellformed"] += 1
            if pred is not None and well_formed(parse(pred)):
                s["prediction_wellformed"] += 1
            if prev_pred is not None and per is not None:
                a, b = parse(prev_pred), parse(per)
                if well_formed(a) and well_formed(b):
                    s["pairs_comparable"] += 1
                    s["pairs_mismatch"] += a != b
            preds.append(pred)
            prev_pred = pred
        if len(preds) > 1 and len(set(preds)) == 1:
            s["episodes_all_predictions_identical"] += 1
        s["state_est_sum"] += r.get("state_estimation_reward") or 0
        s["trans_pred_sum"] += r.get("transition_prediction_reward") or 0

    for ok in (True, False):
        s = stats[ok]
        n, t = s["episodes"], s["turns"]
        print(f"== {'success' if ok else 'failure'}: {n} episodes, {t} turns ({t/n:.2f}/ep)")
        for k in ("turn_non_ascii", "perception_behind", "perception_wellformed", "prediction_wellformed"):
            print(f"  {k}: {s[k]}/{t} = {s[k]/t:.2f}")
        c = s["pairs_comparable"]
        print(f"  prev prediction vs perception comparable: {c}, mismatch {s['pairs_mismatch']} = {s['pairs_mismatch']/max(c,1):.2f}")
        print(f"  episodes with all predictions identical: {s['episodes_all_predictions_identical']}/{n}")
        print(f"  mean state_estimation_reward {s['state_est_sum']/n:.3f}, transition_prediction_reward {s['trans_pred_sum']/n:.3f}")
