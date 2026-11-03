# VAGEN Sokoban training log

All VAGEN training and evaluation jobs for the CS496 pilot, on Northwestern Quest (one node,
4x H100 80 GB). Scripts are in this directory. Logs are in `/projects/p53316/logs/<name>_<job>.log`.
Last updated 2026-11-03.

## Code versions

| Name | Commit | Contents |
|---|---|---|
| main base | `e34bbce` | upstream VAGEN main, verl 0.9.0, SGLang 0.5.13 |
| paper config | `3f4c4f8` | main base + paper reward yaml files (Table 8/17) |
| `paper-run` tag | `8769523` | paper config + PR #121 KL fix + seen-level val yaml |
| `verify` branch | `8db1064` | `paper-run` + self-verification harness (D1, G) |
| legacy | `9cfe7f0` | upstream `vagen-legacy` (head `4e2976d` only changes the README), verl `3f55021e`, vLLM 0.8.2 |

Upstream main is now at `41df5e7`; we have not moved to it.

## Training runs

Validation is 256 held-out maps with greedy decoding every 20 steps, unless noted.

| Job | Date | Script | Code | Changes from the previous setup | Steps | Time | Best val | Final val |
|---|---|---|---|---|---|---|---|---|
| 4047928 | 10-16 | `train_trial` | main | 10-step trial | - | 21 min | - | out of memory |
| 4115605 | 10-17 | `train_trial` | main | Same trial, rerun | 10 | 18 min | - | - |
| 4539759 | 10-20 | `train_full` | main | Script defaults: exact scorer, T=1.0, no KL. Main baseline | 401 | 6 h 16 min | 0.445 (180) | 0.219 |
| 4639622 | 10-20 | `train_paper` | paper config | Paper rewards (0.5/0.5, score_base 0), still exact scorer, T=1.0, no KL | 401 | 5 h 11 min | 0.480 | 0.406 |
| 4770153 | 10-21 | `train_paper_full` | paper config | + judge, T=0.7, top_p=0.95, KL 0.001. Crashed at start | 0 | 4 min | - | NotImplementedError |
| 5463385 | 10-26 | `test_kl` | not recorded | 3-step KL test | 0 | 6 min | - | failed |
| 5612726 | 10-27 | `test_kl` | `paper-run` | KL test passes with PR #121 | 3 | 9 min | - | passed |
| 5614549 | 10-27 | `train_paper_full` | `paper-run` | + rollout gpu_memory_utilization=0.4 to fit the judge | 401 | 6 h 16 min | 0.488 (400) | 0.488 |
| 5896123 | 10-29 | `train_paper_rerun200` | `paper-run` | Same config, 200 steps, checkpoints every 50 | 200 | 3 h 07 min | 0.434 (180) | 0.422 |
| 6312383 | 11-01 | `train_legacy` | legacy | Smoke test, local Qwen judge, 128 seen-overlap test levels | 2 | 11 min | 0.102 | 0.102 |
| 6734879 | 11-03 | `train_legacy` | legacy | Full baseline, local Qwen judge, 12 h limit. Validation is legacy test.parquet: 128 training levels | 300 | 9 h 16 min | 0.742 (300) | 0.742 |

## wandb runs

Team `cs496_vagen`. Open with a team account.

| Job | Run | wandb |
|---|---|---|
| 4115605 | Trial | [22rb1ibe](https://wandb.ai/cs496_vagen/vagen-trial/runs/22rb1ibe) |
| 4539759 | Main baseline | [tk9f26l3](https://wandb.ai/cs496_vagen/vagen-experiments/runs/tk9f26l3) |
| 4639622 | Paper rewards, exact scorer, no KL | [jkuvuc4e](https://wandb.ai/cs496_vagen/vagen-experiments/runs/jkuvuc4e) |
| 5612726 | KL test | [h1say55j](https://wandb.ai/cs496_vagen/vagen-experiments/runs/h1say55j) |
| 5614549 | Paper config + KL | [6fxjbrms](https://wandb.ai/cs496_vagen/vagen-experiments/runs/6fxjbrms) |
| 5896123 | Rerun to 200 | [e87c4bjb](https://wandb.ai/cs496_vagen/vagen-experiments/runs/e87c4bjb) |
| 6312383 | Legacy smoke test | Offline: `offline-run-20261101_022335-fifgq4ip` |
| 6734879 | Legacy baseline | [5037wh3b](https://wandb.ai/cs496_vagen/vagen-legacy/runs/5037wh3b) |

Job 5834258 logs to the console only.

## Evaluation runs

| Job | Date | Script | Result |
|---|---|---|---|
| 3951917 | 10-16 | `eval_zeroshot` | Untrained model through the evaluation script: 0.164 success over 128 episodes |
| 5834258 | 10-28 | `eval_protocol` | Step-400 model from 5614549: 0.570 on 128 seen levels, 0.488 on 256 held-out maps. Sampling and greedy give the same result |
| 7041474 | 11-03 | `eval_legacy` | Legacy steps 300 and 150 on 256 maps that no legacy training room equals (`make_heldout.py`), plus step 300 on test.parquet as a check. Queued |

## Legacy baseline against the authors' curve

`val/success` on the same 128 legacy test levels, which are training levels. Authors: run
`state_reward_finegrained-sokoban-grounding_worldmodeling` in their public report, read from tooltips.

| Step | 0 | 20 | 40 | 80 | 120 | 160 | 180 | 200 | 240 | 260 | 280 | 300 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Authors | 0.125 | 0.156 | 0.273 | 0.547 | 0.555 | 0.688 | 0.734 | 0.750 | 0.758 | 0.789 | 0.758 | - |
| 6734879 | 0.070 | 0.141 | 0.117 | 0.258 | 0.258 | 0.328 | 0.539 | 0.625 | 0.656 | 0.672 | 0.727 | 0.742 |

Steps 60 to 120 of 6734879 are all 0.258 (33/128).

## Decisions and findings

- 2026-10-27: PR #121 lets the KL penalty start. Without it a fused reference policy raises NotImplementedError.
- 2026-10-29: Checkpoints from 5614549 other than step 400 were deleted, so pre-drift checkpoints from that run no longer exist.
- 2026-10-29: Rerun 5896123 is not bitwise identical to 5614549 despite full determinism flags.
- 2026-11-01: Perception text leaves the required vocabulary at about step 40 in 5896123 and step 220 in 5614549, and both reach 0.422 at step 200. Parseable rates at that step are 0.08 and 0.94.
- 2026-11-01: The authors state in issue #106 that the paper's experiments ran on the legacy code, so runs on main are not a direct comparison with 0.79.
- 2026-11-02: `log_metrics.py` on the four main runs. The default script pays almost nothing for
  the description (weight 0.03 above a 0.625 base): its validation state reward stays at
  0.001-0.003, and after step 280 it outputs the same text on every map (success 0.219 and
  tokens per turn 98.964 at six validations in a row). In 5614549 the success drop at steps
  240-280 coincides with actor grad norm 28-31 and the largest KL. Tokens per turn stay at
  85-100 in every run, far below the 512 limit, so the response limit is not a cause.
- 2026-11-02: `judge_vs_parse.py`. In 5896123 the judge's state-estimation score rises from 0.47
  (step 0) to 0.89 (step 200) while the episodes whose perception parses fall from 256 to 0.
  Text such as "The box is left and left of the player" or mixed Chinese gets full score:
  the judge maps unparseable text to relations, so the model is rewarded for it.
- 2026-11-03: Main has no train/validation overlap: rooms are split by a SHA-256 hash of the
  map (`map_partition`), so a validation room cannot be a training room. Legacy test.parquet
  is the first 128 training seeds (1-162), so legacy numbers, including the paper's 0.79, are
  on training levels.
- 2026-11-03: Legacy baseline reaches 0.742, close to the authors' 0.758 at their last point,
  but about 100 steps later. `legacy_text.py`: legacy text stays English, with no mixed
  scripts, but collapses to a few templates in the second fifth of the run (e.g. 94 of 152
  world-model samples are "The player will move left, then down to push the box down to the
  target.", which the legacy judge scores as correct) and diversifies again later.
- 2026-11-03: A 50-room sample already holds a duplicate of one of 12 new candidates, so
  `make_heldout.py` checks candidates against all training rooms.
- 2026-11-03: Smoke-test checkpoint (86 GB) deleted.

## Legacy run setup

`train_legacy.sbatch` follows `scripts/examples/vagen_full/sokoban/run_tmux.sh`. Of 61 training flags,
55 are unchanged. The other 6 are data paths, wandb names, the server URL and the step count.
Differences from a strict baseline:

- The judge is local Qwen3-4B instead of GPT-4.1-nano. Use `JUDGE=openai` for the original judge.
- xgrammar is 0.1.17, since 0.1.16 is no longer on PyPI.
- setuptools is below 80, so `gym_sokoban` can import `pkg_resources`.

The first legacy training step took 131 s; the full run averaged about 110 s per step, against about 56 s on main.
