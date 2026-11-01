# VAGEN Sokoban training log

All VAGEN training and evaluation jobs for the CS496 pilot, on Northwestern Quest (one node,
4x H100 80 GB). Scripts are in this directory. Logs are in `/projects/p53316/logs/<name>_<job>.log`.
Last updated 2026-11-01.

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
| 5463385 | 10-26 | `test_kl` | paper config + fix | 3-step KL test | 0 | 6 min | - | failed |
| 5612726 | 10-27 | `test_kl` | `paper-run` | KL test passes with PR #121 | 3 | 9 min | - | passed |
| 5614549 | 10-27 | `train_paper_full` | `paper-run` | + rollout gpu_memory_utilization=0.4 to fit the judge | 401 | 6 h 16 min | 0.488 (400) | 0.488 |
| 5896123 | 10-29 | `train_paper_rerun200` | `paper-run` | Same config, 200 steps, checkpoints every 50 | 200 | 3 h 07 min | 0.434 (180) | 0.422 |
| 6312383 | 11-01 | `train_legacy` | legacy | Smoke test, local Qwen judge, 128 seen-overlap test levels | 2 | 11 min | 0.102 | 0.102 |
| 6734879 | 11-01 | `train_legacy` | legacy | Full baseline, local Qwen judge, 12 h limit | 300 | queued | - | - |

## Evaluation runs

| Job | Date | Script | Result |
|---|---|---|---|
| 3951917 | 10-16 | `eval_zeroshot` | Untrained model through the evaluation script |
| 5834258 | 10-28 | `eval_protocol` | Step-400 model from 5614549: 0.570 on 128 seen levels, 0.488 on 256 held-out maps. Sampling and greedy give the same result |

## Decisions and findings

- 2026-10-27: PR #121 lets the KL penalty start. Without it a fused reference policy raises NotImplementedError.
- 2026-10-29: Checkpoints from 5614549 other than step 400 were deleted, so pre-drift checkpoints from that run no longer exist.
- 2026-11-01: Rerun 5896123 is not bitwise identical to 5614549 despite full determinism flags.
- 2026-11-01: Perception text leaves the required vocabulary at about step 40 in 5896123 and step 220 in 5614549, and both reach 0.422 at step 200. Parseable rates at that step are 0.08 and 0.94.
- 2026-11-01: The authors state in issue #106 that the paper's experiments ran on the legacy code, so runs on main are not a direct comparison with 0.79.

## Legacy run setup

`train_legacy.sbatch` follows `scripts/examples/vagen_full/sokoban/run_tmux.sh`. Of 61 training flags,
55 are unchanged. The other 6 are data paths, wandb names, the server URL and the step count.
Differences from a strict baseline:

- The judge is local Qwen3-4B instead of GPT-4.1-nano. Use `JUDGE=openai` for the original judge.
- xgrammar is 0.1.17, since 0.1.16 is no longer on PyPI.
- setuptools is below 80, so `gym_sokoban` can import `pkg_resources`.

The first legacy training step took 131 s, against about 56 s per step on main.
