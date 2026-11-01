"""Render a validation level and replay the actions a checkpoint took on it.

bash /projects/p53316/run.sh python render_case.py <seed> <out_dir> <name>=<a1,a2,...> ...
Environment settings follow examples/train/sokoban/val_sokoban_vision_sr_paper.yaml.
"""
import asyncio, os, sys
from vagen.envs.sokoban.sokoban_env import Sokoban

CFG = {"render_mode": "vision", "min_solution_steps": [1, 5], "map_partition": "eval",
       "map_partition_modulus": 4, "map_partition_eval_bucket": 0, "strict_format": False,  # replay passes a bare <answer>
       "format_reward": 0.5, "use_env_reward": True}

async def replay(seed, out, name, actions):
    env = Sokoban(dict(CFG))
    obs, _ = await env.reset(seed=seed)
    frames = [obs["multi_modal_input"][env.config.image_placeholder][0]]
    for a in actions:
        obs, reward, done, info = await env.step(f"<answer>{a}</answer>")
        frames.append(obs["multi_modal_input"][env.config.image_placeholder][0])
        if done:
            break
    for i, img in enumerate(frames):
        img.save(os.path.join(out, f"{seed}_{name}_{i}.png"))
    print(name, "frames", len(frames), "solved", bool(done) if actions else None)
    await env.close()

async def main():
    seed, out = int(sys.argv[1]), sys.argv[2]
    os.makedirs(out, exist_ok=True)
    for spec in sys.argv[3:]:
        name, _, acts = spec.partition("=")
        await replay(seed, out, name, [a for a in acts.split(",") if a])

asyncio.run(main())
