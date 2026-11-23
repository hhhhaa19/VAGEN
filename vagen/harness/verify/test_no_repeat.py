"""Fake client and environment check of R: python -m vagen.harness.verify.test_no_repeat"""
import asyncio
from types import SimpleNamespace

from PIL import Image

from vagen.harness.verify import NoRepeatHarness

def turn(action):
    return (f"<perception>p</perception><reasoning>r</reasoning>"
            f"<prediction>q</prediction><answer>{action}</answer>")

def frame(position):
    """A 4x4 image whose one white pixel stands for the player."""
    image = Image.new("RGB", (4, 4))
    image.putpixel((position, 0), (255, 255, 255))
    return image

class Client:
    def __init__(self, texts):
        self.texts, self.calls, self.tokens = list(texts), [], 0
    def size(self, messages):
        return 10
    async def create(self, messages, **kwargs):
        self.calls.append([dict(m) for m in messages])
        text = self.texts.pop(0)
        self.tokens += 20
        return SimpleNamespace(text=text, token_ids=[1] * 20, usage=SimpleNamespace(response_tokens=self.tokens))

class Env:
    """Up hits a wall and changes nothing; Right moves the player one pixel."""
    def __init__(self, turns, images=True):
        self.turns, self.images, self.steps, self.position, self.truncated = turns, images, [], 0, None
    def obs(self):
        message = {"role": "user", "content": f"obs{len(self.steps)}"}
        if self.images:
            message["images"] = [frame(self.position)]
        return message
    async def reset(self):
        return self.obs(), {}
    async def system_prompt(self):
        return {"role": "system", "content": "sys"}
    async def step(self, response):
        self.steps.append(response.text)
        if "Right" in response.text:
            self.position += 1
        done = len(self.steps) >= self.turns
        return self.obs(), 0.0, done, False, {}
    def truncate(self, reason):
        self.truncated = reason

async def run(texts, turns, images=True):
    client, env = Client(texts), Env(turns, images)
    harness = NoRepeatHarness(response_len=4000)
    stats = {"setting": harness.setting, "turns": 0, "checked": 0, "retries": 0}
    await harness._run(client, env, stats)
    return client, env, stats

async def main():
    # Up has no effect, the agent repeats Up -> one retry with the rule; only the retry runs.
    c, e, s = await run([turn("Up"), turn("Up"), turn("Right")], 2)
    assert len(c.calls) == 3, "R retries a repeated no-effect action"
    note = c.calls[-1][-1]
    assert note["role"] == "user" and "did not change the image" in note["content"] and "up" in note["content"]
    assert e.steps == [turn("Up"), turn("Right")], "R executes only the retry"
    assert s["no_effect"] == 1 and s["checked"] == 1 and s["retries"] == 1, s
    # Up has no effect, the agent tries something else -> no retry.
    c, e, s = await run([turn("Up"), turn("Right")], 2)
    assert len(c.calls) == 2 and s["retries"] == 0, "R leaves a different action alone"
    # Right moves the player, the agent repeats Right -> repeating an effective action is fine.
    c, e, s = await run([turn("Right"), turn("Right")], 2)
    assert len(c.calls) == 2 and s["checked"] == 0, "R only fires after a no-effect turn"
    # Only an exact repeat counts: "Up, Right" after a blocked "Up" is not a repeat.
    c, e, s = await run([turn("Up"), turn("Up, Right")], 2)
    assert len(c.calls) == 2 and s["retries"] == 0, "R needs the identical action list"
    # A malformed turn executes nothing, so it cannot set up a rule.
    c, e, s = await run(["<answer>Up</answer>", turn("Up")], 2)
    assert len(c.calls) == 2 and s["no_effect"] == 0, "R ignores malformed turns"
    # The retry is not checked again: one note per turn at most, as in D1.
    c, e, s = await run([turn("Up")] * 5, 3)
    assert len(c.calls) == 5 and s["retries"] == 2, s
    # Text observations have no frames -> nothing is judged, the turn is counted as unchecked.
    c, e, s = await run([turn("Up"), turn("Up")], 2, images=False)
    assert len(c.calls) == 2 and s["unchecked"] == 1 and s["retries"] == 0, s
    # Budget: no room left -> truncated, not an exception.
    client, env = Client([turn("Up")] * 5), Env(5)
    harness = NoRepeatHarness(response_len=30)
    await harness.run_episode(client, env)
    assert env.truncated == "no_room", env.truncated
    print("all no-repeat harness checks passed")

asyncio.run(main())
