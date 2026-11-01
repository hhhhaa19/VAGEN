"""Fake client and environment check of the verify harnesses: python -m vagen.harness.verify.test_verify"""
import asyncio
from types import SimpleNamespace

from vagen.harness.verify import GenericNoteHarness, VerifyHarness

def turn(perception, prediction, action="Up"):
    return (f"<perception>{perception}</perception><reasoning>r</reasoning>"
            f"<prediction>{prediction}</prediction><answer>{action}</answer>")

A = "The box is above and same column of the player, and the target is above and same column of the player"
B = "The box will be same row and left of the player, and the target will be below and left of the player"
B_SEEN = "The box is same row and left of the player, and the target is below and left of the player"
BEHIND = "The box is behind and left of the player, and the target is behind and left of the player"

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
    def __init__(self, turns):
        self.turns, self.steps, self.truncated = turns, [], None
    async def reset(self):
        return {"role": "user", "content": "obs0"}, {}
    async def system_prompt(self):
        return {"role": "system", "content": "sys"}
    async def step(self, response):
        self.steps.append(response.text)
        done = len(self.steps) >= self.turns
        return {"role": "user", "content": f"obs{len(self.steps)}"}, 0.0, done, False, {}
    def truncate(self, reason):
        self.truncated = reason

async def run(harness_cls, texts, turns):
    client, env = Client(texts), Env(turns)
    h = harness_cls(response_len=4000)
    await h.run_episode(client, env)
    return client, env

async def main():
    # D1: turn 2 perception B_SEEN matches turn 1 prediction B -> no retry.
    c, e = await run(VerifyHarness, [turn(A, B), turn(B_SEEN, B, "Left")], 2)
    assert len(c.calls) == 2 and e.steps == [turn(A, B), turn(B_SEEN, B, "Left")], "D1 match"
    # D1: turn 2 perception A contradicts prediction B -> one retry, only the retry is executed.
    c, e = await run(VerifyHarness, [turn(A, B), turn(A, B, "Down"), turn(B_SEEN, B, "Left")], 2)
    assert len(c.calls) == 3, "D1 mismatch retries once"
    last = c.calls[-1]
    assert last[-2]["role"] == "assistant" and last[-1]["role"] == "user"
    assert "not executed" in last[-1]["content"] and B in last[-1]["content"] and A in last[-1]["content"]
    assert e.steps == [turn(A, B), turn(B_SEEN, B, "Left")], "D1 executes only the retry"
    # D1: unparseable perception ("behind") -> no check, no retry.
    c, e = await run(VerifyHarness, [turn(A, B), turn(BEHIND, B)], 2)
    assert len(c.calls) == 2, "D1 skips unparseable"
    # G: retries every turn after the first, whatever the content.
    c, e = await run(GenericNoteHarness, [turn(A, B), turn(B_SEEN, B), turn(B_SEEN, B, "Left")], 2)
    assert len(c.calls) == 3 and "previous judgment" in c.calls[-1][-1]["content"], "G retries"
    # Budget: no room left -> truncated, not an exception.
    client, env = Client([turn(A, B)] * 5), Env(5)
    h = VerifyHarness(response_len=30)
    await h.run_episode(client, env)
    assert env.truncated == "no_room", env.truncated
    print("all verify harness checks passed")

asyncio.run(main())
