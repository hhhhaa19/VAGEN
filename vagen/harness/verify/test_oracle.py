"""Fake client and environment check of O: python -m vagen.harness.verify.test_oracle"""
import asyncio
import itertools
from types import SimpleNamespace

import numpy as np

from vagen.harness.verify import OracleHarness
from vagen.harness.verify.verify import describe, relations, sokoban_of, true_state

def turn(perception, prediction, action="Up"):
    return (f"<perception>{perception}</perception><reasoning>r</reasoning>"
            f"<prediction>{prediction}</prediction><answer>{action}</answer>")

A = "The box is above and same column of the player, and the target is above and same column of the player"
A_PRED = "The box will be above and same column of the player, and the target will be above and same column of the player"
B = "The box will be same row and left of the player, and the target will be below and left of the player"
B_SEEN = "The box is same row and left of the player, and the target is below and left of the player"
BEHIND = "The box is behind and left of the player, and the target is behind and left of the player"

def room(player, boxes, target):
    """gym-sokoban encoding: 1 floor, 4 box, 5 player; room_fixed 2 marks the target."""
    state, fixed = np.ones((6, 6), dtype=int), np.ones((6, 6), dtype=int)
    state[player] = 5
    for box in boxes:
        state[box] = 4
    fixed[target] = 2
    return SimpleNamespace(room_state=state, room_fixed=fixed)

START = room((2, 2), [(4, 4)], (5, 5))
LAYOUT_B = room((2, 2), [(2, 1)], (3, 1))         # box same row & left, target below & left
TWO_BOXES = room((2, 2), [(2, 1), (4, 4)], (3, 1))

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

class Sokoban:
    """Stands in for the Sokoban env: ``.env`` is the gym env holding the room."""
    def __init__(self, rooms):
        self.rooms, self.steps, self.truncated = list(rooms), [], None
        self.env = self.rooms[0]
    async def reset(self, seed=None):
        return {"role": "user", "content": "obs0"}, {}
    async def system_prompt(self):
        return {"role": "system", "content": "sys"}
    async def step(self, response):
        self.steps.append(response.text)
        self.env = self.rooms[len(self.steps)]
        done = len(self.steps) >= len(self.rooms) - 1
        return {"role": "user", "content": f"obs{len(self.steps)}"}, 0.0, done, False, {}
    def truncate(self, reason):
        self.truncated = reason

class Layer:
    """A wrapper like the seam, adapter or state-reward wrapper, holding the env as ``attr``."""
    def __init__(self, attr, inner):
        self.attr = attr
        setattr(self, attr, inner)
    def _inner(self):
        return getattr(self, self.attr)
    async def reset(self, seed=None):
        return await self._inner().reset(seed)
    async def system_prompt(self):
        return await self._inner().system_prompt()
    async def step(self, response):
        return await self._inner().step(response)
    def truncate(self, reason):
        self._inner().truncate(reason)

async def run(texts, rooms, wrap=False):
    client, sokoban = Client(texts), Sokoban(rooms)
    env = Layer("env", Layer("inner", Layer("env", sokoban))) if wrap else sokoban
    harness = OracleHarness(response_len=4000)
    stats = {"setting": harness.setting, "turns": 0, "checked": 0, "retries": 0}
    await harness._run(client, env, stats)
    return client, sokoban, stats

async def main():
    # The truth sentence is written in the env's vocabulary and parses back exactly.
    for v1, h1, v2, h2 in itertools.product(("above", "below", "same"), ("left", "right", "same"), repeat=2):
        state = {"box": (v1, h1), "target": (v2, h2)}
        assert relations(describe(state)) == state, describe(state)

    # The Sokoban env is found under seam, adapter and state-reward layers.
    sokoban = Sokoban([START, LAYOUT_B])
    assert sokoban_of(Layer("env", Layer("inner", Layer("env", sokoban)))) is sokoban
    assert true_state(sokoban) == {"box": ("below", "right"), "target": ("below", "right")}
    assert true_state(Sokoban([TWO_BOXES])) is None, "O needs exactly one box and one target"

    # Prediction and perception both right -> no retry; D1 agrees.
    c, e, s = await run([turn(A, B), turn(B_SEEN, B, "Left")], [START, LAYOUT_B, LAYOUT_B])
    assert len(c.calls) == 2 and s["retries"] == 0 and s["checked"] == 1, s
    assert s["d1_comparable"] == 1 and s["d1_fires"] == 0, s

    # Wrong prediction, right perception -> retry; the note names the prediction and gives the truth.
    c, e, s = await run([turn(A, A_PRED), turn(B_SEEN, B), turn(B_SEEN, B, "Left")],
                        [START, LAYOUT_B, LAYOUT_B], wrap=True)
    assert len(c.calls) == 3 and s["retries"] == 1, s
    note = c.calls[-1][-1]["content"]
    assert c.calls[-1][-1]["role"] == "user" and "not executed" in note
    assert "This was wrong." in note and "This is correct." in note
    assert describe({"box": ("same", "left"), "target": ("below", "left")}) in note
    assert e.steps == [turn(A, A_PRED), turn(B_SEEN, B, "Left")], "O executes only the retry"
    assert s["prediction_wrong"] == 1 and s["perception_wrong"] == 0 and s["d1_fires"] == 1, s
    assert s["d1_false_alarm"] == 0 and s["d1_missed"] == 0, s

    # Right prediction, misperception -> D1 would raise a false alarm; O retries on the perception.
    c, e, s = await run([turn(A, B), turn(A, B), turn(B_SEEN, B, "Left")], [START, LAYOUT_B, LAYOUT_B])
    note = c.calls[-1][-1]["content"]
    assert s["retries"] == 1 and "This was correct." in note and "This is wrong." in note, note
    assert s["perception_wrong"] == 1 and s["d1_false_alarm"] == 1, s

    # Wrong prediction and the same misperception -> D1 sees a match and misses it; O does not.
    c, e, s = await run([turn(A, A_PRED), turn(A, A_PRED), turn(B_SEEN, B)], [START, LAYOUT_B, LAYOUT_B])
    assert s["retries"] == 1 and s["both_wrong"] == 1 and s["d1_missed"] == 1 and s["d1_fires"] == 0, s

    # Unparseable perception: the prediction alone is still judged.
    c, e, s = await run([turn(A, A_PRED), turn(BEHIND, B), turn(B_SEEN, B)], [START, LAYOUT_B, LAYOUT_B])
    assert s["retries"] == 1 and s["d1_comparable"] == 0 and "perceive" not in c.calls[-1][-1]["content"], s

    # Nothing parseable -> no check, no retry.
    c, e, s = await run([turn(BEHIND, "?"), turn(BEHIND, "?")], [START, LAYOUT_B, LAYOUT_B])
    assert len(c.calls) == 2 and s["checked"] == 0, s

    # Two boxes: no single true state -> counted, no retry.
    c, e, s = await run([turn(A, A_PRED), turn(A, A_PRED)], [START, TWO_BOXES, TWO_BOXES])
    assert len(c.calls) == 2 and s["no_truth"] == 1 and s["retries"] == 0, s

    # The retry is not checked again: one note per turn at most, as in D1.
    c, e, s = await run([turn(A, A_PRED)] * 5, [START, LAYOUT_B, LAYOUT_B, LAYOUT_B])
    assert len(c.calls) == 5 and s["retries"] == 2, s

    # Budget: no room left -> truncated, not an exception.
    client, env = Client([turn(A, A_PRED)] * 5), Sokoban([START] + [LAYOUT_B] * 5)
    await OracleHarness(response_len=30).run_episode(client, env)
    assert env.truncated == "no_room", env.truncated
    print("all oracle harness checks passed")

asyncio.run(main())
