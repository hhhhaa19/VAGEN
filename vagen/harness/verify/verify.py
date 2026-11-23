"""Self-verification: check the last prediction against the new perception.

D1 (``VerifyHarness``). Each turn is generated exactly as in ``ConcatHarness``. If the
agent's ``<perception>`` this turn contradicts its ``<prediction>`` from the previous turn,
that response is not executed: it stays in the conversation, a user message stating both
descriptions follows, and the turn is generated once more. Only the second response
reaches the environment. When the two agree, or either cannot be parsed, the turn is
exactly what concat would have produced.

G (``GenericNoteHarness``). The same retry on every turn after the first, with a note that
names neither description. It controls for the second attempt and the extra text.

R (``NoRepeatHarness``). The same retry when the agent repeats exactly the actions that
just left the image unchanged, with a note stating that rule. It controls for a trivial
loop fix.

All keep one conversation per episode, so rows and metrics are assembled as for concat.
Select one with an import path, e.g. ``trainer.harness=vagen.harness.verify:VerifyHarness``.
Per-episode counts are appended as JSON lines to ``$VERIFY_LOG`` when it is set.
"""

from __future__ import annotations

import json
import os
import re

from vagen.envs.sokoban.state_reward_spec import exact_relations
from vagen.envs.sokoban.utils.utils import parse_response
from vagen.harness._common import assistant, obs_to_message, user
from vagen.harness.concat import ConcatHarness

D1_NOTE = (
    "Your response above was not executed. Last turn you predicted: \"{prediction}\". "
    "Now you perceive: \"{perception}\". These do not match. Look at the image again, "
    "then give your full response again in the required format."
)
G_NOTE = (
    "Your response above was not executed. Check your previous judgment against the "
    "current image, then give your full response again in the required format."
)


def section(text: str | None, name: str) -> str | None:
    match = re.search(rf"<{name}>(.*?)</{name}>", text or "", re.S)
    return match.group(1).strip() if match else None


def relations(text: str | None) -> dict | None:
    """Box and target relations to the player, or None unless both are fully stated."""
    if text is None:
        return None
    found = {}
    for item in exact_relations(text):
        pair = (item["vertical_relation"], item["horizontal_relation"])
        if None in pair or item["object_id"] in found:
            return None
        found[item["object_id"]] = pair
    return found if set(found) == {"box", "target"} else None


class VerifyHarness(ConcatHarness):
    """Retry a turn whose perception contradicts the previous prediction."""

    setting = "D1"

    def note(self, prediction: str | None, perception: str | None) -> str | None:
        """The message to send before a retry, or None to execute the response as is."""
        expected, seen = relations(prediction), relations(perception)
        if expected is None or seen is None:
            return None
        if expected == seen:
            return None
        return D1_NOTE.format(prediction=prediction, perception=perception)

    async def _generate(self, client, messages, spent):
        pending = 0 if not spent else client.size([messages[-1]])
        limit = self.generation_limit(self.response_len, self.floor, spent, pending)
        if limit == 0:
            return None, "no_room"
        response = await client.create(messages, **self.sampling(limit))
        if self.empty(response):
            return None, "empty_generation"
        return response, None

    async def run_episode(self, client, env) -> None:
        stats = {"setting": self.setting, "turns": 0, "checked": 0, "retries": 0}
        try:
            await self._run(client, env, stats)
        finally:
            path = os.environ.get("VERIFY_LOG")
            if path:
                with open(path, "a") as f:
                    f.write(json.dumps(stats) + "\n")

    async def _run(self, client, env, stats) -> None:
        observation, _info = await env.reset()
        messages = [await env.system_prompt(), obs_to_message(observation)]
        spent = 0
        previous_prediction = None

        while True:
            response, reason = await self._generate(client, messages, spent)
            if response is None:
                env.truncate(reason)
                return
            spent = response.usage.response_tokens

            if stats["turns"] > 0:
                perception = section(response.text, "perception")
                if relations(previous_prediction) and relations(perception):
                    stats["checked"] += 1
                note = self.note(previous_prediction, perception)
                if note is not None:
                    stats["retries"] += 1
                    messages.append(assistant(response.text))
                    messages.append(user(note))
                    response, reason = await self._generate(client, messages, spent)
                    if response is None:
                        env.truncate(reason)
                        return
                    spent = response.usage.response_tokens

            stats["turns"] += 1
            messages.append(assistant(response.text))
            previous_prediction = section(response.text, "prediction")
            observation, _reward, terminated, truncated, _info = await env.step(response)
            if terminated or truncated:
                return
            messages.append(obs_to_message(observation))


class GenericNoteHarness(VerifyHarness):
    """Retry every turn after the first with a note that carries no state."""

    setting = "G"

    def note(self, prediction, perception):
        return G_NOTE


R_NOTE = (
    "Your response above was not executed. Last turn your action(s) \"{actions}\" did not "
    "change the image. Do not repeat an action that had no effect. Choose a different "
    "action, then give your full response again in the required format."
)


def actions(text: str | None) -> list[str]:
    """The actions the environment would execute, or [] when the format is wrong."""
    parsed = parse_response(text or "", prompt_format="wm")
    return list(parsed["actions"]) if parsed.get("format_correct") else []


def same_image(before, after) -> bool | None:
    """Whether two observation messages show identical frames, or None without images."""
    first = list((before or {}).get("images") or [])
    second = list((after or {}).get("images") or [])
    if not first or len(first) != len(second):
        return None
    return all(
        a.size == b.size and a.mode == b.mode and a.tobytes() == b.tobytes()
        for a, b in zip(first, second)
    )


class NoRepeatHarness(VerifyHarness):
    """R: retry a turn that repeats exactly the actions which just left the image unchanged.

    It controls for a trivial loop fix. "No effect" is judged only from the two frames the
    agent saw, never from the true state. The retry and its single re-generation work as
    in D1, so B, G, R and D1 differ only in when a note is sent and what it says.
    """

    setting = "R"

    async def _run(self, client, env, stats) -> None:
        stats.update({"no_effect": 0, "unchecked": 0})
        observation, _info = await env.reset()
        current = obs_to_message(observation)
        messages = [await env.system_prompt(), current]
        spent = 0
        blocked: list[str] = []  # last turn's actions, if they left the image unchanged

        while True:
            response, reason = await self._generate(client, messages, spent)
            if response is None:
                env.truncate(reason)
                return
            spent = response.usage.response_tokens

            if blocked:
                stats["checked"] += 1
                if actions(response.text) == blocked:
                    stats["retries"] += 1
                    messages.append(assistant(response.text))
                    messages.append(user(R_NOTE.format(actions=", ".join(blocked))))
                    response, reason = await self._generate(client, messages, spent)
                    if response is None:
                        env.truncate(reason)
                        return
                    spent = response.usage.response_tokens

            stats["turns"] += 1
            messages.append(assistant(response.text))
            executed = actions(response.text)
            observation, _reward, terminated, truncated, _info = await env.step(response)
            if terminated or truncated:
                return
            previous, current = current, obs_to_message(observation)
            unchanged = same_image(previous, current)
            if unchanged is None:
                stats["unchecked"] += 1
            if unchanged and executed:
                stats["no_effect"] += 1
                blocked = executed
            else:
                blocked = []
            messages.append(current)
