# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
A @seq.policy class as serve()'s copy of the policy (sequence_ai/engine/episodes.py PolicyCopy): each episode's frames
in order to its adapter, on the server's policy thread; an act's handle on its episode, which the adapter acts on; an
episode's end; its plans on their clock - a first observation planned at once, a key of wake_on at once between ticks,
the server's own episodes never; an act that carries its observation an episode of its own, begun and ended; a batch one
call of the adapter.

    python -m pytest -q tests/engine/test_episodes.py
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
import types

import pytest

pytest.importorskip("sequence_base")

from sequence_base import SeqFlags  # noqa: E402
from sequence_base.contract.policy import ImageSpec, PolicyRegistration, PolicySpec  # noqa: E402

from sequence_ai.engine import obs_format  # noqa: E402
from sequence_ai.engine.adapter import RegisteredAdapter  # noqa: E402
from sequence_ai.engine.episodes import PolicyCopy  # noqa: E402


class _Observes:
    """A class that keeps each episode in its seq.Episode: each frame its observe function saw, on which thread, and
    each end of its episodes."""

    def __init__(self) -> None:
        self.phases, self.threads = [], set()

    def on_episode(self, episode):
        self.phases.append((episode.id, episode.phase))
        self.threads.add(threading.current_thread().name)
        if episode.phase == "start":
            episode.state["seen_user"] = []

    def update(self, obs, episode):
        self.threads.add(threading.current_thread().name)
        episode.state["seen_user"].append(obs["x"])

    def act(self, episode):
        self.threads.add(threading.current_thread().name)
        return {"seen": list(episode.state["seen_user"])}


class _Latest:
    """A class whose act function reads its episode's latest observation, and plans: what the plan saw."""

    def __init__(self) -> None:
        self.planned = []

    def act(self, obs, episode):
        plans = episode.state.get("seq.plans")
        return {"x": obs["x"], "plan": plans[-1]["value"] if plans else None}

    def plan_user(self, robot):
        self.planned.append((robot.id, robot.state["seq.observations"][-1]["x"]))
        return f"plan of {robot.id}"


def _copy(cls, bindings, *, every_s=None, wake=None, executor=None) -> PolicyCopy:
    """`cls` registered as the SDK registers it, its adapter loaded, as serve()'s copy."""
    spec = PolicySpec(name="p", gpu="cpu", image=ImageSpec(base="python:3.11-slim"), episodes="shared")
    if wake:
        cls.plan_user.__seq_plan_wake_on__ = list(wake)
    reg = PolicyRegistration(spec=spec, handler_ref=f"{__name__}:{cls.__name__}", bindings=bindings,
                             plan_every_s=every_s)
    adapter = RegisteredAdapter(cls, types.SimpleNamespace(registration=reg, cls=cls))
    adapter.load(None)
    return PolicyCopy(adapter, executor=executor)


OBSERVES = {SeqFlags.INFER: "act", SeqFlags.OBSERVE: "update", SeqFlags.ON_EPISODE: "on_episode"}
PLANS = {SeqFlags.INFER: "act", SeqFlags.PLAN: "plan_user"}


def test_frames_go_to_the_adapter_in_order_on_the_policy_thread_and_an_act_reads_its_episode():
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy-thread")
    copy = _copy(_Observes, OBSERVES)
    copy.attach(pool)

    async def go():
        took = await copy.observe("e1", [(0, {"x": 1}), (1, {"x": 2})], lambda: None)
        answer = await asyncio.get_running_loop().run_in_executor(pool, copy.call, copy.latest("e1"))
        copy.end("e1")
        await asyncio.get_running_loop().run_in_executor(pool, lambda: None)      # its end ran before this
        return took, answer

    try:
        took, answer = asyncio.run(go())
    finally:
        pool.shutdown(wait=True)
    inst = copy._adapter._inst
    assert took == [{"ok": True}, {"ok": True}] and answer == {"seen": [1, 2]}
    assert inst.phases == [("e1", "start"), ("e1", "end")]
    assert {t.split("_")[0] for t in inst.threads} == {"policy-thread"}           # every call of its code there


def test_an_act_before_any_observation_of_a_policy_that_reads_the_latest_is_refused_as_the_data_plane_refuses():
    copy = _copy(_Latest, {SeqFlags.INFER: "act"})
    with pytest.raises(obs_format.NoObservation):
        copy.latest("e1")


def test_an_act_that_carries_its_observation_is_an_episode_of_its_own_begun_and_ended():
    copy = _copy(_Observes, OBSERVES)
    assert copy.call({"x": 7}) == {"seen": [7]}
    assert copy.call({"x": 8}) == {"seen": [8]}                                    # another, its own
    phases = copy._adapter._inst.phases
    assert [p for _, p in phases] == ["start", "end", "start", "end"]
    assert all(e.startswith("seq-once-") for e, _ in phases) and len({e for e, _ in phases}) == 2


def test_a_plan_runs_for_each_episode_at_once_on_its_first_observation_and_on_wake_on_never_for_the_servers_own():
    copy = _copy(_Latest, PLANS, every_s=60.0, wake=["say_user"])
    clock, inst = copy._clock, copy._adapter._inst

    def settle(calls):
        deadline = time.monotonic() + 5
        while not (clock.plan_calls >= calls and not clock.in_flight):
            assert time.monotonic() < deadline, "no plan within the wait"
            time.sleep(0.01)

    async def go():
        await copy.observe("seq-warm", [(0, {"x": 0})], lambda: None)            # the server's own: never planned
        await copy.observe("e1", [(0, {"x": 1})], lambda: None)                   # its first: at once
        settle(1)
        await copy.observe("e1", [(1, {"x": 2})], lambda: None)                   # its clock a minute away: nothing
        time.sleep(0.2)
        quiet = clock.plan_calls
        await copy.observe("e1", [(2, {"x": 3, "say_user": "go"})], lambda: None)  # wake_on: at once
        settle(2)
        return quiet, copy.call(copy.latest("e1"))

    try:
        quiet, answer = asyncio.run(go())
    finally:
        copy.close()
    assert quiet == 1 and inst.planned == [("e1", 1), ("e1", 3)]
    assert answer == {"x": 3, "plan": "plan of e1"}


def test_a_batch_is_one_call_of_the_adapter_each_act_its_own_episode():
    class _Batched:
        def act(self, obs, *, episode):
            return [{"id": e.id} for e in episode]

    _Batched.act.__seq_concurrent__ = {"sizes": [1, 2, 4], "collate": False}
    copy = _copy(_Batched, {SeqFlags.INFER: "act"})
    assert copy.batch == [1, 2, 4]

    async def go():
        for eid in ("a", "b"):
            await copy.observe(eid, [(0, {"x": eid})], lambda: None)
        return copy.call([copy.latest("a"), copy.latest("b")])

    answers = asyncio.run(go())
    assert answers == [{"id": "a"}, {"id": "b"}]


def test_each_plans_outcome_is_said_in_the_servers_log_one_that_raises_failing_no_act(caplog):
    class _Fails(_Latest):
        def plan_user(self, robot):
            raise RuntimeError("the model is down")

    copy = _copy(_Fails, PLANS, every_s=60.0)
    clock = copy._clock

    async def go():
        await copy.observe("e1", [(0, {"x": 1})], lambda: None)
        deadline = time.monotonic() + 5
        while not (clock.plan_calls and not clock.in_flight):
            assert time.monotonic() < deadline, "no plan within the wait"
            time.sleep(0.01)
        return copy.call(copy.latest("e1"))                    # the act after it: said there, and answered

    import logging

    try:
        with caplog.at_level(logging.INFO, logger="sequence_ai.serve"):
            answer = asyncio.run(go())
    finally:
        copy.close()
    assert answer == {"x": 1, "plan": None}
    said = [r.getMessage() for r in caplog.records if "plan of episode e1" in r.getMessage()]
    assert said and "raised (transient): RuntimeError: the model is down" in said[0]


def test_what_serve_calls_is_named_as_the_policys_own_act_function():
    """serve() names what it serves on its start line and in its ping's answer: the policy's class and act function, as
    a robot's author wrote them - never this copy's own method."""
    copy = _copy(_Observes, OBSERVES)
    assert (copy.call.__qualname__, copy.call.__name__) == ("_Observes.act", "act")
    assert copy.call({"x": 4}) == {"seen": [4]}                 # and it answers as before


def test_a_stacked_concurrent_infer_takes_an_act_alone_as_a_batch_of_one():
    """One shape, whatever path an act comes by: an act alone - a build check's sample, a robot's act - reaches a
    @seq.concurrent infer that takes its acts stacked as a batch of one, as the batcher hands it several, and its one
    row's answer comes back, so the infer reads one shape and never a lone observation beside it."""
    import numpy as np

    seen = []

    class _Stacked:
        def act(self, obs, *, seed):
            seen.append((np.shape(obs["state"]), list(seed)))
            return {"actions": np.zeros((len(obs["state"]), 2), np.float32)}

    _Stacked.act.__seq_concurrent__ = {"sizes": [1, 2], "collate": True}
    adapter = _copy(_Stacked, {SeqFlags.INFER: "act"})._adapter
    for episode_id in ("e1", "e2"):
        assert adapter.observe({"state": np.zeros(3, np.float32)}, episode_id=episode_id, seq=0) == {"ok": True}
    alone = adapter.infer(seed=7, episode_id="e1")
    assert seen == [((1, 3), [7])] and np.shape(alone["actions"]) == (2,)
    two = adapter.infer_batch(seeds=[1, 2], episode_ids=["e1", "e2"], params=[None, None])
    assert seen[1] == ((2, 3), [1, 2]) and [np.shape(a["actions"]) for a in two] == [(2,), (2,)]
