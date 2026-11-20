# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
A policy's plan clock (sequence_ai/engine/plan_clock.py PlanClock), the one serve() and a deployment's worker share:
each episode's plan begins every every_s after its last began, however many episodes share the server; one that runs
past its tick is followed by the next the moment it returns, never two of one episode at once; an episode is planned
only on an observation its last plan did not have; what plan returns is appended as it is to episode.state["seq.plans"]
with the observation's number, None appending nothing; a plan that raises is noted and called again at the next tick;
an ended episode is planned no more and what its plan in flight returns is dropped (an async one cancelled); a plain
plan on a thread of its own, an async one on the clock's loop, neither on the thread that observes; a clean stop.

    python -m pytest -q tests/engine/test_plan_clock.py
"""
from __future__ import annotations

import asyncio
import threading
import time

from sequence_ai.engine.plan_clock import PLANS, PlanClock, drain_events


class Episode:
    """An episode as the clock reads one: its id and its state (a deployment's seq.Episode has more)."""

    def __init__(self, id: str) -> None:  # noqa: A002 - the episode's id
        self.id, self.state = id, {}


class _Policy:
    """What the clock drives (a loaded policy's plan surface): the episodes the server holds, and a plan
    each call of which takes `latency` seconds and returns `answer(episode, n)` - n its call's number, from 1 -
    recording when it began and ended, for which episode and on which thread, and each episode's plan as it begins
    (`started`: a plan the planner began, under way). `is_async`: an async def plan."""

    plan_every_s = 0.2

    def __init__(self, latency: float = 0.0, answer=None, *, is_async: bool = False) -> None:
        self.latency = latency
        self.answer = answer or (lambda episode, n: {"n": n})
        self.plan_is_async = is_async
        self.episodes: dict[str, Episode] = {}
        self.calls: list[tuple[float, float, str, str]] = []
        self.started: list[str] = []
        self.cancelled: list[str] = []
        self.lock = threading.Lock()
        self.n = 0

    def hold(self, *ids: str) -> None:
        for i in ids:
            self.episodes[i] = Episode(i)

    def plan_episode(self, episode_id: str):
        return self.episodes.get(episode_id)

    def _answer(self, episode, began: float):
        with self.lock:
            self.n += 1
            n = self.n
        try:
            return self.answer(episode, n)
        finally:
            with self.lock:
                self.calls.append((began, time.monotonic(), episode.id, threading.current_thread().name))

    def _began(self, episode) -> float:
        with self.lock:
            self.started.append(episode.id)
        return time.monotonic()

    def plan(self, episode):
        if self.plan_is_async:
            return self._plan_async(episode)
        began = self._began(episode)
        time.sleep(self.latency)
        return self._answer(episode, began)

    async def _plan_async(self, episode):
        began = self._began(episode)              # its task runs: the planner's loop got to it
        try:
            await asyncio.sleep(self.latency)
        except asyncio.CancelledError:
            self.cancelled.append(episode.id)
            raise
        return self._answer(episode, began)


def _feed(planner: PlanClock, episodes: list[str], seconds: float, until=None) -> None:
    """The episodes' robots observing for `seconds` - or until `until()` says enough - every 5 ms each, the planner
    started first and stopped after."""
    stop = threading.Event()

    def feed() -> None:
        n = 0
        while not stop.is_set():
            for e in episodes:
                planner.observe(e, n)
            n += 1
            time.sleep(0.005)

    feeder = threading.Thread(target=feed, daemon=True)
    planner.start()
    feeder.start()
    try:
        end = time.monotonic() + seconds
        while time.monotonic() < end and not (until and until()):
            time.sleep(0.005)
    finally:
        stop.set()
        feeder.join()
        planner.stop(grace=5.0)


def _starts(calls, episode: str) -> list[float]:
    return sorted(began for began, _, e, _ in calls if e == episode)


def _until(cond, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.005)
    return cond()


def test_each_episode_is_planned_every_s_from_its_last_plans_start_however_many_share_the_server():
    """With 1, 2 and 4 episodes on one server, each episode's plans begin every_s apart - start to start, so a plan a
    third of every_s long is planned every 3, where the clock of before (every_s after the last answer) gave 4 - and the
    gap does not grow with the episodes beside it: their plans run at once, each on a thread of its own. On a clock the
    test moves: real gaps read past their margin on a busy machine (0.387 s for every_s 0.3 s, with the suite running
    twice over on 12 cores)."""
    every_s, plan_s = 3, 1
    for n in (1, 2, 4):
        now = [0]
        beside, ended = threading.Barrier(n), threading.Event()

        def answer(episode, call, beside=beside, ended=ended):
            beside.wait(30)                         # every episode's plan under way at once
            ended.wait(30)                          # until the clock has moved by its time
            return {"n": call}

        policy = _Policy(answer=answer)
        episodes = [f"e{i}" for i in range(n)]
        policy.hold(*episodes)
        planner = PlanClock(policy, every_s=every_s, clock=lambda now=now: now[0])
        starts: dict[str, list[int]] = {e: [] for e in episodes}
        for t in range(19):
            now[0] = t
            for e in episodes:
                planner.observe(e, t)
            planner._schedule(t)
            begun = [e for e in episodes if planner._episodes[e].busy]
            for e in begun:
                starts[e].append(t)
            now[0] = t + plan_s
            ended.set()
            for e in begun:
                _settled(planner, e)
            ended.clear()
        assert all(got == list(range(0, 19, every_s)) for got in starts.values()), (n, starts)


def test_a_plan_that_runs_past_its_tick_is_followed_by_the_next_the_moment_it_returns():
    """A plan longer than every_s: never two of the episode at once, and the next begins as the last returns. On a clock
    only the plans move, each two ticks long, a tick an hour: a next plan that waited for any tick would be an hour off,
    however busy the machine (a real 30 ms bound read 79 ms on a loaded machine running the suite across 12
    processes)."""
    now = [0.0]

    def answer(episode, n):
        now[0] += 7200.0
        return {"n": n}

    policy = _Policy(answer=answer)
    policy.hold("a")
    planner = PlanClock(policy, every_s=3600.0, forget_after_s=1e7, clock=lambda: now[0])
    _feed(planner, ["a"], 30.0, until=lambda: len(policy.calls) >= 4)
    calls = sorted(c for c in policy.calls if c[2] == "a")
    assert len(calls) >= 4
    for (_, ended, _, _), (began, _, _, _) in zip(calls, calls[1:]):
        assert began >= ended                       # never two at once


def test_what_plan_returns_is_kept_as_it_is_with_its_observations_number_and_none_keeps_nothing():
    """Odd calls return an object, even ones None: the episode's plans hold the objects alone, in order, each the very
    object returned, with the number of the episode's latest observation when its plan began."""
    drain_events()                                  # what the tests before noted
    returned = []

    def answer(episode, n):
        if n % 2 == 0:
            return None
        got = {"instruction": f"step {n}"}
        returned.append(got)
        return got

    policy = _Policy(answer=answer)
    policy.hold("a")
    planner = PlanClock(policy, every_s=0.02)
    planner.start()
    try:
        for step in range(6):
            planner.observe("a", step)
            assert _until(lambda step=step: policy.n == step + 1)
            time.sleep(0.03)
    finally:
        planner.stop(grace=2.0)
    plans = policy.episodes["a"].state[PLANS]
    assert [p["value"] for p in plans] == returned and all(p["value"] is r for p, r in zip(plans, returned))
    assert [p["step"] for p in plans] == [0, 2, 4]
    done = [ev for ev in drain_events() if ev["ev"] == "plan.done" and ev["env"] == "a"]
    assert [ev["step"] for ev in done] == ["{'instruction': 'step 1'}", None, "{'instruction': 'step 3'}", None,
                                           "{'instruction': 'step 5'}", None]


def test_an_episode_is_planned_only_on_an_observation_its_last_plan_did_not_have():
    policy = _Policy()
    policy.hold("a")
    planner = PlanClock(policy, every_s=0.02)
    planner.start()
    try:
        planner.observe("a", 0)
        assert _until(lambda: policy.n == 1)
        time.sleep(0.2)                             # ten ticks, no new observation
        assert policy.n == 1
        planner.observe("a", 1)
        assert _until(lambda: policy.n == 2)
    finally:
        planner.stop(grace=2.0)


def test_a_plan_that_raises_is_noted_and_called_again_at_the_next_tick_and_the_plans_stay():
    drain_events()

    def answer(episode, n):
        if n == 2:
            raise RuntimeError("the model said nothing sk-or-SECRET")
        return f"plan {n}"

    policy = _Policy(answer=answer)
    policy.hold("a")
    planner = PlanClock(policy, every_s=0.02)
    planner.start()
    try:
        for step in range(3):
            planner.observe("a", step)
            assert _until(lambda step=step: policy.n == step + 1)
            time.sleep(0.03)
    finally:
        planner.stop(grace=2.0)
    assert [p["value"] for p in policy.episodes["a"].state[PLANS]] == ["plan 1", "plan 3"]
    (error,) = [ev for ev in drain_events() if ev["ev"] == "plan.error"]
    assert error["env"] == "a" and error["kind"] == "transient"
    assert error["error"].startswith("RuntimeError: the model said nothing")


def test_an_ended_episode_is_planned_no_more_and_what_its_plan_in_flight_returns_is_dropped():
    drain_events()
    policy = _Policy(latency=0.2)
    policy.hold("a")
    planner = PlanClock(policy, every_s=0.01)
    planner.start()
    try:
        planner.observe("a", 0)
        assert _until(lambda: planner.in_flight == 1)
        planner.forget("a")                         # its end, mid-plan
        assert _until(lambda: planner.in_flight == 0)
        planner.observe("b", 0)                     # another episode: the planner goes on
        time.sleep(0.1)
    finally:
        planner.stop(grace=2.0)
    assert policy.n == 1 and PLANS not in policy.episodes["a"].state      # its one plan ran out, and nothing kept
    assert not [ev for ev in drain_events() if ev.get("env") == "a"]


def test_an_async_plan_runs_on_the_planners_loop_and_is_cancelled_when_its_episode_ends():
    policy = _Policy(latency=0.05, is_async=True)
    policy.hold("a", "b")
    planner = PlanClock(policy, every_s=0.01)
    planner.start()
    try:
        planner.observe("a", 0)
        assert _until(lambda: policy.n == 1)
        policy.latency = 5.0
        planner.observe("a", 1)
        # its next plan under way on the loop - not only counted in flight, which it is from the moment the planner
        # hands it to the loop: an end before the loop gets to it starts no plan, so cancels none (PlanClock._begin)
        assert _until(lambda: policy.started.count("a") == 2)
        planner.forget("a")
        assert _until(lambda: policy.cancelled == ["a"]) and _until(lambda: planner.in_flight == 0)
    finally:
        planner.stop(grace=2.0)
    assert [p["value"] for p in policy.episodes["a"].state[PLANS]] == [{"n": 1}]
    assert {thread for _, _, _, thread in policy.calls} == {"seq-plan-loop"}


def test_a_plain_plan_runs_on_a_thread_of_its_own_never_the_observing_one():
    policy = _Policy()
    policy.hold("a", "b")
    planner = PlanClock(policy, every_s=0.01)
    planner.start()
    try:
        planner.observe("a", 0)
        planner.observe("b", 0)
        assert _until(lambda: policy.n == 2)
    finally:
        planner.stop(grace=2.0)
    assert {thread for _, _, _, thread in policy.calls} == {"seq-plan"}


def test_stop_waits_for_the_plans_in_flight_drops_what_they_return_and_cancels_the_async_ones():
    for is_async in (False, True):
        policy = _Policy(latency=0.3, is_async=is_async)
        policy.hold("a")
        planner = PlanClock(policy, every_s=0.01)
        planner.start()
        planner.observe("a", 0)
        # its plan under way - a stop before the planner's loop gets to an async one starts none, and so cancels none
        assert _until(lambda policy=policy: policy.started == ["a"])
        t0 = time.monotonic()
        planner.stop(grace=2.0)
        took = time.monotonic() - t0
        assert planner.in_flight == 0 and PLANS not in policy.episodes["a"].state
        if is_async:
            assert policy.cancelled == ["a"] and took < 0.2       # cancelled, not waited out
        else:
            assert 0.1 < took < 1.0                                 # the plain one ran out, its answer dropped


def test_an_episode_none_of_whose_robots_observations_came_for_forget_after_s_is_planned_no_more():
    now = [0.0]
    policy = _Policy()
    policy.hold("a")
    planner = PlanClock(policy, every_s=0.01, forget_after_s=10.0, clock=lambda: now[0])
    planner.observe("a", 0)
    with planner._cv:
        planner._schedule(now[0])                   # its plan at the first tick
    assert _until(lambda: planner.in_flight == 0)
    now[0] = 10.0
    with planner._cv:
        planner._schedule(now[0])
        assert "a" not in planner._episodes


def test_an_episode_the_server_does_not_hold_is_not_planned_and_said_nothing_of():
    drain_events()
    policy = _Policy()
    planner = PlanClock(policy, every_s=0.01)
    planner.start()
    try:
        planner.observe("nowhere", 0)
        assert _until(lambda: planner.plan_calls == 1) and _until(lambda: planner.in_flight == 0)
    finally:
        planner.stop(grace=2.0)
    assert policy.n == 0 and not drain_events()


def test_a_state_that_keeps_no_list_under_seq_plans_is_said_and_planning_goes_on():
    drain_events()
    policy = _Policy()
    policy.hold("a")
    policy.episodes["a"].state[PLANS] = "the policy's own"
    planner = PlanClock(policy, every_s=0.01)
    planner.start()
    try:
        planner.observe("a", 0)
        assert _until(lambda: policy.n == 1) and _until(lambda: planner.in_flight == 0)
    finally:
        planner.stop(grace=2.0)
    (error,) = [ev for ev in drain_events() if ev["ev"] == "plan.error"]
    assert error["error"].startswith("AttributeError")


def test_what_a_plan_returned_is_said_short_however_large_it_is():
    drain_events()
    policy = _Policy(answer=lambda episode, n: {"instruction": "x" * 10_000, "tools": list(range(10_000))})
    policy.hold("a")
    planner = PlanClock(policy, every_s=0.01)
    planner.start()
    try:
        planner.observe("a", 0)
        assert _until(lambda: policy.n == 1) and _until(lambda: planner.in_flight == 0)
    finally:
        planner.stop(grace=2.0)
    (done,) = [ev for ev in drain_events() if ev["ev"] == "plan.done"]
    assert len(done["step"]) <= 120 and done["step"].startswith("{'instruction': 'xxx")


def test_an_observation_that_holds_a_wake_on_key_plans_at_once_and_one_woken_mid_plan_once_right_after():
    """wake_on: an observation that holds one of its keys plans its episode at once, without waiting for the clock;
    woken three times while a plan runs, the next plan begins the moment that one returns - once - and then the clock
    again, however many woke it meanwhile."""
    policy = _Policy(latency=0.15)
    policy.hold("a")
    planner = PlanClock(policy, every_s=30.0)       # a clock that never comes round in the test
    planner.start()
    try:
        planner.observe("a", 0)                         # a first observation plans at once anyway
        assert _until(lambda: policy.n == 1)
        planner.observe("a", 1)                         # an ordinary one waits for the clock
        time.sleep(0.3)
        assert policy.n == 1
        planner.observe("a", 2, wake=True)              # one that holds a wake_on key: at once
        assert _until(lambda: planner.in_flight == 1)
        for step in (3, 4, 5):
            planner.observe("a", step, wake=True)       # woken three times mid-plan
        assert _until(lambda: policy.n == 3, timeout=2.0)
        time.sleep(0.4)
        assert policy.n == 3                            # the woken plan and one right after it; then the clock
    finally:
        planner.stop(grace=2.0)
    assert [p["step"] for p in policy.episodes["a"].state[PLANS]] == [0, 2, 5]


def test_a_new_clock_takes_effect_at_once_for_every_episode():
    """set_every_s (a deployment's plan.every_s changed with no new deploy): an episode waiting on
    the old clock is due on the new one, counted from its last plan's start."""
    policy = _Policy()
    policy.hold("a")
    planner = PlanClock(policy, every_s=30.0)
    planner.start()
    try:
        planner.observe("a", 0)
        assert _until(lambda: policy.n == 1)
        planner.observe("a", 1)                         # due 30 s after the first began
        time.sleep(0.1)
        assert policy.n == 1
        planner.set_every_s(0.05)                       # now due 0.05 s after it: past already
        assert _until(lambda: policy.n == 2, timeout=1.0)
    finally:
        planner.stop(grace=2.0)
    for bad in (0, -1, "2", True, None):
        try:
            planner.set_every_s(bad)
        except ValueError:
            continue
        raise AssertionError(f"set_every_s({bad!r}) was taken")


def _settled(planner: PlanClock, episode_id: str) -> None:
    deadline = time.monotonic() + 5.0
    while planner._episodes[episode_id].busy and time.monotonic() < deadline:
        time.sleep(0.005)
