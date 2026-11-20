# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
The calls waiting for a model batched into one call to it (sequence_ai/engine/batcher.py): a full batch is dispatched at
once and a short one after its wait; the waiting calls are dispatched in the largest listed size they fill, and the
remainder as the next batch; while the model is busy they accumulate and are dispatched together as soon as it is
free; each call gets its own element back; a batch in which the model raised is re-run call by call, so only the call
that fails by itself fails; a plain function runs off the event loop; and, given `hold` and `first`, a batch waits as
long as hold says for its calls' keys, asked again as each call arrives and as the model frees, and the calls first
names go apart.

    python -m pytest -q tests/engine/test_batcher.py
"""
from __future__ import annotations

import asyncio
import threading
import time

import pytest

from sequence_ai.engine.batcher import Batcher


def _model(seen: list, *, gate: asyncio.Event | None = None, bad=()):
    async def run(items: list) -> list:
        seen.append(list(items))
        if gate is not None:
            await gate.wait()
        if any(item in bad for item in items):
            raise RuntimeError(f"cannot do {sorted(set(items) & set(bad))}")
        return [item * 10 for item in items]
    return run


def test_a_full_batch_is_dispatched_at_once_and_each_call_gets_its_own_output():
    async def go():
        seen: list = []
        batcher = Batcher(_model(seen), sizes=[1, 2, 4], wait_ms=10_000)       # a wait no test would sit through
        got = await asyncio.wait_for(asyncio.gather(*(batcher(i) for i in range(4))), 2.0)
        return seen, got, dict(batcher.sent)

    assert asyncio.run(go()) == ([[0, 1, 2, 3]], [0, 10, 20, 30], {4: 1})


def test_a_short_batch_is_dispatched_after_its_wait_in_the_largest_listed_size_it_fills_then_the_rest():
    async def go():
        seen: list = []
        batcher = Batcher(_model(seen), sizes=[1, 2, 4], wait_ms=20)
        started = time.perf_counter()
        got = await asyncio.gather(*(batcher(i) for i in range(3)))
        return seen, got, time.perf_counter() - started

    seen, got, took = asyncio.run(go())
    assert seen == [[0, 1], [2]] and got == [0, 10, 20]                # 3 calls of sizes [1, 2, 4]: 2, then 1
    assert took >= 0.02                                                 # it waited for more first


def test_while_the_model_is_busy_calls_accumulate_and_are_dispatched_together_once_it_is_free():
    async def go():
        seen: list = []
        gate = asyncio.Event()
        batcher = Batcher(_model(seen, gate=gate), sizes=[1, 2, 4, 8], wait_ms=1)
        first = asyncio.ensure_future(batcher(0))
        await asyncio.sleep(0.02)                                       # the first is in the model, alone
        rest = [asyncio.ensure_future(batcher(i)) for i in (1, 2, 3)]
        await asyncio.sleep(0.05)                                       # long past wait_ms: the model is still busy
        gathered = [list(batch) for batch in seen]
        gate.set()
        return gathered, seen, await asyncio.gather(first, *rest)

    gathered, seen, got = asyncio.run(go())
    assert gathered == [[0]]                                            # nothing was cut off behind the busy model
    assert seen == [[0], [1, 2], [3]] and got == [0, 10, 20, 30]


def test_two_runners_take_two_batches_at_once():
    async def go():
        seen: list = []
        gate = asyncio.Event()
        batcher = Batcher(_model(seen, gate=gate), sizes=[1, 2], wait_ms=1, runners=2)
        calls = [asyncio.ensure_future(batcher(i)) for i in range(4)]
        await asyncio.sleep(0.05)
        at_once = len(seen)
        gate.set()
        return at_once, await asyncio.gather(*calls)

    assert asyncio.run(go()) == (2, [0, 10, 20, 30])


def test_a_batch_the_model_raised_in_is_run_again_call_by_call_so_only_its_own_call_fails():
    async def go():
        seen: list = []
        batcher = Batcher(_model(seen, bad={2}), sizes=[1, 2, 4], wait_ms=10_000)
        got = await asyncio.gather(*(batcher(i) for i in range(4)), return_exceptions=True)
        return seen, got, batcher.reruns

    seen, got, reruns = asyncio.run(go())
    assert seen == [[0, 1, 2, 3], [0], [1], [2], [3]] and reruns == 1
    assert got[:2] == [0, 10] and got[3] == 30
    assert isinstance(got[2], RuntimeError) and "cannot do [2]" in str(got[2])


def test_a_call_that_fails_alone_raises_what_the_model_raised_and_a_wrong_count_is_an_error():
    async def go():
        alone = Batcher(_model([], bad={7}), wait_ms=0)
        with pytest.raises(RuntimeError, match="cannot do"):
            await alone(7)

        async def short(items):
            return items[:-1]

        with pytest.raises(ValueError, match="returned 0 outputs for a batch of 1"):
            await Batcher(short, wait_ms=0)(1)

    asyncio.run(go())


def test_sizes_always_list_one():
    with pytest.raises(ValueError, match="must include 1"):
        Batcher(lambda items: items, sizes=[2, 4])


def test_a_plain_function_runs_on_a_thread_so_the_event_loop_keeps_running():
    def slow(items: list) -> list:
        time.sleep(0.2)                                                 # a forward pass
        return [threading.current_thread() is threading.main_thread() for _ in items]

    async def go():
        batcher = Batcher(slow, wait_ms=0)
        call = asyncio.ensure_future(batcher(1))
        turns = 0
        while not call.done():
            await asyncio.sleep(0.01)
            turns += 1
        return await call, turns

    on_main_thread, turns = asyncio.run(go())
    assert on_main_thread is False and turns >= 5                       # the loop turned while the model ran


def test_a_caller_that_left_does_not_break_its_batch():
    async def go():
        seen: list = []
        gate = asyncio.Event()
        batcher = Batcher(_model(seen, gate=gate), sizes=[1, 2], wait_ms=10_000)
        stays, leaves = asyncio.ensure_future(batcher(1)), asyncio.ensure_future(batcher(2))
        await asyncio.sleep(0.02)
        leaves.cancel()                                                 # its request was cancelled while the batch ran
        gate.set()
        return await stays, seen

    assert asyncio.run(go()) == (10, [[1, 2]])


# ── how long a batch waits and which calls go first, decided for each call (hold, first) ─────────────────────────────
def test_with_hold_a_batch_waits_as_long_as_hold_says_and_goes_as_soon_as_it_says_zero():
    async def go():
        seen: list = []
        asked: list = []

        def hold(keys: list) -> float:
            asked.append(list(keys))
            return 0.0 if keys == ["a", "b"] else 0.05             # wait for b, never longer than 50 ms

        batcher = Batcher(_model(seen), sizes=[1, 2, 4], wait_ms=10_000, hold=hold)
        started = time.perf_counter()
        a = asyncio.ensure_future(batcher(1, key="a"))
        await asyncio.sleep(0.01)
        both = await asyncio.gather(a, batcher(2, key="b"))
        joined = time.perf_counter() - started
        alone = time.perf_counter()
        one = await batcher(3, key="c")                            # nobody comes: it goes when its wait is over
        return seen, both, joined, one, time.perf_counter() - alone, asked

    seen, both, joined, one, alone_took, asked = asyncio.run(go())
    assert seen == [[1, 2], [3]] and both == [10, 20] and one == 30
    assert joined < 0.5                                             # b's arrival ended a's wait, not wait_ms (10 s)
    assert 0.04 <= alone_took < 0.5                                 # c waited what hold said, then went alone
    assert asked[:2] == [["a"], ["a", "b"]]                         # asked again as each call arrived


def test_with_hold_calls_that_arrive_while_the_model_is_busy_are_held_again_once_it_is_free():
    async def go():
        seen: list = []
        gate = asyncio.Event()
        asked: list = []

        def hold(keys: list) -> float:
            asked.append(list(keys))
            return 0.0 if len(keys) >= 2 else 10.0

        batcher = Batcher(_model(seen, gate=gate), sizes=[1, 2], wait_ms=0, hold=hold)
        first = asyncio.ensure_future(batcher(1, key="a"))         # held 10 s for another: none comes
        await asyncio.sleep(0.01)
        second = asyncio.ensure_future(batcher(2, key="b"))        # its arrival ends the wait: [1, 2] goes
        await asyncio.sleep(0.01)
        third = asyncio.ensure_future(batcher(3, key="c"))         # arrives while the model is busy: not asked yet
        await asyncio.sleep(0.01)
        busy_asks = len(asked)
        gate.set()
        await asyncio.gather(first, second)
        await asyncio.sleep(0.01)
        return seen, busy_asks, asked, third

    seen, busy_asks, asked, third = asyncio.run(go())
    assert seen == [[1, 2]] and busy_asks == 1                      # a full batch goes unasked; c, while it ran, too
    assert asked[-1] == ["c"]                                       # ... and was once it was free: held on


def test_first_names_the_calls_that_go_apart_and_the_rest_wait_on_as_hold_says():
    async def go():
        seen: list = []
        gate = asyncio.Event()

        def first(keys: list, free: bool):
            return [keys.index("far")] if "far" in keys else None  # the far robot's call goes first, alone

        batcher = Batcher(_model(seen, gate=gate), sizes=[1, 2, 4], wait_ms=0,
                          hold=lambda keys: 0.0 if "near-b" in keys else 10.0, first=first)
        calls = [asyncio.ensure_future(batcher(n, key=k)) for n, k in ((1, "near-a"), (2, "far"), (3, "near-b"))]
        await asyncio.sleep(0.02)
        gate.set()
        return await asyncio.gather(*calls), seen

    got, seen = asyncio.run(go())
    assert seen == [[2], [1, 3]] and got == [10, 20, 30]           # far apart, then the near ones as one batch


def test_first_naming_a_number_of_calls_that_is_not_a_listed_size_is_not_followed():
    async def go():
        seen: list = []
        batcher = Batcher(_model(seen), sizes=[1, 4], wait_ms=0, first=lambda keys, free: [0, 1])
        return await asyncio.gather(*(batcher(n, key=n) for n in range(4))), seen

    got, seen = asyncio.run(go())
    assert seen == [[0, 1, 2, 3]] and got == [0, 10, 20, 30]       # 2 is not a listed size: the four went as one
