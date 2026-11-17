# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
Replicas of a model, each in its own process (sequence_ai/engine/replicas.py): a call runs in a replica's process, on
the GPU assigned to it, with the model loaded once; two replicas run two calls concurrently; a model that raises is its
caller's error and the replica stays alive; a replica that dies mid-call, or exceeds its deadline, is the caller's
ReplicaDied and is replaced; a load that raises is reported; replicas that die in a row are not replaced indefinitely;
from an event loop the loop keeps running while the model runs. Every test starts real processes.

    python -m pytest -q tests/engine/test_replicas.py
"""
from __future__ import annotations

import asyncio
import os
import threading
import time

import pytest

from sequence_ai.engine.replicas import PoolHalted, ReplicaDied, ReplicaError, ReplicaPool, ReplicaStartFailed
from tests.engine import replica_models as models


def _until(what, seconds: float = 20.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        got = what()
        if got:
            return got
        time.sleep(0.05)
    raise AssertionError("it never came")


def test_a_call_runs_in_a_replicas_own_process_on_its_gpu_with_the_model_loaded_once():
    pool = ReplicaPool(models.load_echo, replicas=1, devices=[1])
    try:
        first, second = pool.call({"x": 1}), pool.call({"x": 2})
        assert first["pid"] != os.getpid() and first["pid"] == second["pid"] == pool.pids[0]
        assert (first["calls"], second["calls"]) == (1, 2)                   # one model, kept between calls
        assert first["gpu"] == "1" and second["payload"] == {"x": 2}
    finally:
        pool.close()


def test_two_replicas_run_two_calls_at_once():
    pool = ReplicaPool(models.load_echo, replicas=2, devices=[0, 0])
    try:
        answers: list = []
        started = time.perf_counter()
        threads = [threading.Thread(target=lambda: answers.append(pool.call({"sleep": 0.5}))) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert time.perf_counter() - started < 0.9                           # 0.5 s each, together
        assert len({a["pid"] for a in answers}) == 2 and pool.size == 2
    finally:
        pool.close()


def test_a_model_that_raises_is_its_callers_error_and_the_replica_stays_alive():
    pool = ReplicaPool(models.load_echo)
    try:
        before = pool.call(0)["pid"]
        with pytest.raises(ReplicaError, match="ValueError: the model refuses this") as raised:
            pool.call("raise")
        assert "replica_models.py" in raised.value.traceback               # where in the model's code
        assert pool.call(0)["pid"] == before                               # the same replica, still loaded
    finally:
        pool.close()


def test_a_replica_that_dies_mid_call_raises_to_its_caller_and_is_replaced():
    pool = ReplicaPool(models.load_echo)
    try:
        before = pool.call(0)["pid"]
        with pytest.raises(ReplicaDied, match="exit code 3"):
            pool.call("die")
        after = pool.call(0)                                               # waits for the replica started in its place
        assert after["pid"] != before and after["calls"] == 1 and pool.size == 1
    finally:
        pool.close()


def test_a_call_past_its_deadline_terminates_its_replica_which_is_replaced():
    pool = ReplicaPool(models.load_echo)
    try:
        before = pool.call(0)["pid"]
        started = time.perf_counter()
        with pytest.raises(ReplicaDied, match="deadline") as raised:
            pool.call({"sleep": 30}, timeout=0.3)
        assert raised.value.reason == "deadline" and time.perf_counter() - started < 10
        assert pool.call(0)["pid"] != before
    finally:
        pool.close()


def test_a_load_that_raises_reports_what_it_raised():
    with pytest.raises(ReplicaStartFailed, match="RuntimeError: no weights here") as raised:
        ReplicaPool(models.load_that_raises)
    assert "load_that_raises" in raised.value.traceback


def test_replicas_that_die_in_a_row_are_not_replaced_forever():
    pool = ReplicaPool(models.load_then_always_dies, deaths_in_a_row=2)
    try:
        with pytest.raises(ReplicaDied):
            pool.call(0)                                                    # the first: replaced
        with pytest.raises(ReplicaDied):
            pool.call(0)                                                    # the second in a row: not replaced
        with pytest.raises(PoolHalted, match="2 replicas died in a row"):
            pool.call(0)
        assert pool.size == 0
    finally:
        pool.close()


def test_a_call_on_a_closed_pool_is_refused():
    pool = ReplicaPool(models.load_echo)
    pid = pool.pids[0]
    pool.close()
    with pytest.raises(PoolHalted, match="closed"):
        pool.call(0)

    def gone() -> bool:
        try:
            os.kill(pid, 0)
        except OSError:
            return True
        return False

    assert _until(gone)


def test_from_an_event_loop_the_loop_keeps_running_while_the_model_runs():
    async def go():
        pool = ReplicaPool(models.load_echo)
        try:
            call = asyncio.ensure_future(pool.run({"sleep": 0.3}))
            turns = 0
            while not call.done():
                await asyncio.sleep(0.01)
                turns += 1
            return (await call)["pid"] != os.getpid(), turns
        finally:
            pool.close()

    in_a_copy, turns = asyncio.run(go())
    assert in_a_copy and turns >= 5


def test_one_device_per_replica():
    with pytest.raises(ValueError, match="one device per replica"):
        ReplicaPool(models.load_echo, replicas=2, devices=[0])


def test_a_load_a_new_process_cannot_import_raises_with_how_to_write_it():
    with pytest.raises(TypeError, match="defined at a module's top level"):
        ReplicaPool(lambda: len)
