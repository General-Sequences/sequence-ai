# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""The clock tests/engine/test_optimize.py and test_xla.py time their acts on, and test_serve_slack.py its warm-up: a
stand-in's time is what it says it is, however busy the machine.

optimize() keeps a switch only when the act is at least 10% faster with it (MIN_SPEEDUP), timed with time.perf_counter.
The tests' stand-ins take their time by name - a kernel's launch, a heavy step, 4 ms each - and one that saves nothing
is exactly as fast as eager mode. On the wall clock that held only on a quiet machine: under load the medians of the
same work came out more than 10% apart, and a switch that saved nothing was kept. Here a stand-in moves this clock
instead of sleeping, and optimize() and cuda_graphs' calls counters read it in place of time.perf_counter, so the same
work is the same time on every run, and the rule each test pins - kept when 10% faster, taken out when not - is judged
on what the switch saves alone.

    clock = install(monkeypatch)   # in an autouse fixture: from 0, for the one test
    CLOCK.wait(0.004)              # in a stand-in: 4 ms of its time
"""
from __future__ import annotations

import importlib
import time
import types

# Each time optimize() reads the clock it moves on 10 us: an act it times takes some time, as a real one does (a
# replayed whole call that waits for nothing is 0.01 ms, not 0), and an act's time is its waits and that one tick.
TICK_NS = 10_000


class Clock:
    """Time in whole nanoseconds, moved on only by the stand-ins' waits and by optimize()'s own reads."""

    def __init__(self) -> None:
        self.ns = 0

    def wait(self, seconds: float) -> None:
        """A stand-in's time passing: what a sleep of `seconds` would have taken, taken at once."""
        self.ns += round(seconds * 1e9)

    def timed(self) -> float:
        """optimize()'s time.perf_counter: one tick a read, so an act timed between two reads is its waits and a tick,
        the same for the same work."""
        self.ns += TICK_NS
        return self.ns / 1e9

    def counted(self) -> float:
        """A read that does not tick: the calls counters' (cuda_graphs.Census and Calls), so a module's time is the
        waits made inside it alone - what decides which modules are worth a graph (min_ms) and which call is the
        slowest; and serve()'s, so a batch its warm-up times is the stand-in's waits to the nanosecond."""
        return self.ns / 1e9


class TimeWith:
    """The time module with `perf_counter` for its perf_counter and nothing else changed: for a module that times its
    work with perf_counter and runs other clocks too (serve(): time.monotonic for what is not its batches' timing)."""

    def __init__(self, perf_counter) -> None:
        self.perf_counter = perf_counter

    def __getattr__(self, name: str):
        return getattr(time, name)            # every other name: the time module's own


CLOCK = Clock()


def install(monkeypatch) -> Clock:
    """CLOCK from 0, read by optimize() and cuda_graphs in place of time.perf_counter until the test ends (monkeypatch
    puts the time module back). Their only use of the time module is perf_counter, so a clock with that alone is all
    they read: a use of anything else fails the test at once rather than reading the wall clock."""
    CLOCK.ns = 0
    readers = (("sequence_ai.engine.optimize", CLOCK.timed), ("sequence_ai.engine.cuda_graphs", CLOCK.counted))
    for module, read in readers:
        # import_module: `sequence_ai.engine.optimize` the module, which `from sequence_ai.engine import optimize`
        # would give as the function the package exports
        monkeypatch.setattr(importlib.import_module(module), "time", types.SimpleNamespace(perf_counter=read))
    return CLOCK
