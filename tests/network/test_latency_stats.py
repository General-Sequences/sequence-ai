# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
One connection's counters, gauges and latencies in microseconds (sequence_ai/network/latency_stats.py): each latency's
percentiles are nearest-rank over its latest 256 samples, and None until the window holds enough of them; counters and
gauges are separate; reset clears the counts and the windows and keeps the gauges; concurrent threads lose nothing.

    python -m pytest -q tests/network/test_latency_stats.py
"""
from __future__ import annotations

import math
import threading

import pytest

from sequence_ai.network import latency_stats as ls


def nearest_rank(xs, q):
    """The reference the percentiles must match: nearest rank, computed directly."""
    xs = sorted(xs)
    return xs[max(0, math.ceil(q / 100 * len(xs)) - 1)] if xs else None


def test_a_latencys_percentiles_are_nearest_rank_over_its_latest_256_samples():
    stats = ls.LatencyStats()
    samples = [(i * 7919) % 1000 + 1 for i in range(1000)]     # scrambled: a window kept sorted would fail
    for us in samples:
        stats.sample("loop_us", us)
    got = stats.snapshot()["latency_us"]["loop_us"]
    window = samples[-256:]
    assert got["n"] == 256 and got["total"] == 1000 and got["last"] == samples[-1]
    assert (got["p50"], got["p95"], got["p99"]) == tuple(nearest_rank(window, q) for q in (50, 95, 99))
    assert got["max"] == max(window) and got["mean"] == round(sum(window) / 256)


def test_a_percentile_is_none_until_the_window_holds_enough_samples_for_it():
    """With fewer than 1/(1-q) samples a q-percentile equals the window's max (the p99 of 20 samples is the largest of
    the 20, not a p99), so it is None until there are 2 samples for p50, 20 for p95 and 100 for p99.
    """
    stats = ls.LatencyStats()
    assert stats.snapshot()["latency_us"] == {}                  # nothing recorded yet: no series at all
    seen = {}
    for n in range(1, 101):
        stats.sample("frame_us", 1000 + n)
        got = stats.snapshot()["latency_us"]["frame_us"]
        seen[n] = tuple(got[k] is not None for k in ("p50", "p95", "p99"))
        assert got["last"] == 1000 + n and got["max"] == 1000 + n
    assert seen[1] == (False, False, False)
    assert seen[2] == seen[19] == (True, False, False)
    assert seen[20] == seen[99] == (True, True, False)
    assert seen[100] == (True, True, True)


def test_counters_and_gauges_are_separate_and_reset_clears_the_counts_and_windows_and_keeps_the_gauges():
    """A gauge is a current value (here the robot clock's offset), not a count: set to zero, it would read as a clock
    that lost its synchronization (LiveKit Portal's reset_metrics() leaves its offset and its synced flag unchanged for
    the same reason)."""
    stats = ls.LatencyStats()
    stats.count("commands_dropped")
    stats.count("commands_dropped", 2)
    stats.gauge("clock_offset_us", -1234)
    stats.sample("rtt_us", 5000)
    snap = stats.snapshot()
    assert snap["counters"] == {"commands_dropped": 3} and snap["gauges"] == {"clock_offset_us": -1234}
    stats.reset()
    snap = stats.snapshot()
    assert snap["counters"] == {"commands_dropped": 0} and snap["gauges"] == {"clock_offset_us": -1234}
    assert snap["latency_us"]["rtt_us"] == {"n": 0, "total": 0, "last": None, "mean": None, "max": None,
                                             "p50": None, "p95": None, "p99": None}


def test_since_measures_from_a_wall_clock_timestamp_in_microseconds(monkeypatch):
    """A timestamp sent between machines is in wall-clock microseconds (now_us), so since() records now_us() minus
    it."""
    monkeypatch.setattr(ls.time, "time_ns", lambda: 1_700_000_000_123_456_789)
    assert ls.now_us() == 1_700_000_000_123_456
    stats = ls.LatencyStats()
    stats.since("frame_us", 1_700_000_000_100_000)
    assert stats.snapshot()["latency_us"]["frame_us"]["last"] == 23_456


def test_threads_counting_and_timing_at_once_lose_nothing():
    stats = ls.LatencyStats()
    go = threading.Barrier(8)

    def work():
        go.wait()
        for i in range(5000):
            stats.count("frames_received")
            stats.sample("frame_us", i)
            if i % 500 == 0:
                stats.snapshot()                                 # a reader concurrent with the writers

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    snap = stats.snapshot()
    assert snap["counters"]["frames_received"] == 40_000
    assert snap["latency_us"]["frame_us"]["total"] == 40_000 and snap["latency_us"]["frame_us"]["n"] == 256


def test_a_sample_that_is_not_a_finite_number_raises():
    """A bad timestamp is a bug in the caller that should surface, not a sample to drop silently."""
    stats = ls.LatencyStats()
    for bad in (float("nan"), float("inf"), None, "12"):
        with pytest.raises((TypeError, ValueError)):
            stats.sample("loop_us", bad)
    assert stats.snapshot()["latency_us"].get("loop_us", {}).get("total", 0) == 0
