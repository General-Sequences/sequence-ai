# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
latency_stats, alone: counters, a gauge and latencies, read as one snapshot.

    python examples/components/latency_stats.py
"""
import random

from sequence_ai.network.latency_stats import LatencyStats, now_us

stats = LatencyStats(counters=["requests"], gauges=["clock_offset_us"], latencies=["request_us"])
for _ in range(200):
    stats.sample("request_us", random.gauss(25_000, 2_000))          # microseconds
    stats.count("requests")
sent = now_us()                                                # a timestamp sent from another machine
stats.since("request_us", sent)
stats.gauge("clock_offset_us", 1200)
snap = stats.snapshot()
timing = snap["latency_us"]["request_us"]
p50, p95, p99 = (timing[key] / 1000 for key in ("p50", "p95", "p99"))
print(f"{snap['counters']['requests']} requests: p50 {p50:.0f} ms, p95 {p95:.0f} ms, p99 {p99:.0f} ms")
