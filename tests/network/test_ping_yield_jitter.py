# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
ping_yield on a line whose round trips jitter: a ping is judged by the slowest tenth of the last 32 pings' round trips,
not their median. LINE_RTT_MS are 520 round trips measured once a second from a robot on a home Wi-Fi network whose
line out is cellular, to a GPU nearby (p50 52 ms, p90 128, p99 253, the longest 891): replayed here, each decision a
ping takes some time before the robot's next act is due, and the round trip it then takes drawn from the same line.
"""
import random
import time

from sequence_ai.network import connect
from sequence_ai.network import link as link_module

LINE_RTT_MS = [
    43, 51, 47, 97, 109, 109, 76, 198, 175, 230, 31, 38, 35, 112, 39, 35, 102, 34, 26, 42, 49, 52, 47, 52, 36, 109,
    45, 74, 126, 75, 62, 36, 30, 51, 38, 41, 140, 28, 44, 37, 42, 37, 42, 49, 41, 56, 93, 46, 891, 124, 186, 49, 29,
    39, 45, 44, 50, 46, 147, 33, 56, 39, 51, 38, 56, 97, 46, 49, 55, 118, 38, 72, 33, 45, 75, 30, 42, 98, 112, 96,
    49, 391, 56, 36, 108, 88, 76, 51, 89, 106, 80, 218, 153, 71, 68, 41, 33, 35, 38, 117, 416, 253, 200, 30, 44, 66,
    189, 45, 24, 41, 37, 107, 47, 35, 31, 45, 30, 36, 36, 44, 30, 109, 39, 75, 122, 36, 30, 35, 30, 51, 79, 45, 42,
    120, 42, 84, 134, 43, 40, 40, 106, 31, 44, 112, 60, 34, 149, 33, 61, 48, 87, 35, 29, 107, 50, 40, 151, 34, 33,
    78, 82, 34, 43, 35, 107, 37, 35, 36, 33, 42, 85, 37, 43, 35, 116, 50, 95, 76, 31, 136, 99, 55, 53, 44, 50, 111,
    28, 184, 41, 38, 81, 38, 41, 45, 37, 36, 91, 68, 64, 32, 37, 69, 92, 54, 37, 53, 27, 172, 76, 34, 55, 105, 26,
    74, 79, 120, 85, 79, 242, 164, 100, 38, 83, 46, 155, 50, 93, 120, 35, 53, 144, 210, 115, 81, 85, 155, 53, 108,
    58, 142, 146, 22, 146, 226, 233, 74, 41, 69, 84, 107, 48, 36, 46, 148, 52, 52, 69, 32, 44, 51, 40, 95, 97, 56,
    58, 47, 216, 137, 36, 86, 44, 41, 50, 106, 44, 36, 64, 124, 35, 51, 42, 45, 56, 56, 66, 102, 26, 55, 42, 39, 59,
    46, 89, 60, 56, 118, 121, 48, 122, 44, 36, 40, 170, 51, 32, 94, 47, 59, 29, 110, 87, 36, 57, 60, 123, 168, 161,
    128, 49, 40, 32, 81, 107, 74, 37, 76, 42, 70, 101, 56, 68, 42, 57, 44, 74, 212, 69, 159, 92, 257, 178, 29, 379,
    72, 49, 35, 41, 45, 40, 167, 42, 66, 122, 69, 43, 26, 57, 43, 98, 97, 49, 46, 42, 38, 108, 64, 32, 29, 65, 39,
    93, 110, 72, 36, 151, 43, 67, 40, 44, 44, 35, 35, 126, 43, 40, 57, 39, 54, 48, 36, 41, 43, 34, 105, 45, 146, 45,
    178, 34, 57, 65, 42, 63, 68, 91, 48, 57, 101, 69, 172, 39, 90, 40, 95, 40, 34, 100, 105, 103, 73, 41, 36, 99,
    45, 37, 53, 100, 42, 68, 101, 81, 53, 80, 39, 42, 47, 53, 110, 131, 54, 36, 46, 65, 191, 45, 42, 56, 37, 53,
    136, 49, 38, 113, 46, 36, 49, 40, 120, 40, 52, 38, 36, 64, 33, 48, 111, 53, 45, 128, 87, 35, 71, 53, 45, 35,
    118, 69, 117, 31, 97, 72, 39, 74, 38, 32, 105, 45, 39, 108, 49, 30, 45, 39, 30, 70, 105, 64, 36, 42, 124, 39,
    34, 40, 34, 39, 39, 29, 51, 88, 105, 190, 44, 76, 49, 57, 73, 57, 94, 32, 37
]
GAP = 0.533                                                     # 15 Hz, an act every 8 steps


def met_share(seed: int, trials: int = 400) -> "tuple[float, int]":
    """Of the pings sent within 300 ms before the next act is due, the share still on their way when it comes; and how
    many were sent."""
    rng = random.Random(seed)
    link = connect("http://127.0.0.1:9", check=False)
    link._window_at_stake = False                                # macOS: the ping keeps no window, only the act counts
    sent = met = 0
    for _ in range(trials):
        lead = rng.uniform(0.0, 0.3)                             # the act is due this long after the decision
        now = time.perf_counter()
        with link._timing:
            link._act_starts.clear()
            link._ping_trips.clear()
        for i in range(5):
            at = now + lead - (5 - i) * GAP                      # the next one due `lead` from now
            link._act_sent(at, at)
        for trip in rng.sample(LINE_RTT_MS, link_module.YIELD_TRIPS):
            link._ping_done(now - 2.0, now - 2.0 + trip / 1000)  # the last pings' round trips, from the line
        if link._yields_now():
            continue
        sent += 1
        met += rng.choice(LINE_RTT_MS) / 1000 > lead             # its answer came after the act was asked for
    link.close()
    return met / max(sent, 1), sent


def test_on_a_jittery_line_a_ping_sent_before_an_act_rarely_meets_it(monkeypatch):
    now, sent_now = met_share(seed=1)
    monkeypatch.setattr(link_module, "YIELD_TRIP_Q", 0.5)       # the median of the last 8, as before
    monkeypatch.setattr(link_module, "YIELD_TRIPS", 8)
    before, sent_before = met_share(seed=1)
    assert now <= 0.03 and before >= 2 * now and before >= 0.04, (now, before)    # 1.6% against 5.2% for seed 1
    assert sent_now >= 100, sent_now                            # a ping well before the act still goes


def test_a_ping_well_before_the_act_goes_whatever_the_line():
    link = connect("http://127.0.0.1:9", check=False)
    link._window_at_stake = False
    now = time.perf_counter()
    for i in range(5):
        at = now + 0.45 - (5 - i) * GAP                         # the next act due in 450 ms
        link._act_sent(at, at)
    for trip in LINE_RTT_MS[:32]:
        link._ping_done(now - 2.0, now - 2.0 + trip / 1000)
    assert not link._yields_now()
    link.close()
