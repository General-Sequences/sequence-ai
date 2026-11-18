# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
ping_yield: a ping whose answer would come after the robot's next act is left out - that act would have waited for it -
while the connection's silence stays under a round trip and 200 ms more, so its window is kept.
"""
import time

import numpy as np
import pytest

pytest.importorskip("httpx")
pytest.importorskip("uvicorn")

from sequence_ai.network import connect, window  # noqa: E402
from sequence_ai.network.link import FIRST_TRIP_S, RTO_FLOOR_S, YIELD_MIN_GAPS  # noqa: E402
from tests.network.test_link import serving  # noqa: E402
from tests.network.test_link_defaults import Server, observation  # noqa: E402


def steady(link, *, gap: float, last_ago: float, n: int = YIELD_MIN_GAPS + 2) -> float:
    """`n` acts sent `gap` apart, the last `last_ago` ago; returns now."""
    now = time.perf_counter()
    for i in range(n):
        at = now - last_ago - (n - 1 - i) * gap
        link._act_sent(at, at)
    return now


def at_stake(link, stake: bool = True):
    """The link as on a machine where a pause restarts the window (Linux, no BBR), or where none does (macOS, BBR)."""
    link._window_at_stake = stake
    return link


def test_a_ping_that_would_meet_the_next_act_yields_while_the_silence_stays_under_a_round_trip_and_200_ms():
    link = at_stake(connect("http://127.0.0.1:9", check=False))
    now = steady(link, gap=0.5, last_ago=0.45)                    # the next act is due in 50 ms
    link._ping_done(now - 0.1, now - 0.05)                        # a 50 ms ping, sent 100 ms ago
    assert link._yields_now()                                     # its answer would come after the act
    link._ping_done(now - 0.2, now - 0.15)                        # sent 200 ms ago: the silence would reach 300 ms
    assert not link._yields_now()                                 # past 50 + 200 ms: sent, to keep the window
    link.close()


def test_once_a_long_inference_has_spent_the_window_s_silence_a_ping_that_would_meet_the_act_yields():
    """A 200 ms inference and its round trip are a silence past a round trip and 200 ms already: the next send restarts
    the window whatever it is, and a ping would only meet the act. Before this, the ping went, as one that kept the
    window would: with a 200 ms policy and a 61 ms round trip no ping yielded."""
    link = at_stake(connect("http://127.0.0.1:9", check=False))
    now = steady(link, gap=0.5, last_ago=0.45)                    # the act sent 450 ms ago, and nothing since
    link._ping_done(now - 1.2, now - 1.14)                        # a ping before it: a 60 ms round trip
    with link._timing:
        link._last_send = now - 0.45
    assert link._yields_now()                                     # the deadline passed 190 ms ago
    link.close()


def test_where_a_pause_restarts_no_window_a_ping_that_would_meet_the_act_yields_whatever_the_silence():
    link = at_stake(connect("http://127.0.0.1:9", check=False), False)      # macOS, or BBR
    now = steady(link, gap=0.5, last_ago=0.45)
    link._ping_done(now - 0.2, now - 0.15)                        # the silence would reach 300 ms: nothing to keep
    assert link._yields_now()
    link.close()


def test_a_ping_back_before_the_act_a_late_act_too_few_acts_and_ping_yield_false_send_it_as_before():
    link = connect("http://127.0.0.1:9", check=False)
    now = steady(link, gap=0.5, last_ago=0.2)                     # the next act is due in 300 ms
    link._ping_done(now - 0.05, now - 0.0)
    assert not link._yields_now()                                 # back long before it
    link.close()
    link = connect("http://127.0.0.1:9", check=False)
    now = steady(link, gap=0.5, last_ago=0.56)                    # due 60 ms ago: late past a tenth of the gap
    link._ping_done(now - 0.05, now)
    assert not link._yields_now()
    link.close()
    link = connect("http://127.0.0.1:9", check=False)
    now = steady(link, gap=0.5, last_ago=0.45, n=YIELD_MIN_GAPS)  # too few gaps to tell the spacing
    link._ping_done(now - 0.1, now - 0.05)
    assert not link._yields_now()
    link.close()
    link = connect("http://127.0.0.1:9", check=False, ping_yield=False)
    now = steady(link, gap=0.5, last_ago=0.45)
    link._ping_done(now - 0.1, now - 0.05)
    assert not link._yields_now()
    link.close()
    assert FIRST_TRIP_S == 0.1 and RTO_FLOOR_S == 0.2


def test_a_ping_left_out_has_warm_ask_again_before_the_window_could_go():
    link = at_stake(connect("http://127.0.0.1:9", check=False, ping_every_s=0.5))
    now = steady(link, gap=0.5, last_ago=0.45)                    # due in 50 ms, late in 100 ms
    link._ping_done(now - 0.1, now - 0.05)                        # the next send must go within 150 ms
    assert link._yields_now()
    assert 0.08 < link._warm._wait() < 0.11                       # asked again when the act is late, not in 500 ms
    assert link._warm._wait() == 0.5                              # once: then its turn as before
    link.close()


def test_a_ping_that_sends_nothing_is_not_counted_by_warm():
    sent = []
    warm = window.Warm(lambda: sent.append(1) or False, every_s=0.0, for_s=60.0)
    warm._used_at = time.monotonic() - 1.0
    assert warm.tick() and sent == [1] and warm.pings == 0
    warm = window.Warm(lambda: None, every_s=0.0, for_s=60.0)    # a ping that returns nothing: one sent, as before
    warm._used_at = time.monotonic() - 1.0
    assert warm.tick() and warm.pings == 1


def test_acts_at_a_steady_spacing_no_longer_wait_behind_a_slow_ping():
    server = Server(ping_s=0.08)                                  # each ping answered 80 ms after it is asked
    with serving(server) as url, connect(url, ping_every_s=0.05, check=False) as link:
        link.wait_until_ready()
        began = time.perf_counter()
        for i in range(18):                                       # an act every 300 ms, as a robot's loop sends them
            time.sleep(max(0.0, began + i * 0.3 - time.perf_counter()))
            link.act(observation())
            if i == 5:
                link._stats.reset()                               # their spacing learned: the 12 after it are counted
        snap = link.stats()["counters"]
    assert snap["acts"] == 12 and snap["pings_yielded"] >= 3 and snap["pings"] >= 6, snap
    assert snap["acts_behind_ping"] <= 3, snap                   # a ping would otherwise meet most of them; a loaded
                                                                  # machine's timing lets a few


@pytest.mark.parametrize("stake", [True, False], ids=["linux-no-bbr", "macos-or-bbr"])
def test_acts_after_a_200_ms_inference_no_longer_wait_behind_a_ping(stake):
    """pi0.5's inference is about 195 ms: the silence from an act's send to its answer passes a round trip and 200 ms
    before any ping can go, and the ping that would meet the next act yields all the same, on either kind of machine.
    A loaded machine's timing lets a few meet: without yielding, about a third of the acts here would."""
    server = Server(act_s=0.2, ping_s=0.06)                       # a 200 ms policy; each ping answered 60 ms after
    with serving(server) as url, connect(url, ping_every_s=0.05, check=False) as link:
        at_stake(link, stake)
        link.wait_until_ready()
        began = time.perf_counter()
        for i in range(18):                                       # an act every 500 ms
            time.sleep(max(0.0, began + i * 0.5 - time.perf_counter()))
            link.act(observation())
            if i == 5:
                link._stats.reset()
        snap = link.stats()["counters"]
    assert snap["acts"] == 12 and snap["pings_yielded"] >= 1, snap
    assert snap["acts_behind_ping"] <= 3, snap
