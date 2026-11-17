# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
Scheduling by the slack an action chunk leaves, for one replica of a model (sequence_ai/engine/slack.py): each listed
batch size is timed after an untimed run; a policy's two numbers and its rate give the playback time left when its robot
requests the next response; a replica's capacity is as many robots as a batch can answer within that, and no more than
its cycle can take; each robot is evaluated by its own RTT, so a far one never lowers the capacity available to the
nearer ones and is the first over capacity; an open batch waits for the replica's other robots until its tightest
request can wait no longer, and for none that cannot arrive or cannot share it; a far robot's request is dispatched
separately, first, where the whole batch would make it late; and where a policy sets a wait, that wait replaces the
buffer.

    python -m pytest -q tests/engine/test_slack.py
"""
from __future__ import annotations

import math

import pytest

from sequence_ai.engine import slack

MARGIN = slack.MARGIN_MS / 1000.0
TRIP = slack.ROUND_TRIP_MS / 1000.0
# 15 valid steps per response, the next requested at the 8th. At 15 Hz a response is fully executed in 1 s, with 467 ms
# of it left when its robot requests the next, which it does every 533 ms; at 20 Hz 350 ms are left, and it sends a
# request every 400 ms.
AT_15 = slack.playback(hz=15.0, valid_len=15, replan_at=8)
AT_20 = slack.playback(hz=20.0, valid_len=15, replan_at=8)


def _timed(forward: dict, runs: int = 3) -> slack.Table:
    """A table whose sizes each ran `runs` times at the given ms: that is its p50 and its p95."""
    table = slack.Table()
    for size, ms in forward.items():
        for _ in range(runs):
            table.add(size, ms)
    return table


# ── what a batch takes: the table, and its timing ────────────────────────────────────────────────────────────────────

def test_a_tables_p95_is_the_slowest_of_three_runs_and_an_untimed_size_reads_the_next_one_up():
    table = slack.Table()
    for ms in (200.0, 210.0, 250.0):
        table.add(1, ms)
    for ms in (300.0, 320.0, 330.0):
        table.add(4, ms)
    assert table.p95(1) == 250.0 and table.p50(1) == 210.0
    assert table.p95(2) == 330.0                          # a policy pads 2 up to 4
    assert table.p95(8) == pytest.approx(330.0 * 8 / 4)   # past the largest: its time per request, times the size
    assert table.p50(8) is None and slack.Table().p95(1) is None


def test_a_table_reports_each_sizes_p50_p95_and_runs_and_takes_anothers_runs_as_its_own():
    table = slack.Table()
    for ms in (200.0, 210.0, 250.0):
        table.add(1, ms)
    assert table.as_dict() == {"1": {"p50": 210.0, "p95": 250.0, "n": 3}}
    held = slack.Table()
    held.take(table)                                      # a timing made later, seen by whoever holds `held`
    assert held.sizes() == [1] and held.p95(1) == 250.0
    held.drop(1)
    assert held.sizes() == [] and table.sizes() == [1]


def test_each_listed_size_is_timed_after_an_untimed_run_that_compiles_it():
    calls = []
    ticks = iter(range(1000))
    table = slack.measure(calls.append, sizes=[1, 2, 4], clock=lambda: next(ticks) / 1000.0)
    assert calls == [1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4]                   # one untimed, then RUNS timed, each size
    assert table.sizes() == [1, 2, 4] and set(table.first_ms) == {1, 2, 4}


def test_a_size_whose_forward_raises_reports_which_with_the_sizes_timed_under_it_and_keeps_none_of_its_own():
    """A size that compiled and failed on a later run is not kept either: no batch is handed a size it could not run."""
    calls = []

    def run(size):
        calls.append(size)
        if size == 4 and calls.count(4) == 3:          # its compile run and one timed run succeeded, the next did not
            raise RuntimeError("CUDA out of memory")

    with pytest.raises(slack.SizeFailed) as failed:
        slack.measure(run, sizes=[1, 2, 4])
    assert failed.value.size == 4 and failed.value.table.sizes() == [1, 2] and 4 not in failed.value.table.first_ms
    assert isinstance(failed.value.error, RuntimeError) and failed.value.__cause__ is failed.value.error
    assert str(failed.value) == "batch size 4: RuntimeError: CUDA out of memory"


FORWARD = {1: 100.0, 2: 120.0, 4: 130.0}       # each listed size's forward, as its timing had it


def _served(ms: dict) -> slack.Served:
    """Each size answered SERVED_MIN times, `ms` after its batch was dispatched: enough for it to be timed from that."""
    served = slack.Served()
    for size, took in ms.items():
        for _ in range(slack.SERVED_MIN):
            served.add(size, took, FORWARD.get(size))
    return served


def test_a_size_served_too_few_times_is_its_forward_and_the_servers_own_handling():
    table, served = _timed(FORWARD), slack.Served()
    assert served.post_ms(2, table) == pytest.approx(FORWARD[2])              # nothing served: its forward pass alone
    for _ in range(3):
        served.add(1, 170.0, FORWARD[1])           # 70 ms of the server's handling, three requests: too few for size 1
    assert served.post_ms(1, table) == pytest.approx(FORWARD[1] + 70.0)
    assert served.post_ms(2, table) == pytest.approx(FORWARD[2] + 70.0)
    for _ in range(slack.SERVED_MIN - 3):
        served.add(1, 170.0, FORWARD[1])
    assert served.post_ms(1, table) == pytest.approx(170.0)
    assert slack.Served().post_ms(2, slack.Table()) is None                   # nothing timed at all


def test_a_batchs_typical_time_is_its_median_as_served_and_before_that_its_forwards_and_the_handlings():
    table, served = _timed(FORWARD), slack.Served()
    assert served.typical_ms(2, table) == pytest.approx(FORWARD[2])
    for took in (150.0, 160.0, 190.0):
        served.add(1, took, FORWARD[1])            # 50, 60 and 90 ms of handling: its median 60
    assert served.typical_ms(2, table) == pytest.approx(FORWARD[2] + 60.0)
    assert served.post_ms(2, table) == pytest.approx(FORWARD[2] + 90.0)       # the waiting rule uses the tail
    for _ in range(slack.SERVED_MIN):
        served.add(2, 140.0, FORWARD[2])
    assert served.typical_ms(2, table) == pytest.approx(140.0)
    assert slack.Served().typical_ms(2, slack.Table()) is None


# ── what a robot reports about itself ────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("said, read", [("131", 131.0), (131, 131.0), ("0", 0.0), ("5000", 5000.0), (" 42 ", 42.0)])
def test_a_robots_time_outside_its_server_reads_as_a_whole_number_of_ms_up_to_five_seconds(said, read):
    assert slack.outside_ms(said) == read


@pytest.mark.parametrize("said", [None, "", "5001", "-1", "12.5", "1e3", "abc", "١٢", True, 12.5, [131]])
def test_any_other_value_for_the_rtt_counts_as_not_reported(said):
    assert slack.outside_ms(said) is None


@pytest.mark.parametrize("said, read", [("280", 280.0), (280, 280.0), ("0", 0.0), ("600000", 600000.0)])
def test_the_time_a_robot_reports_as_left_in_its_chunk_reads_as_whole_ms(said, read):
    assert slack.left_ms(said) == read


@pytest.mark.parametrize("said", [None, "", "-5", "600001", "1.5", "x", True, 280.5])
def test_any_other_value_for_the_time_left_counts_as_not_reported(said):
    assert slack.left_ms(said) is None


# ── how a policy's chunks are executed ───────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("said, valid, replan", [
    # its N valid steps at hz, and how many are left when its robot requests the next at step k: N - k
    (dict(hz=15.0, valid_len=15, replan_at=8), 15, 8),
    (dict(hz=15.0, valid_len=8, replan_at=8), 8, 8),
    (dict(hz=15, valid_len=15, replan_at=20), 15, 15),      # requested past its valid steps: taken as the last of them
    # k not given: a request is sent once a response is fully executed, nothing left; N not given: its k steps are all
    (dict(hz=15.0, valid_len=15), 15, 15),
    (dict(hz=15.0, replan_at=8), 8, 8),
    # neither number, or no rate: its robots are not batched with others; nothing is read from a response
    (dict(hz=15.0), None, None), (dict(hz=None, valid_len=15, replan_at=8), None, None),
    (dict(hz=0, valid_len=15), None, None), (dict(hz=1001, valid_len=15), None, None),
    (dict(hz=True, valid_len=15), None, None), (dict(hz=15.0, valid_len=True), None, None),
    (dict(hz=15.0, valid_len=0), None, None), (dict(hz=15.0, valid_len=7.5), None, None),
])
def test_playback_is_a_chunks_valid_steps_and_what_is_left_when_the_next_request_is_sent(said, valid, replan):
    got = slack.playback(**said)
    assert got == (None if valid is None else pytest.approx(
        {"play_s": valid / 15.0, "buffer_s": (valid - replan) / 15.0, "step_s": 1 / 15.0}))


def test_a_wait_is_on_a_policys_playback_only_where_it_sets_one():
    assert slack.playback(20, 10, 10, 1.7)["wait_s"] == 1.7
    assert slack.playback(20, 10, 10, max_wait_s=60)["wait_s"] == 60.0
    for unset in (None, 0, -1.0, 60.5, True):
        assert "wait_s" not in slack.playback(20, 10, 10, unset)


# ── how many robots a replica can serve, robots with the same RTT ────────────────────────────────────────────────────

def test_a_replica_seats_the_robots_one_batch_serves_within_their_buffer_less_the_rtt():
    """213 / 327 / 599 ms at batch 1 / 2 / 4 (one policy on one GPU). A robot with 15 valid steps per response that
    requests the next at the 8th, at 15 Hz, has 7 steps left when it sends its request (467 ms): less the RTT and the
    margin, a batch of 2 does not answer in that time, so the capacity is one robot per replica; requesting at the 4th
    (733 ms) gives a capacity of two, at the 1st (933 ms) four; requesting at the last valid step leaves no slack."""
    table = _timed({1: 213.0, 2: 327.0, 4: 599.0})
    assert slack.per_replica(table, 7 / 15 * 1000.0, sizes=[1, 2, 4]) == 1
    assert slack.per_replica(table, 11 / 15 * 1000.0, sizes=[1, 2, 4]) == 2
    assert slack.per_replica(table, 14 / 15 * 1000.0, sizes=[1, 2, 4]) == 4
    assert slack.per_replica(table, 0.0, sizes=[1, 2, 4]) == 1              # none fits: one all the same


def test_a_replica_seats_by_the_listed_sizes_it_timed_alone():
    table = _timed({1: 10.0, 2: 20.0, 8: 60.0})
    assert slack.per_replica(table, 500.0, sizes=[1, 2, 8]) == 8
    assert slack.per_replica(table, 500.0, sizes=[1, 2]) == 2            # 8 not listed
    assert slack.per_replica(table, 500.0, sizes=[1, 2, 4, 8, 16]) == 8  # 4 and 16 never timed: never used for capacity


BUFFER_MS = AT_20["buffer_s"] * 1000.0                                   # 350 ms
CYCLE_MS = (AT_20["play_s"] - AT_20["buffer_s"]) * 1000.0                # 400 ms: it holds 360 ms of batches


def test_a_replica_seats_as_many_robots_as_a_batch_serves_in_time_for_its_furthest_robot():
    table = _timed(FORWARD)
    assert slack.per_replica(table, BUFFER_MS, sizes=[1, 2, 4]) == 4                             # 130 + 100 + 50
    served = _served({1: 150.0, 2: 160.0, 4: 197.0})
    assert slack.per_replica(table, BUFFER_MS, sizes=[1, 2, 4], served=served) == 4              # 197 + 100 + 50 = 347
    assert slack.per_replica(table, BUFFER_MS, sizes=[1, 2, 4], served=served, outside_ms=131.0) == 2  # 160 + 131 + 50
    assert slack.per_replica(table, BUFFER_MS, sizes=[1, 2, 4], served=served, outside_ms=200.0) == 1  # never under one


# A stand-in paced like a 15 Hz policy (AT_15: 467 ms of buffer, 533 ms between a robot's requests): each listed size's
# forward pass as it was timed, from the load measurement in the module's docstring.
PACED_FORWARD = {1: 234.0, 2: 254.0, 4: 299.0, 8: 382.0, 16: 550.0}
PACED_BUFFER_MS = AT_15["buffer_s"] * 1000.0
PACED_CYCLE_MS = (AT_15["play_s"] - AT_15["buffer_s"]) * 1000.0


def test_a_server_that_has_served_nothing_yet_seats_by_its_forwards_not_by_a_handling_it_guessed():
    """Nothing served, its cycle not given: a batch of four takes its 299 ms forward pass, and 299 + 100 + 50 <= 467, so
    the capacity is four robots per replica by the buffer alone, with no guess added for the server's own handling (not
    measured yet): a guessed 50 ms would make it 499 and give a capacity of two."""
    table, sizes = _timed(PACED_FORWARD), sorted(PACED_FORWARD)
    assert slack.per_replica(table, PACED_BUFFER_MS, sizes=sizes) == 4
    assert slack.per_replica(table, PACED_BUFFER_MS, sizes=sizes, served=slack.Served()) == 4


def test_a_replica_seats_on_a_clock_not_only_by_one_batch_fitting_the_buffer():
    """Given its cycle (k / hz = 8 / 15 s = 533 ms): a batch of four answers within the 467 ms buffer, so the buffer
    alone gives a capacity of four. Run on a clock, two robots 100 ms away are late in the starts in which one request
    arrives as the other's batch of one runs and waits for it, 16 of 64: over the starts 91.6% of their responses are
    back in time, short of IN_TIME_MIN. The capacity is one. Two robots 20 ms away have every response in time, and
    three have 93.8%, at a load of 0.92 and more. Never more than the buffer already gives."""
    table, sizes = _timed(PACED_FORWARD), sorted(PACED_FORWARD)
    assert PACED_CYCLE_MS == pytest.approx(8 / 15 * 1000.0)                                      # k / hz
    assert slack.per_replica(table, PACED_BUFFER_MS, sizes=sizes) == 4                           # buffer alone: four
    assert slack.per_replica(table, PACED_BUFFER_MS, sizes=sizes, cycle_ms=PACED_CYCLE_MS) == 1  # on a clock: one
    assert slack.per_replica(table, PACED_BUFFER_MS, sizes=sizes, cycle_ms=PACED_CYCLE_MS, outside_ms=20.0) == 2


def test_a_batch_of_one_a_millisecond_slower_seats_the_same():
    """The formula for the load that the clock replaced took four robots to run as batches of one where those fit in the
    400 ms cycle (4 * 100), and as two batches of two where they did not (4 * 101): three robots for the faster model,
    four for the slower. On a clock both are three: four robots 100 ms away have every response in time, and in 2
    starts of 128 their requests fall so that each runs alone, which fills the cycle."""
    assert slack.per_replica(_timed({1: 100.0, 2: 120.0, 4: 130.0}), BUFFER_MS, sizes=[1, 2, 4], cycle_ms=CYCLE_MS) == 3
    assert slack.per_replica(_timed({1: 101.0, 2: 120.0, 4: 130.0}), BUFFER_MS, sizes=[1, 2, 4], cycle_ms=CYCLE_MS) == 3


# ── a replica's robots, run on a clock ───────────────────────────────────────────────────────────────────────────────

SLOW_MS = {1: 256.0, 2: 273.0, 4: 303.0, 8: 377.0}     # SmolVLA on an L4, as its server timed each batch size


def test_starts_are_spread_evenly_over_a_cycle_and_the_same_every_time():
    """Two robots are run from 64 starts, the second robot's request a 64th of a cycle further after the first's from
    one to the next (the first 32 of them a 32nd apart); more robots from 128, each further robot at starts of its own
    all over the cycle."""
    cycle = 8 / 15
    two = slack.starts(2, cycle)
    gaps = sorted(begin[1] for begin in two)
    assert len(two) == slack.CLOCK_STARTS // 2 == 64 and all(begin[0] == 0.0 for begin in two)
    assert gaps[0] == pytest.approx(cycle / (3 * slack.CLOCK_STARTS))
    assert [b - a for a, b in zip(gaps, gaps[1:])] == pytest.approx([cycle / 64] * 63)
    coarse = sorted(begin[1] for begin in slack.starts(2, cycle, 32))
    assert [b - a for a, b in zip(coarse, coarse[1:])] == pytest.approx([cycle / 32] * 31)
    four = slack.starts(4, cycle)
    assert len(four) == slack.CLOCK_STARTS == 128 and all(0.0 <= at < cycle for begin in four for at in begin)
    for robot in (1, 2, 3):                                  # no eighth of the cycle without a start of each robot
        assert {int(begin[robot] / cycle * 8) for begin in four} == set(range(8))
    assert slack.starts(4, cycle) == four and slack.starts(1, cycle) == [[0.0]]


def test_on_a_clock_four_robots_of_a_slow_model_are_late_however_they_start_and_two_are_not():
    """Batches of 1, 2, 4 and 8 in 256, 273, 303 and 377 ms, a 15 Hz policy with 467 ms of buffer, robots 20 ms away.
    Four robots run as two pairs in turn, each request behind the other pair's batch: late in 127 starts of 128 (in
    the one where their requests arrive within 4 ms they share a batch of four), as four simulated robots were on the
    GPU (95.4% and 96.8% of their responses, in two runs). Two robots have every response back in time in every start,
    at a load of up to 0.96: two batches of one in each cycle. The clock runs a batch of two at 280 ms, its 273 at
    the clock's resolution."""
    table, sizes = _timed(SLOW_MS), sorted(SLOW_MS)
    cycle = AT_15["play_s"] - AT_15["buffer_s"]
    four = [slack.on_a_clock(table, AT_15, [20.0] * 4, begin, sizes=sizes) for begin in slack.starts(4, cycle)]
    assert sum(late > 0 for late, _, _ in four) == len(four) - 1 == 127
    assert sum(late for late, _, _ in four) > 0.95 * sum(total for _, total, _ in four)
    two = [slack.on_a_clock(table, AT_15, [20.0] * 2, begin, sizes=sizes) for begin in slack.starts(2, cycle)]
    assert all(late == 0 for late, _, _ in two)
    assert max(load for _, _, load in two) == pytest.approx(2 * 256.0 / PACED_CYCLE_MS, abs=0.005)       # 0.96
    assert min(load for _, _, load in two) == pytest.approx(280.0 / PACED_CYCLE_MS, abs=0.005)           # one of two


def test_a_slow_models_capacity_is_one_where_a_formula_for_its_load_gave_eight():
    """The same model: a batch of eight answers within the buffer (377 + 20 + 50), and the formula for the load took
    eight robots to share it, where four were late on the GPU. On a clock two robots are answered in time at a load of
    0.96, past LOAD_MAX: the capacity is one. Asked as robots arrive one at a time, Seating stops at the same
    number."""
    table, sizes = _timed(SLOW_MS), sorted(SLOW_MS)
    assert slack.per_replica(table, PACED_BUFFER_MS, sizes=sizes, outside_ms=20.0) == 8          # by the buffer alone
    assert slack.per_replica(table, PACED_BUFFER_MS, sizes=sizes, outside_ms=20.0, cycle_ms=PACED_CYCLE_MS) == 1
    assert not slack.in_time(table, AT_15, [20.0, 20.0], sizes=sizes)
    seating = slack.Seating(table, PACED_BUFFER_MS, sizes=sizes, cycle_ms=PACED_CYCLE_MS)
    robots: list = []
    for i in range(8):
        robots.append((f"r{i}", 20.0))
        robots = robots[:len(robots) - len(seating.seat(robots)[1])]
    assert len(robots) == 1


@pytest.mark.parametrize("forward", [{1: 198.0, 2: 320.0, 4: 565.0}, {1: 228.0, 2: 277.0, 4: 400.0}])
def test_two_models_keep_the_two_robots_that_were_in_time_on_a_gpu(forward):
    """π0.5 and GR00T N1.6 on an L4, 15 Hz, robots 20 ms away: two simulated robots had no late response on the GPU
    (0.4% in one run of eight), and the capacity is two."""
    table, sizes = _timed(forward), sorted(forward)
    assert slack.per_replica(table, PACED_BUFFER_MS, sizes=sizes, outside_ms=20.0, cycle_ms=PACED_CYCLE_MS) == 2


def test_robots_late_in_under_a_twentieth_of_their_responses_are_within_capacity_and_over_it_are_not():
    """π0.5 on an L4 (batches of 1, 2 and 3 in 195, 312 and 429 ms), a 15 Hz policy. Two robots 100 ms away have a late
    response only while their requests fall 5 to 25 ms apart, the second too late to share the first one's batch: 5
    starts of 64, and 3.9% of the responses over the starts. That is within what IN_TIME_MIN leaves, so the two are in
    time and the capacity is 2; two such robots on the GPU had no late response in 225. The same two 120 ms away have
    8.6% late over the starts, at the same load, and are not in time."""
    table, sizes = _timed({1: 195.0, 2: 312.0, 3: 429.0}), [1, 2, 3]
    cycle = AT_15["play_s"] - AT_15["buffer_s"]

    def late_share(trip: float) -> tuple[float, int, float]:
        runs = [slack.on_a_clock(table, AT_15, [trip, trip], begin, sizes=sizes, early=True)
                for begin in slack.starts(2, cycle)]
        return (sum(late / total for late, total, _ in runs) / len(runs), sum(late > 0 for late, _, _ in runs),
                max(load for _, _, load in runs))

    share, starts_late, load = late_share(100.0)
    assert share == pytest.approx(0.039, abs=0.002) and starts_late == 5 and load <= slack.LOAD_MAX
    assert 0.0 < share <= 1.0 - slack.IN_TIME_MIN and slack.in_time(table, AT_15, [100.0, 100.0], sizes=sizes)
    assert slack.per_replica(table, PACED_BUFFER_MS, sizes=sizes, outside_ms=100.0, cycle_ms=PACED_CYCLE_MS) == 2
    share, _, load = late_share(120.0)
    assert share == pytest.approx(0.086, abs=0.002) and load <= slack.LOAD_MAX
    assert not slack.in_time(table, AT_15, [120.0, 120.0], sizes=sizes)
    for gap_ms, late in ((0, False), (4, False), (5, True), (25, True), (26, False), (150, False)):
        got, _, _ = slack.on_a_clock(table, AT_15, [100.0, 100.0], [0.0, gap_ms / 1000.0], sizes=sizes)
        assert (got > 0) == late, gap_ms


def test_a_request_sent_late_is_due_when_the_chunk_before_it_runs_out():
    """20 Hz, 15 valid steps, the next requested at the 4th: a request every 200 ms, 550 ms of buffer. One robot 100 ms
    away of a model whose batch of one takes 300 ms has each response back 400 ms after its request, past the 200 ms,
    so each next request is sent as the response arrives, with 750 - 400 = 350 ms of its chunk left, and is back 400 ms
    later: after the chunk ran out, though within the 550 ms a request sent on time has."""
    play = slack.playback(hz=20, valid_len=15, replan_at=4)
    late, total, _ = slack.on_a_clock(_timed({1: 300.0}), play, [100.0], [0.0], sizes=[1])
    assert total > 0 and late == total


@pytest.mark.parametrize("trip", [47.0, 100.0])
def test_the_load_is_at_the_rate_robots_ask_not_the_rate_a_slow_replica_answers(trip):
    """10 Hz, 40 valid steps, the next requested at the 10th: a request every 1,000 ms, 3,000 ms of buffer. A batch of
    one takes 1,088 ms, so a robot sends its next request as its response arrives, in time, and the replica runs all
    the time less the robot's RTT: 0.96 of it for a robot 47 ms away, 0.92 for one 100 ms away. The load is what one
    request from each robot in a cycle takes: 1.09 with one robot, and with two 1.15 where they share every batch,
    more where they do not, at either distance, so the capacity is one at both. By the share of the time the replica
    ran, two robots 100 ms away were within LOAD_MAX and the same two 47 ms away were not."""
    play = slack.playback(hz=10, valid_len=40, replan_at=10)
    table, sizes = _timed({1: 1088.0, 2: 1152.0, 4: 1280.0, 8: 1600.0}), [1, 2, 4, 8]
    late, total, load = slack.on_a_clock(table, play, [trip], [0.0], sizes=sizes)
    assert total > 0 and late == 0 and load == pytest.approx(1.088, abs=0.001)
    for begin in slack.starts(2, 1.0, 4):
        late, total, load = slack.on_a_clock(table, play, [trip, trip], begin, sizes=sizes)
        assert total > 0 and late == 0 and 1.151 <= load <= 2 * 1.088 + 0.001
    assert not slack.in_time(table, play, [trip, trip], sizes=sizes)
    assert slack.per_replica(table, 3000.0, sizes=sizes, cycle_ms=1000.0, outside_ms=trip) == 1


SMALL_MS = {1: 40.0, 2: 45.0, 4: 55.0, 8: 75.0}
EARLY_REPLAN = slack.playback(hz=20.0, valid_len=15, replan_at=4)


@pytest.mark.parametrize("forward, play, robots, repeats_in", [
    (SLOW_MS, AT_15, 2, 1),                # two robots, the same batches every cycle
    (SMALL_MS, AT_15, 3, 2),               # three: a batch of two and a batch of one, the pair changing every cycle
    (SMALL_MS, AT_15, 5, 4),
    (SLOW_MS, AT_15, 4, None),             # late: robots that send as their responses arrive
    ({1: 80.0, 2: 110.0, 4: 170.0, 8: 290.0}, EARLY_REPLAN, 4, None),      # robots that wait for their responses
])
def test_a_run_that_returns_early_has_the_outcome_of_the_whole_run(forward, play, robots, repeats_in):
    """A run on a clock that returns once it repeats itself (early, as in_time asks) against the same run of every
    cycle: the share of late responses and the load are the same. Where no response is late and each robot sends
    every cycle, it returns within the first cycles, with the responses of the stretch that repeats: one, two or four
    cycles here."""
    table, sizes = _timed(forward), sorted(forward)
    cycle = play["play_s"] - play["buffer_s"]
    for begin in slack.starts(robots, cycle, 8):
        late, total, load = slack.on_a_clock(table, play, [20.0] * robots, begin, sizes=sizes)
        soon, some, its_load = slack.on_a_clock(table, play, [20.0] * robots, begin, sizes=sizes, early=True)
        assert 0 < some <= total and soon / some == pytest.approx(late / total, abs=0.03)
        if not late:
            assert its_load == pytest.approx(load, abs=0.002)
        if repeats_in is not None:
            assert some in [robots * cycles for cycles in range(1, repeats_in + 1)]


@pytest.mark.parametrize("ms, up", [(0.4, 1.0), (40.0, 40.0), (63.2, 64.0), (65.0, 66.0), (198.0, 200.0),
                                    (256.0, 256.0), (273.0, 280.0), (377.0, 384.0), (1094.0, 1120.0),
                                    (312.00000000000006, 312.0), (312.001, 320.0)])
def test_the_clock_rounds_a_batch_time_up_by_three_percent_at_most(ms, up):
    """312.00000000000006 is 312 ms as the difference of two clock readings gave it, and is run as 312."""
    assert slack._up(ms) == up and slack._up(up) == up and up <= math.ceil(ms) * (1 + 1 / 32)


def test_the_clock_rounds_an_rtt_up_to_ten_ms_and_takes_the_default_for_none():
    assert [slack._far(ms) for ms in (None, 0.0, 9.0, 20.0, 21.0, 47.0, 100.0)] == [100, 0, 10, 20, 30, 50, 100]


def _every_start(table, play, trips, sizes) -> bool:
    """in_time's verdict by running every start: at least IN_TIME_MIN of the responses in time, each start counting
    the same, and no start's load above LOAD_MAX."""
    runs = [slack.on_a_clock(table, play, trips, begin, sizes=sizes, early=True)
            for begin in slack.starts(len(trips), play["play_s"] - play["buffer_s"])]
    late = sum(late / total for late, total, _ in runs if total)
    return (late <= (1.0 - slack.IN_TIME_MIN) * len(runs) + 1e-9
            and all(load <= slack.LOAD_MAX + 1e-9 for _, _, load in runs))


def test_a_verdict_stops_at_the_start_that_decides_it_and_is_the_verdict_of_every_start(monkeypatch):
    """in_time stops running starts once its verdict cannot change. Not in time: at the first start that makes too many
    responses late, or whose load is above LOAD_MAX. In time: once the starts left could all be late without making
    too many, 3 starts of 64 and 6 of 128, where the robots' batches cannot fill a cycle above LOAD_MAX however their
    requests fall. Over 5 timing tables, 3 RTTs and 2 to 5 robots, the verdict is that of running every start."""
    runs, run = [], slack.on_a_clock

    def counted(*args, **kwargs):
        runs.append(1)
        return run(*args, **kwargs)

    tables = [{1: 195.0, 2: 312.0, 3: 429.0}, {1: 228.0, 2: 277.0, 4: 400.0}, SLOW_MS, SMALL_MS,
              {1: 234.0, 2: 254.0, 4: 299.0, 8: 382.0, 16: 550.0}]
    differ = []
    for forward in tables:
        table, sizes = _timed(forward), sorted(forward)
        for trip in (20.0, 60.0, 100.0):
            for robots in range(2, min(5, sizes[-1]) + 1):
                want = _every_start(table, AT_15, [trip] * robots, sizes)
                slack._CLOCKED.clear()
                if slack.in_time(table, AT_15, [trip] * robots, sizes=sizes) != want:
                    differ.append((forward, trip, robots, want))
    assert not differ, differ
    monkeypatch.setattr(slack, "on_a_clock", counted)
    slack._CLOCKED.clear()
    pi05 = _timed({1: 195.0, 2: 312.0, 3: 429.0})
    assert slack.in_time(pi05, AT_15, [20.0, 20.0], sizes=[1, 2, 3]) and len(runs) == 64 - 3     # none late: 61 run
    runs.clear()
    assert slack.in_time(_timed(SMALL_MS), AT_15, [20.0] * 3, sizes=sorted(SMALL_MS)) and len(runs) == 128 - 6
    runs.clear()
    assert not slack.in_time(_timed(SLOW_MS), AT_15, [20.0, 20.0], sizes=sorted(SLOW_MS)) and len(runs) < 64
    runs.clear()
    # two batches of one can fill 0.96 of a cycle: a start left unrun could be above LOAD_MAX, so every one is run
    slower = _timed({1: 256.0, 2: 262.0})
    assert slack._most_load(slower, 2, PACED_CYCLE_MS / 1000.0, [1, 2]) > slack.LOAD_MAX
    assert slack.in_time(slower, AT_15, [20.0, 20.0], sizes=[1, 2]) == _every_start(slower, AT_15, [20.0, 20.0], [1, 2])


def test_a_verdict_is_kept_at_the_clocks_resolution_and_run_once():
    """in_time keeps its verdict by what the clock runs: each listed size's times rounded up by 3% at most and each RTT
    up to 10 ms. A table a few ms faster and robots a few ms nearer ask the same question, and no run follows; a robot
    21 ms away is run as one 30 ms away, a new question."""
    table, sizes = _timed(SLOW_MS), sorted(SLOW_MS)
    slack._CLOCKED.clear()
    assert not slack.in_time(table, AT_15, [20.0, 20.0], sizes=sizes)
    kept = dict(slack._CLOCKED)
    assert len(kept) == 1
    faster = _timed({1: 254.0, 2: 275.0, 4: 300.0, 8: 380.0})
    assert not slack.in_time(faster, AT_15, [12.0, 19.0], sizes=sizes) and slack._CLOCKED == kept
    assert not slack.in_time(table, AT_15, [20.0, 21.0], sizes=sizes) and len(slack._CLOCKED) == 2


# ── one replica, each robot by its own RTT ───────────────────────────────────────────────────────────────────────────

def _seat(table, buffer_ms, *, sizes, outsides=None, cycle_ms=None, most=slack.MAX_ROBOTS_PER_REPLICA):
    """One replica's capacity with the robots on it now (slack.Seating.seat): (seats, over)."""
    seating = slack.Seating(table, buffer_ms, sizes=sizes, cycle_ms=cycle_ms, most=most)
    return seating.seat(list((outsides or {}).items()))


def _seats(outsides: dict, most: int = 4) -> tuple:
    """One replica with sizes [1, 2, 4] whose forward passes take 100 / 120 / 130 ms, at 20 Hz: 350 ms of buffer, a 400
    ms cycle."""
    return _seat(_timed(FORWARD), BUFFER_MS, sizes=[1, 2, 4], outsides=outsides, cycle_ms=CYCLE_MS, most=most)


def test_the_module_docstrings_example_seats_four_and_names_the_far_robot_over():
    play = slack.playback(hz=20, valid_len=15, replan_at=8)
    seating = slack.Seating(_timed({1: 80.0, 2: 110.0, 4: 130.0}), play["buffer_s"] * 1000, sizes=[1, 2, 4],
                            cycle_ms=(play["play_s"] - play["buffer_s"]) * 1000)
    assert seating.seat([("near", 30.0), ("far", 200.0)]) == (4, ["far"])


def test_a_robots_fit_is_the_largest_listed_batch_that_serves_it_in_time_at_its_own_rtt():
    seating = slack.Seating(_timed(FORWARD), BUFFER_MS, sizes=[1, 2, 4], cycle_ms=CYCLE_MS)
    # took + RTT + 50 <= 350: 130 at an RTT of 100 or 170 ms, 120 at 175, 100 at 200, nothing at 300
    assert [seating.fit(ms) for ms in (None, 30.0, 170.0, 175.0, 200.0, 300.0)] == [4, 4, 4, 2, 1, 0]


@pytest.mark.parametrize("forward, buffer_ms, cycle_ms", [
    (FORWARD, BUFFER_MS, CYCLE_MS),                                            # three a replica by its load
    (FORWARD, BUFFER_MS, None),                                                # no cycle known: the buffer alone
    (PACED_FORWARD, PACED_BUFFER_MS, PACED_CYCLE_MS),                          # two by its load
    ({1: 50.0, 2: 60.0, 4: 80.0, 8: 120.0}, 300.0, 250.0),
    ({1: 400.0, 2: 450.0}, BUFFER_MS, CYCLE_MS),                               # too slow for anyone: one, as ever
])
def test_robots_with_the_same_rtt_seat_what_per_replica_does(forward, buffer_ms, cycle_ms):
    table, sizes = _timed(forward), sorted(forward)
    alike = slack.per_replica(table, buffer_ms, sizes=sizes, cycle_ms=cycle_ms)
    assert _seat(table, buffer_ms, sizes=sizes, cycle_ms=cycle_ms) == (alike, [])
    # as many robots on it as its capacity, and two more that reported nothing: those two are over, the last by name
    robots = {f"r{i}": None for i in range(alike + 2)}
    assert _seat(table, buffer_ms, sizes=sizes, outsides=robots, cycle_ms=cycle_ms) == (
        alike, [f"r{alike}", f"r{alike + 1}"])


def test_a_robot_far_away_never_lowers_the_seats_of_the_robots_nearer_than_it():
    # by the furthest robot's RTT, as per_replica computes a replica's capacity: at 200 ms no batch of two is in time
    # (120 + 200 + 50 > 350), so the capacity is one robot per replica, with none for the two near ones beside it
    assert slack.per_replica(_timed(FORWARD), BUFFER_MS, sizes=[1, 2, 4], outside_ms=200.0, cycle_ms=CYCLE_MS) == 1
    # each by its own RTT: the near ones keep their places, with two more for robots at ROUND_TRIP_MS, and the far
    # one is the one named to move. Alone, a batch of one answers it (100 + 200 + 50 = 350); beside them its request
    # may wait behind their batch of two (100 + 120 + 200 + 50 > 350: a replica runs one batch at a time)
    assert _seats({"near-1": 30.0, "near-2": 30.0, "far": 200.0}) == (4, ["far"])
    assert _seats({"far": 200.0}) == (1, [])                       # alone: in time, and no one at 100 ms fits beside it


def test_when_a_replica_cannot_seat_them_all_the_furthest_are_over():
    # the far one's request may wait behind the three near ones' batch (100 + 130 + 200 + 50 > 350): it is the one
    # over, wherever it is listed, and a robot at ROUND_TRIP_MS still fits beside the three
    assert _seats({"far": 200.0, "a": 30.0, "b": 40.0, "c": 50.0}) == (4, ["far"])


def test_a_robot_late_even_alone_is_over_beside_others_and_seated_alone():
    # 300 ms away: not even a batch of one answers it in time (100 + 300 + 50 > 350), where a robot at ROUND_TRIP_MS
    # is answered in time; beside a nearer robot it is the one over, and three robots at ROUND_TRIP_MS fit beside the
    # near one
    assert _seats({"near": 30.0, "late": 300.0}) == (4, ["late"])
    # alone on a replica it is admitted, as an empty replica admits one; with a robot at ROUND_TRIP_MS beside it, whose
    # request may wait behind its batch (100 + 100 + 100 + 50 = 350); two would batch in 120 (370)
    assert _seats({"late": 300.0}) == (2, [])


@pytest.mark.parametrize("forward, outsides, seated", [
    # the far one's request, come as the near ones' batch of two runs (120 ms), would leave 43 ms late: the near two
    # keep their places, with one more for a robot at ROUND_TRIP_MS
    ({1: 100.0, 2: 120.0, 3: 140.0}, {"N1": 40.0, "N2": 40.0, "F": 300.0}, (3, ["F"])),
    # the near ones' requests, come as the far one's batch runs (100 ms), would leave 3 and 103 ms late. 280 ms away,
    # the two near ones keep their places: on a clock 98.4% of their responses are back in time (one is late in 2
    # starts of 64, where its request arrives as the other's batch of one runs)
    ({1: 100.0, 2: 130.0, 3: 150.0}, {"N1": 280.0, "N2": 280.0, "F": 310.0}, (2, ["F"])),
])
def test_a_batch_of_one_fit_is_in_time_behind_another_fits_batch_or_the_furthest_moves(forward, outsides, seated):
    """A replica runs one batch at a time, and a robot of one fit is not held for another's, so a request may come as
    another fit's batch runs: each fit's batch + the longest of another's + its furthest robot's RTT + MARGIN_MS
    must be within the buffer (15 Hz, 467 ms of buffer, sizes 1 to 3). Seated by the fits alone, a replica held all
    three and a request came late. The furthest is the one over; alone it is in time."""
    table, sizes = _timed(forward), sorted(forward)
    assert _seat(table, PACED_BUFFER_MS, sizes=sizes, outsides=outsides, cycle_ms=PACED_CYCLE_MS) == seated
    assert _seat(table, PACED_BUFFER_MS, sizes=sizes, outsides={"F": outsides["F"]}, cycle_ms=PACED_CYCLE_MS)[1] == []


def test_robots_that_reported_nothing_are_assumed_to_have_the_default_rtt():
    assert _seats({"a": None, "b": None}) == _seats({"a": slack.ROUND_TRIP_MS, "b": slack.ROUND_TRIP_MS})


# ── how long an open batch may wait ──────────────────────────────────────────────────────────────────────────────────

HZ = 15.0
PLAY = 15 / HZ                         # a response executes fully in 1 s
EVERY = 8 / HZ                         # a robot sends a request every 533 ms, counted from when it sent its last one
BUFFER = 7 / HZ                        # the playback time left when its robot sends a request: 467 ms
BATCH_S = {1: 0.100, 2: 0.120}         # a batch of one, of two


def _two() -> slack.Table:
    return _timed({size: s * 1000.0 for size, s in BATCH_S.items()})


def _aligner(seated=None) -> slack.Aligner:
    """`seated`: the robots admitted to its replica: every robot it has heard from, unless the test names them."""
    al = slack.Aligner(_two(), AT_15, seated=seated or (lambda: list(al._robots)))
    return al


def _serve(offsets: dict, rounds: int, batch_s: dict = BATCH_S, outside_ms: float | None = None) -> tuple:
    """Robots sending a request every EVERY from their first request, whatever their responses do (k steps counted from
    each request sent), each request to one replica that runs one batch at a time, its open batch held as the aligner
    decides: the batches in order, and each response's lead over the latest time it may leave: its request's arrival
    plus its buffer, less the RTT (`outside_ms` as each robot reports it; ROUND_TRIP_MS with none). `batch_s`: what a
    batch of each size takes."""
    table = _timed({size: s * 1000.0 for size, s in batch_s.items()})
    al = slack.Aligner(table, AT_15, seated=lambda: list(al._robots))
    trip = TRIP if outside_ms is None else outside_ms / 1000.0
    acts = sorted((off + i * EVERY, name) for name, off in offsets.items() for i in range(rounds))
    came: dict = {}
    batches: list = []
    leads: list = []
    state = {"open": [], "due": None, "free": 0.0}

    def run(at: float) -> None:
        batch = state["open"]
        done = max(at, state["free"]) + table.p95(len(batch)) / 1000.0
        leads.extend(came[r] + BUFFER - trip - done for r in batch)
        al.answered(list(batch), done)
        batches.append(sorted(batch))
        state.update(open=[], due=None, free=done)

    for t, name in acts:
        if state["open"] and state["due"] is not None and state["due"] <= t:
            run(state["due"])
        al.arrived(name, t, outside_ms=outside_ms)
        came[name] = t
        state["open"].append(name)
        hold = al.hold_s(state["open"], t)
        if hold <= 0.0:
            run(t)
        else:
            state["due"] = t + hold
    if state["open"]:
        run(state["due"])
    return batches, leads


@pytest.mark.parametrize("offset, together", [(0.05, True), (0.15, True), (0.30, False)])
def test_robots_requesting_on_their_own_clocks_share_a_batch_when_they_arrive_within_its_wait(offset, together):
    """B sends its request `offset` s after A, both every 533 ms from their own requests: from their second requests on,
    one that arrives within what the first can wait (467 ms less the RTT, a batch of two and the margin: 197 ms) shares
    its batch every round; one further off is answered alone, every round, and holding moves neither. Every response
    leaves the margin before its request's buffer runs out."""
    batches, leads = _serve({"A": 0.0, "B": offset}, rounds=12)
    assert all((b == ["A", "B"]) == together for b in batches[2:]), (offset, batches)
    assert leads and min(leads) >= MARGIN - 1e-9, (offset, leads)


def test_robots_spread_over_a_round_each_get_a_response_within_their_buffer():
    """Four robots spread over 400 ms of a 533 ms round: a batch takes those that arrive within what its first request
    can wait, the rest go in the next, and no response leaves later than the margin before its request's buffer runs
    out."""
    batches, leads = _serve({"A": 0.0, "B": 0.13, "C": 0.27, "D": 0.40}, rounds=12)
    assert max(len(b) for b in batches) >= 2, batches
    assert min(leads) >= MARGIN - 1e-9, leads


def test_a_request_waits_for_the_replicas_other_robots_until_its_robot_would_run_out_of_actions():
    """A's next request arrives at 0.533 with 7 steps of its response left: the open batch waits for B, admitted to the
    replica and due at 0.683 (a request every 533 ms, its last at 0.15), for those 467 ms less the RTT, a batch of two
    and the margin: until 0.73."""
    al = _aligner()
    al.arrived("A", 0.0)
    al.arrived("B", 0.15)
    al.answered(["A"], 0.0)
    al.answered(["B"], 0.3)
    al.arrived("A", EVERY)
    assert al.hold_s(["A"], EVERY) == pytest.approx(BUFFER - TRIP - BATCH_S[2] - MARGIN)


def test_a_batch_is_dispatched_once_every_robot_the_replica_seats_has_a_request_in_it():
    al = _aligner(seated=lambda: ["A", "B"])
    al.arrived("A", 0.0)
    al.arrived("B", 0.1)
    al.answered(["A", "B"], 0.2)
    al.arrived("A", 0.7)
    assert al.hold_s(["A"], 0.7) > 0.0
    al.arrived("B", 0.75)
    assert al.hold_s(["A", "B"], 0.75) == 0.0


def test_a_batch_is_dispatched_when_its_first_request_can_wait_no_longer():
    """The request of the open batch that can wait least sets how long it waits: B came first, so the batch of the two
    waits for C only until B's buffer, less the RTT, would leave a batch of three no time to run."""
    al = _aligner()
    for name, t in (("A", 0.0), ("B", 0.05), ("C", 0.1)):
        al.arrived(name, t)
        al.answered([name], t)
    al.arrived("B", 0.5)
    al.arrived("A", 0.55)
    last = 0.5 + BUFFER - TRIP - _two().p95(3) / 1000.0 - MARGIN  # B's response leaves by then
    assert al.hold_s(["B", "A"], 0.55) == pytest.approx(last - 0.55)
    assert al.hold_s(["B", "A"], last) == 0.0 and al.hold_s(["B", "A"], last + 0.1) == 0.0


def test_a_request_from_a_robot_with_no_chunk_executing_is_dispatched_at_once():
    """A robot's first request (it has no response from the replica being executed while it waits) is dispatched at
    once, with any batch it is in."""
    al = _aligner()
    for name, t in (("A", 0.0), ("C", 0.05)):
        al.arrived(name, t)
        al.answered([name], t)
    al.arrived("B", 0.1)                                              # B's first
    assert al.hold_s(["B"], 0.1) == 0.0
    al.arrived("A", EVERY)
    assert al.hold_s(["A", "B"], EVERY) == 0.0                        # C has yet to arrive, but B has no actions


def test_a_robot_held_round_after_round_is_held_still():
    """Holding a response does not move the next request of a robot that sends it k steps after its previous request,
    and waiting is not abandoned for that: every round its request waits for the replica's other robot as long as its
    buffer allows."""
    al = _aligner()
    al.arrived("B", 0.0)
    al.answered(["B"], 0.0)
    al.arrived("A", 0.0)
    al.answered(["A"], 0.0)
    t = 0.0
    for _ in range(6):
        t += EVERY
        al.arrived("A", t)
        hold = al.hold_s(["A"], t)
        assert hold > 0.0
        al.answered(["A"], t + hold + BATCH_S[1])
        al.arrived("B", t + 0.01)                                     # B is heard from, in another batch
        al.answered(["B"], t + 0.02)


def test_a_robot_the_replica_does_not_seat_or_that_left_is_waited_for_by_none_of_its_batches():
    """B is on another replica now: this replica's batches wait only for the robots admitted to it. A robot that
    reported it is done (Aligner.left) is no longer waited for either."""
    seated = ["A", "B"]
    al = _aligner(seated=lambda: seated)
    for name, t in (("A", 0.0), ("B", 0.05)):
        al.arrived(name, t)
        al.answered([name], t)
    al.arrived("A", EVERY)
    assert al.hold_s(["A"], EVERY) > 0.0
    seated.remove("B")
    assert al.hold_s(["A"], EVERY) == 0.0
    seated.append("B")
    assert al.hold_s(["A"], EVERY) > 0.0
    al.left("B")
    assert al.hold_s(["A"], EVERY) == 0.0 and "B" not in al._robots


def test_a_policy_with_no_playback_holds_no_request_until_it_is_told_how_its_chunks_execute():
    """Nothing states how long its robots can wait: none is held. When playback is given later (Aligner.replay), its
    requests from then on are scheduled."""
    al = slack.Aligner(_two(), None, seated=lambda: ["A", "B"])
    al.arrived("A", 0.0)
    al.arrived("B", 0.05)
    al.answered(["A", "B"], 0.2)
    al.arrived("B", 0.5)
    assert al.hold_s(["A"], 0.45) == 0.0 and al.due("A") is None
    al.replay(AT_15)
    al.arrived("A", 0.0)
    al.arrived("B", 0.15)
    al.answered(["A"], 0.0)
    al.answered(["B"], 0.3)
    al.arrived("A", EVERY)
    assert al.hold_s(["A"], EVERY) == pytest.approx(BUFFER - TRIP - BATCH_S[2] - MARGIN)
    assert al.due("A") == pytest.approx(EVERY + BUFFER - TRIP) and al.due("nobody") is None


def test_a_robot_silent_for_a_few_periods_is_no_longer_waited_for():
    """A robot that disconnects without notice: one not heard from in GONE_PERIODS of its period is dropped as the next
    request arrives, and no batch waits for it."""
    al = _aligner()
    al.arrived("A", 0.0)
    al.arrived("B", 0.1)
    al.answered(["A", "B"], 0.1)
    late = 0.1 + slack.GONE_PERIODS * PLAY + 0.01
    al.arrived("B", late)
    al.answered(["B"], late)
    al.arrived("B", late + EVERY)
    assert "A" not in al._robots and al.hold_s(["B"], late + EVERY) == 0.0


def test_a_batch_does_not_wait_for_robots_that_cannot_come_before_it_must_go():
    """C and D, admitted to the replica, arrived at 0 and were answered: each sends its next request k steps after its
    previous one at the earliest (one request in flight), so neither can arrive before 0.47, which is past 0.37, when
    the open batch of A and B must be dispatched to be answered in time. It is dispatched at once: waiting for them
    would only idle the replica while robots queue behind it."""
    al = _aligner()
    for name in ("A", "B"):
        al.arrived(name, -EVERY)
    al.answered(["A", "B"], -0.4)                   # A's and B's actions last until 0.6
    for name in ("C", "D"):
        al.arrived(name, 0.0)
    al.answered(["C", "D"], 0.12)
    for name in ("A", "B"):
        al.arrived(name, 0.27)
    must_go = 0.6 - _two().p95(3) / 1000.0 - MARGIN  # their actions run out at 0.6, less a batch of three and margin
    assert must_go < 0.0 + EVERY - 1 / HZ            # before C's or D's next request can come in
    assert al.hold_s(["A", "B"], 0.27) == 0.0


def test_a_batch_still_waits_for_a_robot_that_can_come_before_it_must_go():
    """C arrived at -0.3: its next request may arrive from 0.17 on, before the open batch must be dispatched at 0.37, so
    the batch waits for it."""
    al = _aligner()
    for name in ("A", "B"):
        al.arrived(name, -EVERY)
    al.answered(["A", "B"], -0.4)
    al.arrived("C", -0.3)
    al.answered(["C"], -0.2)
    for name in ("A", "B"):
        al.arrived(name, 0.27)
    must_go = 0.6 - _two().p95(3) / 1000.0 - MARGIN
    assert al.hold_s(["A", "B"], 0.27) == pytest.approx(must_go - 0.27)


def test_a_batch_does_not_wait_for_a_robot_due_just_after_it_must_go():
    """C arrived at -0.14 and sends a request every 533 ms: its next is due at 0.39, after the open batch of A and B
    must be dispatched at 0.37, though with a faster uplink it could arrive a control step sooner, at 0.33. The batch
    is dispatched at once. Held until 0.37 it would run without C all the same, and C's request would wait behind it
    for the whole batch."""
    al = _aligner()
    for name in ("A", "B"):
        al.arrived(name, -EVERY)
    al.answered(["A", "B"], -0.4)
    al.arrived("C", -0.14)
    al.answered(["C"], -0.04)
    for name in ("A", "B"):
        al.arrived(name, 0.27)
    must_go = 0.6 - _two().p95(3) / 1000.0 - MARGIN
    assert -0.14 + EVERY - 1 / HZ < must_go < -0.14 + EVERY          # C could arrive before it, and is due after it
    assert al.hold_s(["A", "B"], 0.27) == 0.0


def _strays(by: float = 0.012):
    """An aligner that has heard six requests from B, 12 ms early and late in turn: (the aligner, when B's next is due
    by its period, how much later it may well arrive: three times the spread of its gaps, a control step at most)."""
    al = _aligner()
    for i in range(6):
        at = (i - 6) * EVERY + (by if i % 2 else 0.0)
        al.arrived("B", at)
        al.answered(["B"], at + 0.1)
    b = al._robots["B"]
    return al, b.seen + b.period_s, min(1 / HZ, 3.0 * b.spread_s)


def test_a_batch_waits_only_for_a_robot_due_safely_before_it_must_go():
    """B's requests have arrived 12 ms early and late in turn. An open batch that must be dispatched 10 ms after B's
    next request is due is dispatched at once: B may as well arrive after it, and would then wait behind it for the
    whole batch, where a B that arrives before only waits for what is left of a batch dispatched earlier. A batch that
    must be dispatched after even a late B is due waits for it."""
    wait = BUFFER - TRIP - BATCH_S[2] - MARGIN                # what A's request can wait for a batch of two: 197 ms
    al, due, strays = _strays()
    assert 0.020 < strays <= 1 / HZ
    at = due + 0.010 - wait                                    # A arrives: its batch must go 10 ms after B is due
    al.arrived("A", at - EVERY)
    al.answered(["A"], at - EVERY + 0.1)
    al.arrived("A", at)
    assert al.hold_s(["A"], at) == 0.0
    al, due, strays = _strays()
    at = due + strays + 0.005 - wait                           # its batch must go after even a late B is due
    al.arrived("A", at - EVERY)
    al.answered(["A"], at - EVERY + 0.1)
    al.arrived("A", at)
    assert al.hold_s(["A"], at) == pytest.approx(wait)


SLOW_S = {1: 0.256, 2: 0.273}          # a model whose batch takes more than half of the buffer


@pytest.mark.parametrize("apart", [0.06, 0.13, 0.15, 0.18, 0.30])
def test_two_robots_of_a_slow_model_are_answered_in_time_whatever_the_gap_between_their_requests(apart):
    """Batches of one and two in 256 and 273 ms, robots 20 ms away, a 467 ms buffer: a request can wait 124 ms for a
    batch of two. A robot whose request comes within that of the first shares its batch. One further behind is due
    after the batch must go, so the first request is dispatched at once and the second runs behind it. Either way every
    response leaves within its request's buffer, from each robot's second request on (its first is never held). Held
    for a robot that could arrive a control step sooner than it does, the first request ran alone 124 ms later, and the
    second, behind it, was late for gaps from 124 to 190 ms."""
    batches, leads = _serve({"A": 0.0, "B": apart}, rounds=12, batch_s=SLOW_S, outside_ms=20.0)
    assert min(leads[2:]) >= 0.0, (apart, min(leads[2:]))
    assert all((b == ["A", "B"]) == (apart <= 0.124) for b in batches[2:]), (apart, batches)


# ── ... by what each robot reports, and what its batches took ────────────────────────────────────────────────────────

def _held(outside_ms=None, entered_ago: float = 0.0, served=None, left_ms=None) -> float:
    """How long robot a's request may wait for robot b. b is admitted and its next request is due about now (k / hz =
    0.4 s after its last), so it may join a's batch; a's last response left 50 ms ago (its actions last 750 ms, so the
    buffer, not running out of actions, is the binding limit). a's request entered the server `entered_ago` before it
    reached its replica, with its robot reporting `outside_ms`, and `left_ms` left of the chunk it is executing. At 20
    Hz: 350 ms of buffer."""
    now = 10.0
    al = slack.Aligner(_timed(FORWARD), AT_20, seated=lambda: ["a", "b"], served=served)
    al.arrived("b", now - 0.4)
    al.arrived("a", now - 0.5)
    al.answered(["a"], now - 0.05)
    al.arrived("a", now, entered=now - entered_ago, outside_ms=outside_ms, left_ms=left_ms)
    return al.hold_s(["a"], now)


def test_a_robot_that_reports_nothing_is_assumed_to_have_the_default_rtt():
    # its buffer, less the 100 ms RTT, the forward of a batch of two and the margin
    assert _held() == pytest.approx(BUFFER_MS / 1000 - TRIP - FORWARD[2] / 1000 - MARGIN)


def test_a_robot_whose_requests_spend_131_ms_outside_its_server_leaves_its_batch_that_much_less_to_wait():
    assert _held(outside_ms=131.0) == pytest.approx(BUFFER_MS / 1000 - 0.131 - FORWARD[2] / 1000 - MARGIN)
    assert _held(outside_ms=131.0) == pytest.approx(_held() - 0.031)


def test_a_requests_buffer_counts_from_when_it_reached_its_server_not_from_when_it_reached_its_replica():
    assert _held(entered_ago=0.04) == pytest.approx(_held() - 0.04)


def test_once_a_size_was_served_enough_its_served_time_not_its_forward_is_what_a_batch_takes():
    assert _held(served=_served({2: 150.0})) == pytest.approx(BUFFER_MS / 1000 - TRIP - 0.150 - MARGIN)
    # the measurement in the module's docstring: 131 ms outside the server and 197 ms in it, so nothing is left to wait
    # and the request is dispatched at once
    assert _held(outside_ms=131.0, served=_served({2: 197.0})) == 0.0


def test_a_batch_waits_by_its_tightest_robot():
    now = 10.0
    al = slack.Aligner(_timed(FORWARD), AT_20, seated=lambda: ["a", "b", "c"])
    for robot in "abc":
        al.arrived(robot, now - 0.5)
    al.answered(["a", "b"], now - 0.05)
    al.arrived("a", now, outside_ms=50.0)
    al.arrived("b", now, outside_ms=100.0)
    # b's response must leave first: its buffer less its 100 ms RTT, less a batch of three (timed as four) and the
    # margin
    assert al.hold_s(["a", "b"], now) == pytest.approx(BUFFER_MS / 1000 - 0.100 - FORWARD[4] / 1000 - MARGIN)


def test_a_robot_keeps_what_it_last_reported_until_it_reports_otherwise():
    now = 10.0
    al = slack.Aligner(_timed(FORWARD), AT_20, seated=lambda: ["a", "b"])
    al.arrived("b", now - 0.4)                     # its next request due about now: it may fall into a's batch
    al.arrived("a", now - 0.5, outside_ms=131.0)
    al.answered(["a"], now - 0.05)
    al.arrived("a", now)                           # a request that reported nothing: its robot is still 131 ms away
    assert al.hold_s(["a"], now) == pytest.approx(BUFFER_MS / 1000 - 0.131 - FORWARD[2] / 1000 - MARGIN)
    assert al.outsides_ms(now) == {"a": 131.0, "b": None}            # b has reported nothing
    assert al.due("a") == pytest.approx(now + BUFFER_MS / 1000 - 0.131)


def test_a_robot_that_requests_late_is_held_by_the_time_it_reports_as_left_not_by_its_policys_buffer():
    """One request is in flight per robot: a response that arrived late moves its next request later than k / hz, so
    less of the chunk it is executing is left than (N - k) / hz."""
    assert _held(left_ms=300.0) == pytest.approx(0.300 - TRIP - FORWARD[2] / 1000 - MARGIN)
    assert _held(left_ms=200.0) == 0.0
    assert _held() == pytest.approx(BUFFER_MS / 1000 - TRIP - FORWARD[2] / 1000 - MARGIN)   # not reported: (N - k) / hz


# ── a far robot's request is dispatched separately, first ────────────────────────────────────────────────────────────

# At 15 Hz (AT_15: 467 ms left when a robot sends its request): a batch of one takes 100 ms, of two 120, of three 140.
BATCH_MS = {1: 100.0, 2: 120.0, 3: 140.0}
# two robots 40 ms from their server, one 300 ms: its request has 467 - 300 = 167 ms in the server, in which a batch
# of one and the margin (150 ms) fit and a batch of two (170) does not
OUTSIDE = {"N1": 40.0, "F": 300.0, "N2": 40.0}


def _open_at_0(outside: dict) -> slack.Aligner:
    """A replica serving `outside`'s robots, each heard from at -0.6 and answered at -0.5 (their actions last until
    0.5), each of their requests at 0 reporting its robot's RTT."""
    al = slack.Aligner(_timed(BATCH_MS), AT_15, seated=lambda: list(outside))
    for name, ms in outside.items():
        al.arrived(name, -0.6, outside_ms=ms)
    al.answered(list(outside), -0.5)
    for name, ms in outside.items():
        al.arrived(name, 0.0, outside_ms=ms)
    return al


def test_a_far_robots_request_is_dispatched_first_in_the_largest_batch_its_deadline_allows():
    """N1, F and N2 open at 0, a batch of three: N1's and N2's responses may leave by 0.427, F's by 0.167, and a batch
    of three (140 ms and the margin) would answer F after that. F is dispatched first, alone (a batch of one leaves it
    17 ms to spare, a batch of two none); N1 and N2 then go as a batch of two, by 0.27, within their deadlines. Two near
    robots alone go as one batch."""
    al = _open_at_0(OUTSIDE)
    assert al.split(["N1", "F", "N2"], 0.0, [1, 2, 3], free=False) == [1]
    assert al.split(["N1", "N2"], 0.0, [1, 2, 3], free=False) is None             # their time leaves the whole batch
    wide = _open_at_0({"N1": 40.0, "F": 150.0, "N2": 40.0})                        # 317 ms of its own: a batch of two
    assert wide.split(["N1", "F", "N2"], 0.0, [1, 2, 3], free=False) is None      # and of three fit it


def test_a_far_robot_is_dispatched_separately_only_where_the_requests_after_it_stay_in_time():
    """N2, with an RTT of 200 ms, may leave by 0.267: in the batch of three, at 0.19, it is in time; after F's batch of
    one, at 0.27, it is not, so F goes with them, unless another replica is free to take their batch at once (0.17)."""
    al = _open_at_0({"N1": 40.0, "F": 300.0, "N2": 200.0})
    assert al.split(["N1", "F", "N2"], 0.0, [1, 2, 3], free=False) is None
    assert al.split(["N1", "F", "N2"], 0.0, [1, 2, 3], free=True) == [1]


def test_a_robot_late_even_alone_is_dispatched_separately_where_later_requests_keep_what_the_whole_batch_left():
    """F, with an RTT of 400 ms, has 67 ms of its own: late even alone (100 ms). It is dispatched separately first where
    the requests after it keep what the batch of three (out at 0.14) left them, which is N1's deadline and margin (out
    at 0.22, due 0.427), and not where one would lose its margin: N2, with an RTT of 230 ms and due out by 0.237, had 97
    ms to spare in the whole batch and would have 17."""
    al = _open_at_0({"N1": 40.0, "F": 400.0, "N2": 40.0})
    assert al.split(["N1", "F", "N2"], 0.0, [1, 2, 3], free=False) == [1]
    tight = _open_at_0({"N1": 40.0, "F": 400.0, "N2": 230.0})
    assert tight.split(["N1", "F", "N2"], 0.0, [1, 2, 3], free=False) is None


def test_no_request_is_dispatched_separately_from_a_batch_with_an_untimed_robot_or_a_single_request():
    al = _open_at_0(OUTSIDE)
    assert al.split(["N1", "F", "stranger"], 0.0, [1, 2, 3], free=False) is None  # one of them never heard from
    assert al.split(["F"], 0.0, [1, 2, 3], free=False) is None                    # nothing to separate from
    assert al.split([], 0.0, [1, 2, 3], free=False) is None
    untimed = slack.Aligner(_timed(BATCH_MS), None, seated=lambda: list(OUTSIDE))
    assert untimed.split(["N1", "F", "N2"], 0.0, [1, 2, 3], free=False) is None   # a policy with no playback


def test_a_batch_waits_for_no_robot_its_own_time_keeps_out_of_it():
    """N1's request at 0 may wait until 0.257 (its response out by 0.427, less a batch of two and the margin). F's next
    request may arrive from 0.167, but F, with an RTT of 300 ms, has 167 ms of its own, short of a batch of two and the
    margin (170): it would be dispatched separately, so N1's batch does not wait for it; the batch does wait for a robot
    with an RTT of 40 ms in F's place."""
    al = slack.Aligner(_timed(BATCH_MS), AT_15, seated=lambda: ["N1", "F"])
    al.arrived("N1", -0.6, outside_ms=40.0)
    al.answered(["N1"], -0.5)
    al.arrived("F", -0.3, outside_ms=300.0)
    al.answered(["F"], -0.2)
    al.arrived("N1", 0.0, outside_ms=40.0)
    assert al.hold_s(["N1"], 0.0) == 0.0
    al.arrived("F", -0.3, outside_ms=40.0)                       # the same robot, near
    assert abs(al.hold_s(["N1"], 0.0) - 0.257) < 1e-3


def test_a_near_batch_with_time_left_still_waits_for_another_near_robot():
    """The cap on a far batch does not affect a near batch: a near robot whose own deadline still has room waits for
    another near robot to join."""
    al = slack.Aligner(_timed(BATCH_MS), AT_15, seated=lambda: ["N1", "N2"])
    al.arrived("N1", -0.6, outside_ms=40.0)
    al.answered(["N1"], -0.5)
    al.arrived("N2", -0.3, outside_ms=40.0)
    al.answered(["N2"], -0.2)
    al.arrived("N1", 0.0, outside_ms=40.0)
    assert abs(al.hold_s(["N1"], 0.0) - 0.257) < 1e-3           # N1 waits for N2, as its own deadline leaves room


# ── a far robot holds back none of the others ────────────────────────────────────────────────────────────────────────

# At 15 Hz, a robot asking every 533 ms: batches of 1, 2 and 4 in 230, 250 and 290 ms; the near robot 23 ms outside
# the server, the far one 250 ms further.
SLOW_FORWARD = {1: 230.0, 2: 250.0, 4: 290.0}
TRIPS = {"near": 23.0, "far": 273.0}


def _near_and_far(near_ms: float = TRIPS["near"]) -> slack.Aligner:
    """The near robot heard from at 0 and at EVERY, each request answered 230 ms later; the far one at 0.1, answered at
    0.33."""
    al = slack.Aligner(_timed(SLOW_FORWARD), AT_15, seated=lambda: list(TRIPS))
    for t in (0.0, EVERY):
        al.arrived("near", t, outside_ms=near_ms)
        al.answered(["near"], t + 0.23)
    al.arrived("far", 0.1, outside_ms=TRIPS["far"])
    al.answered(["far"], 0.33)
    return al


def test_a_far_robots_batch_waits_for_a_near_request_due_as_it_would_run():
    """The near robot sends a request every 533 ms, its next due at 2 * EVERY; the far one's request arrives 13 ms
    before it. Dispatched at once, the far batch runs 230 ms: the near request would wait for the whole batch, then run
    alone, and be out 444 ms after it arrived, using all of its own time and none of the margin. The far batch waits for
    the near request instead, at most a little past when it is due."""
    al = _near_and_far()
    now = 2 * EVERY - 0.013
    al.arrived("far", now, outside_ms=TRIPS["far"])
    hold = al.hold_s(["far"], now)
    assert 0.013 < hold <= 0.013 + AT_15["step_s"] + 1e-9, hold
    # 200 ms before the near request: behind the far batch it still leaves with the margin, so the far batch is not held
    soon = 2 * EVERY - 0.2
    al.arrived("far", soon, outside_ms=TRIPS["far"])
    assert al.hold_s(["far"], soon) == 0.0


def test_the_far_batchs_wait_for_a_near_request_is_capped_at_one_control_step():
    """A near request more than a control step away: the far batch waits a step at most and is then dispatched, never
    the whole gap: a far robot that is already late is not held tens of ms more for a request that is not about to
    arrive."""
    al = _near_and_far()
    now = 2 * EVERY - 0.078                              # the near request is 78 ms off, past a control step (67 ms)
    al.arrived("far", now, outside_ms=TRIPS["far"])
    assert 0.0 < al.hold_s(["far"], now) <= AT_15["step_s"] + 1e-9


def test_no_wait_for_a_robot_that_could_not_share_the_batch():
    al = _near_and_far(near_ms=200.0)               # 467 - 200 ms: no batch of two and margin fits, so no gain
    now = 2 * EVERY - 0.013
    al.arrived("far", now, outside_ms=TRIPS["far"])
    assert al.hold_s(["far"], now) == 0.0


# ── where a policy sets a wait ───────────────────────────────────────────────────────────────────────────────────────

# The grid over which a replica's capacity is pinned, each point as the clock has it: with no wait set, a change to the
# capacity shows here, point by point. Policies (hz, N, k; None: not given), each size's
# (p50, p95) as five runs (p50 - 5, p50 twice, p50 + 5, p95), the listed sizes, and robots by (name, RTT in ms).
POLICIES = {"P1": (15, 15, 8), "P2": (20, 15, 4), "P3": (20, 10, 10), "P4": (15, 30, None), "P5": (10, 40, 10)}
TABLES = {
    "T1": {1: (60.0, 80.0), 2: (90.0, 110.0), 4: (140.0, 170.0), 8: (240.0, 290.0)},
    "T2": {1: (150.0, 180.0), 2: (220.0, 260.0), 4: (350.0, 400.0), 8: (600.0, 700.0)},
    "T3": {1: (1073.0, 1093.6), 2: (1118.6, 1156.4), 4: (1202.0, 1282.8), 8: (1445.8, 1606.7)},
}
SIZES = {"S1": [1, 2, 4, 8], "S2": [2, 4]}
ROBOTS = {
    "R0": [],
    "R1": [("a", None)],
    "R2": [("a", 20.0), ("b", 60.0), ("c", 150.0), ("d", None)],
    "R3": [(f"r{i}", 47.0) for i in range(8)],
    "R4": [("far", 400.0), ("near", 10.0)],
}
# Each point's per_replica, per_replica at a 47 ms RTT, and Seating.seat of each of ROBOTS
PINNED = {
    'P1/T1/S1': (8, 8, (8, ()), (8, ()), (5, ()), (8, ()), (8, ('far',))),
    'P1/T1/S2': (4, 4, (4, ()), (4, ()), (4, ()), (4, ('r4', 'r5', 'r6', 'r7')), (4, ('far',))),
    'P1/T2/S1': (2, 2, (2, ()), (2, ()), (2, ('d', 'c')), (2, ('r2', 'r3', 'r4', 'r5', 'r6', 'r7')), (1, ('far',))),
    'P1/T2/S2': (1, 2, (1, ()), (1, ()), (2, ('d', 'c')), (2, ('r2', 'r3', 'r4', 'r5', 'r6', 'r7')), (1, ('far',))),
    'P1/T3/S1': (1, 1, (1, ()), (1, ()), (1, ('b', 'd', 'c')), (1, ('r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7')), (1,
                 ('far',))),
    'P1/T3/S2': (1, 1, (1, ()), (1, ()), (1, ('b', 'd', 'c')), (1, ('r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7')), (1,
                 ('far',))),
    'P2/T1/S1': (4, 4, (4, ()), (4, ()), (4, ()), (4, ('r4', 'r5', 'r6', 'r7')), (4, ('far',))),
    'P2/T1/S2': (4, 4, (4, ()), (4, ()), (4, ()), (4, ('r4', 'r5', 'r6', 'r7')), (4, ('far',))),
    'P2/T2/S1': (1, 1, (1, ()), (1, ()), (1, ('b', 'd', 'c')), (1, ('r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7')), (1,
                 ('far',))),
    'P2/T2/S2': (1, 1, (1, ()), (1, ()), (1, ('b', 'd', 'c')), (1, ('r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7')), (1,
                 ('far',))),
    'P2/T3/S1': (1, 1, (1, ()), (1, ()), (1, ('b', 'd', 'c')), (1, ('r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7')), (1,
                 ('far',))),
    'P2/T3/S2': (1, 1, (1, ()), (1, ()), (1, ('b', 'd', 'c')), (1, ('r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7')), (1,
                 ('far',))),
    'P3/T1/S1': (1, 1, (1, ()), (1, ()), (1, ('b', 'd', 'c')), (1, ('r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7')), (1,
                 ('far',))),
    'P3/T1/S2': (1, 1, (1, ()), (1, ()), (1, ('b', 'd', 'c')), (1, ('r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7')), (1,
                 ('far',))),
    'P3/T2/S1': (1, 1, (1, ()), (1, ()), (1, ('b', 'd', 'c')), (1, ('r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7')), (1,
                 ('far',))),
    'P3/T2/S2': (1, 1, (1, ()), (1, ()), (1, ('b', 'd', 'c')), (1, ('r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7')), (1,
                 ('far',))),
    'P3/T3/S1': (1, 1, (1, ()), (1, ()), (1, ('b', 'd', 'c')), (1, ('r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7')), (1,
                 ('far',))),
    'P3/T3/S2': (1, 1, (1, ()), (1, ()), (1, ('b', 'd', 'c')), (1, ('r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7')), (1,
                 ('far',))),
    'P4/T1/S1': (1, 1, (1, ()), (1, ()), (1, ('b', 'd', 'c')), (1, ('r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7')), (1,
                 ('far',))),
    'P4/T1/S2': (1, 1, (1, ()), (1, ()), (1, ('b', 'd', 'c')), (1, ('r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7')), (1,
                 ('far',))),
    'P4/T2/S1': (1, 1, (1, ()), (1, ()), (1, ('b', 'd', 'c')), (1, ('r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7')), (1,
                 ('far',))),
    'P4/T2/S2': (1, 1, (1, ()), (1, ()), (1, ('b', 'd', 'c')), (1, ('r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7')), (1,
                 ('far',))),
    'P4/T3/S1': (1, 1, (1, ()), (1, ()), (1, ('b', 'd', 'c')), (1, ('r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7')), (1,
                 ('far',))),
    'P4/T3/S2': (1, 1, (1, ()), (1, ()), (1, ('b', 'd', 'c')), (1, ('r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7')), (1,
                 ('far',))),
    'P5/T1/S1': (8, 8, (8, ()), (8, ()), (8, ()), (8, ()), (8, ())),
    'P5/T1/S2': (4, 4, (4, ()), (4, ()), (4, ()), (4, ('r4', 'r5', 'r6', 'r7')), (4, ())),
    'P5/T2/S1': (8, 8, (8, ()), (8, ()), (8, ()), (8, ()), (8, ())),
    'P5/T2/S2': (4, 4, (4, ()), (4, ()), (4, ()), (4, ('r4', 'r5', 'r6', 'r7')), (4, ())),
    'P5/T3/S1': (1, 1, (1, ()), (1, ()), (1, ('b', 'd', 'c')), (1, ('r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7')), (1,
                 ('far',))),
    'P5/T3/S2': (1, 1, (1, ()), (1, ()), (1, ('b', 'd', 'c')), (1, ('r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7')), (1,
                 ('far',))),
}
# A policy whose robots wait for every response: 20 Hz, 10 valid steps, the next requested at the 10th (nothing left
# when they send their requests), and a batch's p95 by its size, about a second.
WAITS_P95 = {1: 1050.0, 2: 1150.0, 4: 1250.0, 8: 1550.0}


def _spread(lat: dict) -> slack.Table:
    table = slack.Table()
    for b, (p50, p95) in lat.items():
        for ms in (p50 - 5.0, p50, p50, p50 + 5.0, p95):
            table.add(b, ms)
    return table


def _seating(table: slack.Table, play: dict, sizes: list) -> slack.Seating:
    """A replica's Seating from a policy's playback, with no more robots than its largest listed size."""
    wait = play.get("wait_s")
    return slack.Seating(table, play["buffer_s"] * 1000.0, sizes=sizes,
                         cycle_ms=(play["play_s"] - play["buffer_s"]) * 1000.0, most=sizes[-1],
                         **({"wait_ms": wait * 1000.0} if wait is not None else {}))


def test_with_no_wait_set_a_replica_seats_as_pinned_point_by_point():
    for key, want in PINNED.items():
        p, t, s = key.split("/")
        hz, valid, replan = POLICIES[p]
        play, table, sizes = slack.playback(float(hz), valid, replan), _spread(TABLES[t]), SIZES[s]
        assert "wait_s" not in play
        buffer_ms, cycle_ms = play["buffer_s"] * 1000.0, (play["play_s"] - play["buffer_s"]) * 1000.0
        got = (slack.per_replica(table, buffer_ms, sizes=sizes, cycle_ms=cycle_ms),
               slack.per_replica(table, buffer_ms, sizes=sizes, cycle_ms=cycle_ms, outside_ms=47.0),
               *((n, tuple(over)) for n, over in (_seating(table, play, sizes).seat(r) for r in ROBOTS.values())))
        assert got == want, key


@pytest.mark.parametrize("wait, seats", [(None, 1), (1.2, 1), (1.3, 2), (1.4, 4), (1.7, 8)])
def test_a_policy_whose_robots_wait_for_every_response_seats_by_the_wait_it_sets(wait, seats):
    """Not set, one robot per replica; set, the capacity is the size of a batch whose p95, a 47 ms RTT and the 50 ms
    margin fit within the wait, and whose load fills no more than LOAD_MAX of a robot's own time between requests."""
    table = _timed(WAITS_P95, runs=5)
    play = slack.playback(20.0, 10, 10, wait)
    seating = _seating(table, play, [1, 2, 4, 8])
    near = [(f"r{i}", 47.0) for i in range(8)]
    assert seating.seat(near) == (seats, [name for name, _ in near[seats:]])
    assert seating.seat([])[0] == seats                    # an empty replica, its robots at ROUND_TRIP_MS
    w = play.get("wait_s")
    assert slack.per_replica(table, 0.0, sizes=[1, 2, 4, 8], cycle_ms=500.0, outside_ms=47.0,
                          **({"wait_ms": w * 1000.0} if w else {})) == seats


def test_a_wait_shorter_than_what_is_left_of_a_chunk_seats_by_the_wait_and_fewer():
    """Set below the playback time left when its robot sends a request, it is still the wait that a replica's capacity
    is computed from: the policy's own number."""
    table = _spread(TABLES["T2"])
    unset, short = (slack.playback(15.0, 15, 8, wait) for wait in (None, 0.3))
    assert unset["buffer_s"] * 1000.0 == pytest.approx(466.7, abs=0.1) and short["wait_s"] == 0.3
    assert _seating(table, unset, [1, 2, 4, 8]).seat([])[0] == 2
    assert _seating(table, short, [1, 2, 4, 8]).seat([])[0] == 1


def test_wait_fill_is_each_sizes_share_of_its_own_time_between_a_robots_requests():
    table = _timed(WAITS_P95, runs=5)
    # two robots, nothing left when they send their requests, an RTT of 47 ms: batches of one fill 2 * 1050 of
    # 500 + 1050 + 47, which is too much; one batch of two fills 1150 of 500 + 1150 + 47
    assert slack.wait_fill(2, table, timed=[1, 2, 4, 8], cycle_ms=500.0, slack_ms=0.0, trip_ms=47.0) == (
        2, pytest.approx(1150.0 / 1697.0))
    assert slack.wait_fill(2, table, timed=[1], cycle_ms=500.0, slack_ms=0.0, trip_ms=47.0) is None


def _held_waiting(wait_s) -> float:
    """How long robot a's request may wait for robot b, which is admitted and whose next request is due in a step: a's
    last response left 0.6 s ago and its actions ran out 0.1 s ago (they last 0.5 s), with its robot reporting no
    RTT."""
    now = 10.0
    al = slack.Aligner(_timed(FORWARD), slack.playback(20.0, 10, 10, wait_s), seated=lambda: ["a", "b"])
    al.arrived("b", now - 0.4)
    al.arrived("a", now - 1.2)
    al.answered(["a"], now - 0.6)
    al.arrived("a", now)
    return al.hold_s(["a"], now)


def test_where_its_policy_sets_a_wait_a_request_is_held_for_a_batch_by_it_from_when_it_was_sent():
    """Not set, a robot that waits for every response has no slack, and its request is dispatched at once; set, it may
    wait that long from when it was sent (its actions run out, and its robot waits), less its RTT, the batch with one
    more request in it and the margin."""
    assert _held_waiting(None) == 0.0
    assert _held_waiting(1.7) == pytest.approx(1.7 - TRIP - FORWARD[2] / 1000 - MARGIN)
