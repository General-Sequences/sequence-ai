# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
A fleet on a clock: what slack runs to compute a replica's capacity, printed for each number of robots. Several robots
of one policy and one replica, the batches held and dispatched as slack decides, and what share of the responses are
back before their robot's buffer runs out.

    python examples/fleet_on_a_clock.py                                              # a model with slow batches
    python examples/fleet_on_a_clock.py --times 1:198,2:320,4:565 --rtt 50           # yours: ms for each batch size
    python examples/fleet_on_a_clock.py --margin 50                                  # late: under 50 ms to spare

No model and no GPU: a forward pass only advances a clock by the time given for its batch size, so every run prints the
same output. Use it to see why slack.per_replica computes the capacity it does for your model's batch times, and what
one robot more would get.

In one sentence: slack.starts gives the starts a number of robots is run from (spread evenly over every way their
requests can fall in a cycle: 64 for two robots, 128 for more) → slack.on_a_clock runs each until it repeats, 48 cycles
at most: every robot sends a request each k / hz, or as soon as a response is back that took longer, and the server
loop of docs/slack.md batches them → the capacity is the last number of robots, counted up from one, with at least
slack.IN_TIME_MIN of its responses in time over the starts and a load of no more than slack.LOAD_MAX in each of them.

WHY several starts: robots send on their own clocks, and whether two of them share a batch depends on how far apart
their requests are. Their clocks drift, so over a long run a fleet meets every way its requests can fall, and the share
over the starts is the share of that run.
"""
import argparse

from sequence_ai.engine import slack

SLOW_MS = {1: 256.0, 2: 273.0, 4: 303.0, 8: 377.0}     # a model whose batches take more than half of a 467 ms buffer


def fleet(times_ms: dict, robots: int, *, hz: float = 15.0, valid: int = 15, replan: int = 8, rtt_ms: float = 20.0,
          margin_ms: float = 0.0) -> list:
    """`robots` robots on one replica, from each start: [(late responses, responses, load)] of the stretch of the run
    that repeats, which is what slack.in_time counts."""
    clock, sizes = [0.0], sorted(times_ms)

    def forward_of(n: int) -> None:
        clock[0] += times_ms[n] / 1000.0                       # your model's call on a batch of n here

    table = slack.measure(forward_of, sizes=sizes, clock=lambda: clock[0])       # with a real model: no `clock`
    play = slack.playback(hz=hz, valid_len=valid, replan_at=replan)
    cycle_s = play["play_s"] - play["buffer_s"]
    return [slack.on_a_clock(table, play, [rtt_ms] * robots, begin, sizes=sizes, early=True, spare_ms=margin_ms)
            for begin in slack.starts(robots, cycle_s)]


def capacity(times_ms: dict, *, hz: float = 15.0, valid: int = 15, replan: int = 8, rtt_ms: float = 20.0,
             cycle: bool = True) -> int:
    """What slack.per_replica computes for these times: on a clock, or by the buffer alone (`cycle` False)."""
    table = slack.Table()
    for size, ms in times_ms.items():
        for _ in range(slack.RUNS):
            table.add(size, ms)
    play = slack.playback(hz=hz, valid_len=valid, replan_at=replan)
    cycle_ms = (play["play_s"] - play["buffer_s"]) * 1000 if cycle else None
    return slack.per_replica(table, play["buffer_s"] * 1000, sizes=sorted(times_ms), outside_ms=rtt_ms,
                             cycle_ms=cycle_ms, step_ms=play["step_s"] * 1000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--times", help="ms a batch of each listed size takes, as 1:256,2:273,4:303,8:377")
    parser.add_argument("--robots", type=int, help="one number of robots; without it, 1 to the capacity by the buffer")
    parser.add_argument("--hz", type=float, default=15.0)
    parser.add_argument("--valid", type=int, default=15, help="valid steps in a chunk (N)")
    parser.add_argument("--replan", type=int, default=8, help="the step at which the next chunk is requested (k)")
    parser.add_argument("--rtt", type=float, default=20.0, help="each robot's RTT, in ms")
    parser.add_argument("--margin", type=float, default=0.0,
                        help="ms a response must be back before the buffer runs out to count as in time; the capacity "
                             "is counted with 0")
    args = parser.parse_args()
    times = SLOW_MS
    if args.times:
        times = {int(n): float(ms) for n, ms in (item.split(":") for item in args.times.split(","))}
    policy = {"hz": args.hz, "valid": args.valid, "replan": args.replan, "rtt_ms": args.rtt}
    playback = slack.playback(hz=args.hz, valid_len=args.valid, replan_at=args.replan)
    by_buffer = capacity(times, cycle=False, **policy)
    print(f"Batches of {sorted(times)} take {[round(times[n]) for n in sorted(times)]} ms. A policy at {args.hz:g} Hz, "
          f"{args.valid} valid steps per chunk, the next requested at step {args.replan}:")
    print(f"a buffer of {playback['buffer_s'] * 1000:.0f} ms, a request every "
          f"{(playback['play_s'] - playback['buffer_s']) * 1000:.0f} ms. Robots {args.rtt:g} ms away.\n")
    print("  robots   responses in time   starts with a late response   load")
    for count in ([args.robots] if args.robots else range(1, min(by_buffer, 8) + 1)):
        runs = fleet(times, count, margin_ms=args.margin, **policy)
        in_time = 1.0 - sum(late / max(total, 1) for late, total, _ in runs) / len(runs)
        low, high = f"{min(run[2] for run in runs):.2f}", f"{max(run[2] for run in runs):.2f}"
        load = high if low == high else f"{low} to {high}"
        print(f"  {count:>6}   {in_time:>16.1%}    {sum(run[0] > 0 for run in runs):>13} of {len(runs):<3}           "
              f"{load}")
    print(f"\nslack.per_replica computes a capacity of {capacity(times, **policy)} for these times "
          f"({by_buffer} by the buffer alone): the first number of robots\nwith fewer than {slack.IN_TIME_MIN:.0%} of "
          f"its responses in time over the starts, or a load above {slack.LOAD_MAX:g} in one of them, ends the count.")
