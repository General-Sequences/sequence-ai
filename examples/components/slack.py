# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
slack, alone: scheduling by the slack an action chunk leaves, for one replica of a model: how many robots the replica
can serve, each by its own RTT, and which requests wait for a batch and which are dispatched ahead of it.

    python examples/components/slack.py

No model and no GPU: a forward pass here only advances a clock, so every run prints the same output. It shows the
rules on one pattern of arrivals, chosen so that the three ways to batch them differ; it measures nothing.
"""
import heapq
import itertools

from sequence_ai.engine import slack

FORWARD_S = {1: 0.140, 2: 0.160, 4: 0.190}                    # one forward pass of 1, 2 and 4 requests
SIZES = sorted(FORWARD_S)
RTT_MS = {"near-1": 10.0, "near-2": 10.0, "far": 60.0}        # ms each robot's requests spend outside the server
ARRIVES_S = {"near-1": 0.0, "far": 0.005, "near-2": 0.040}    # when each robot's request arrives, every round
clock = [0.0]


def forward_of(n: int) -> None:
    clock[0] += FORWARD_S[n]                                   # your model's call on a batch of n here


table = slack.measure(forward_of, sizes=SIZES, clock=lambda: clock[0])       # with a real model: no `clock`
play = slack.playback(hz=20, valid_len=14, replan_at=8)       # 14 valid steps per chunk, the next requested at the 8th
buffer_ms, cycle_ms = play["buffer_s"] * 1000, (play["play_s"] - play["buffer_s"]) * 1000
seating = slack.Seating(table, buffer_ms, sizes=SIZES, cycle_ms=cycle_ms)


def one_replica(aligner: "slack.Aligner | None" = None, wait_s: float = 0.0, rounds: int = 5) -> list:
    """One replica that runs one batch at a time, each robot sending a request every cycle. With an aligner the open
    batch waits and splits as the aligner decides; with none it is dispatched as batcher.Batcher does it: `wait_s` after
    its first request, or as soon as the replica is free. Returns the last round's batches: (sent at, its robots, out
    at, each one's ms to spare)."""
    order = itertools.count()
    events = [(at + r * cycle_ms / 1000, next(order), "request", name) for r in range(rounds)
              for name, at in ARRIVES_S.items()]
    heapq.heapify(events)
    waiting, came, batches, busy, timer = [], {}, [], False, None

    def arm(at: float) -> None:
        nonlocal timer
        timer = at
        heapq.heappush(events, (at, next(order), "timer", at))

    def send(now: float) -> None:
        nonlocal waiting, busy, timer
        first = aligner.split(waiting, now, SIZES, free=False) if aligner else None      # which are dispatched first
        batch = [waiting[i] for i in first] if first else waiting[:max(n for n in SIZES if n <= len(waiting))]
        waiting = [name for name in waiting if name not in batch]
        busy, timer = True, None
        heapq.heappush(events, (now + FORWARD_S[len(batch)], next(order), "done", (now, batch)))

    def go_or_wait(now: float) -> None:
        hold = aligner.hold_s(waiting, now) if aligner else 0.0                           # how long it may wait
        if hold <= 0:
            send(now)
        else:
            arm(now + hold)

    while events:
        now, _, kind, what = heapq.heappop(events)
        if kind == "request":
            if aligner:
                aligner.arrived(what, now, outside_ms=RTT_MS[what])                       # each request as it arrives
            came[what] = now
            waiting.append(what)
            if busy:
                continue                                       # it accumulates until the replica is free
            if aligner:
                go_or_wait(now)
            elif timer is None:
                arm(now + wait_s)
        elif kind == "timer" and timer == what and not busy:
            send(now)
        elif kind == "done":
            sent_at, batch = what
            if aligner:
                aligner.answered(batch, now)                                              # its responses leave now
                table.add(len(batch), FORWARD_S[len(batch)] * 1000)                       # and its forward time is kept
            spare = {name: (came[name] + (buffer_ms - RTT_MS[name]) / 1000 - now) * 1000 for name in batch}
            batches.append((sent_at, batch, now, spare))
            busy = False
            if waiting:
                go_or_wait(now)
    start = (rounds - 1) * cycle_ms / 1000
    return [(sent_at - start, batch, out - start, spare) for sent_at, batch, out, spare in batches if sent_at >= start]


def show(title: str, batches: list) -> None:
    print(f"  {title}")
    for sent_at, batch, out, spare in batches:
        margins = ", ".join(f"{name} {ms:.0f} ms to spare" if ms >= 0 else f"{name} {-ms:.0f} ms LATE"
                            for name, ms in spare.items())
        print(f"    {sent_at * 1000:4.0f} ms  {' + '.join(batch):<16} out at {out * 1000:3.0f} ms   {margins}")


if __name__ == "__main__":
    print("A policy at 20 Hz, 14 valid steps per chunk, the next requested at the 8th:")
    print(f"A buffer of {buffer_ms:.0f} ms when a robot sends its request, a request every {cycle_ms:.0f} ms. "
          f"Batches of {SIZES} take {[round(table.p95(n)) for n in SIZES]} ms.\n")
    for name, rtt in RTT_MS.items():
        print(f"  {name:<7} {rtt:3.0f} ms away   in time in a batch of up to {seating.fit(rtt)}")
    seats, over = seating.seat(list(RTT_MS.items()))
    uniform = slack.per_replica(table, buffer_ms, sizes=SIZES, outside_ms=max(RTT_MS.values()), cycle_ms=cycle_ms)
    print(f"  the replica can serve {seats} robots, each by its own RTT; over, the one to move: {', '.join(over)}")
    print(f"  (with every robot evaluated by the furthest robot's RTT, it could serve {uniform})\n")

    print(f"One round of arrivals ({', '.join(f'{n} at {at * 1000:.0f} ms' for n, at in ARRIVES_S.items())}), "
          "batched three ways.")
    print("To spare: how much earlier a response left than required to reach its robot before its steps ran out.")
    show("as Batcher dispatches them with wait_ms=0: as soon as the replica is free", one_replica())
    show("as Batcher dispatches them with wait_ms=100: after a wait for a batch", one_replica(wait_s=0.100))
    aligner = slack.Aligner(table, play, seated=lambda: list(RTT_MS))
    show("as slack.Aligner decides: each robot by its own slack", one_replica(aligner))
