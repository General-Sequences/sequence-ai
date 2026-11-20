# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
A fleet in front of serve(): four robots at different distances from one server that holds and seats them by their
chunks' slack, and what share of their answers came in time, beside what slack's clock says for the same robots.

    python examples/fleet_on_serve.py                  # 30 s: robots 20, 20, 100 and 200 ms away
    python examples/fleet_on_serve.py --start 3        # the seated robots starting another way

No GPU and no network: the policy is a stand-in that sleeps the time a batch of its size takes - 80, 110 and 130 ms
for 1, 2 and 4, the batch times engine/slack.py's docstring takes - and each robot reaches the server through a proxy on
this machine that holds each direction for half its round trip. The policy runs at 20 Hz, 15 valid steps, the next
asked for at step 8: a 350 ms buffer, an act every 400 ms.

In one sentence: serve(hz=20, action_chunk_valid_len=15, action_chunk_replan_at=8) times each batch size in its warm-up
→ the robots connect nearest first, each through its own proxy, and each sends an act every k / hz from its own start,
or as soon as an answer is back that took longer → the server seats each at its first act if it and the robots seated
are all answered in time, and holds each act for a batch only as long as its robot's chunk leaves it time → an answer is
late when it is back after the chunk its robot plays ran out (N / hz from when the act before it was sent) → the share
in time is counted as slack.on_a_clock counts it, from its CLOCK_WARM-th cycle on, and the clock is run for the same
robots, starts and cycles, with the batch times the server measured (its /ping's batch_ms).

The robot 200 ms away connects last and is refused: its batch, run beside the near robots', leaves its answer too little
of its chunk. Its act is answered 503 with why, which its policy's act() raises (Unavailable). On the clock, the four
together have answers late in some of its starts, where a fleet is seated only with none late in any start.
"""
from __future__ import annotations

import argparse
import asyncio
import socket
import threading
import time

import httpx

import sequence_ai
from sequence_ai.engine import slack
from sequence_ai.errors import Unavailable

TIMES_MS = {1: 80.0, 2: 110.0, 4: 130.0}
HZ, VALID, REPLAN = 20.0, 15, 8
ROBOTS = [("near-a", 20.0), ("near-b", 20.0), ("mid", 100.0), ("far", 200.0)]      # (its name, its round trip in ms)
OBSERVATION = {"state": [0.0] * 7}


def stand_in(times: dict = TIMES_MS, valid: int = VALID):
    """A policy of batches: one of the listed size it is given takes its time in `times`, and answers `valid` steps."""
    def policy(observations: list) -> list:
        time.sleep(times[len(observations)] / 1000.0)
        return [{"actions": [[0.0] * 7] * valid} for _ in observations]
    return policy


class Delay:
    """A TCP proxy on this machine to the server at `target`, which holds each direction half of `rtt_ms`: a robot that
    far from the server. `port` is where a robot connects to it."""

    def __init__(self, target: int, rtt_ms: float) -> None:
        self._half, self._target = rtt_ms / 2000.0, target
        self._loop = asyncio.new_event_loop()
        self._up = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()
        self._up.wait()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        server = self._loop.run_until_complete(asyncio.start_server(self._carry, "127.0.0.1", 0))
        self.port = server.sockets[0].getsockname()[1]
        self._up.set()
        self._loop.run_forever()

    async def _carry(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        to_reader, to_writer = await asyncio.open_connection("127.0.0.1", self._target)
        await asyncio.gather(self._pipe(reader, to_writer), self._pipe(to_reader, writer))

    async def _pipe(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Each chunk read, written half the round trip later, in order."""
        loop, queue = asyncio.get_running_loop(), asyncio.Queue()

        async def write() -> None:
            while True:
                due, data = await queue.get()
                if due > loop.time():
                    await asyncio.sleep(due - loop.time())
                if not data:
                    writer.close()
                    return
                writer.write(data)
                await writer.drain()

        writing = asyncio.ensure_future(write())
        try:
            while True:
                data = await reader.read(1 << 16)
                queue.put_nowait((loop.time() + self._half, data))
                if not data:
                    break
        except ConnectionError:
            queue.put_nowait((loop.time(), b""))
        try:
            await writing
        except ConnectionError:
            pass


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def drive(policy, begin: float, until: float, counted_from: float, said: dict) -> None:
    """One robot: an act every k / hz from `begin`, or as soon as its answer is back when that took longer; each answer
    late when back after the chunk before it ran out (N / hz from when that act was sent), counted for acts sent from
    `counted_from` on."""
    cycle_s, play_s = REPLAN / HZ, VALID / HZ
    send_at, runs_out = begin, None
    while send_at < until:
        time.sleep(max(0.0, send_at - time.perf_counter()))
        sent = time.perf_counter()
        try:
            policy.observe(OBSERVATION)
            policy.act()
        except Unavailable as refused:
            said["refused"] = str(refused)
            return
        back = time.perf_counter()
        if runs_out is not None and sent >= counted_from:
            said["answers"] += 1
            said["late"] += back > runs_out
        runs_out, send_at = sent + play_s, max(sent + cycle_s, back)


def run(seconds: float = 30.0, start: int = 7, robots: list = ROBOTS, times: dict = TIMES_MS) -> dict:
    """The fleet for `seconds`, the robots seated first starting at slack.starts' start number `start`: each robot's
    own count, what the server's /ping says at the end, and the clock's count for the robots seated. `robots`: (its
    name, its round trip in ms), the last connecting after the others; `times`: the stand-in's batch times, in ms."""
    port = _free_port()
    threading.Thread(target=sequence_ai.serve, args=(stand_in(times),), daemon=True,
                     kwargs=dict(host="127.0.0.1", port=port, batch=sorted(times), example=OBSERVATION, hz=HZ,
                                 action_chunk_valid_len=VALID, action_chunk_replan_at=REPLAN,
                                 log_level="warning")).start()
    links = {}
    for name, rtt in robots:                                       # nearest first: each seated before the next
        links[name] = sequence_ai.connect(f"http://127.0.0.1:{Delay(port, rtt).port}")
        links[name].wait_until_ready(60.0)
    cycle_s = REPLAN / HZ
    last = robots[-1][0]
    first = [name for name, _ in robots if name != last]
    begins = dict(zip(first, slack.starts(len(first), cycle_s, start + 1)[start]))
    begins[last] = cycle_s                                         # after the others' first acts
    t0 = time.perf_counter() + 0.5
    until, counted_from = t0 + seconds, t0 + slack.CLOCK_WARM * cycle_s
    said = {name: {"rtt_ms": rtt, "answers": 0, "late": 0, "refused": None} for name, rtt in robots}
    threads = [threading.Thread(target=drive, args=(links[name], t0 + begins[name], until, counted_from, said[name]))
               for name, _ in robots]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    server = httpx.get(f"http://127.0.0.1:{port}/ping").json()
    for link in links.values():
        link.close()
    seated = [name for name, _ in robots if said[name]["refused"] is None]
    measured = slack.Table()                                       # what a batch took in the server, as it says
    for size, ms in server["batch_ms"].items():
        for _ in range(slack.RUNS):
            measured.add(int(size), ms)
    play, sizes = slack.playback(hz=HZ, valid_len=VALID, replan_at=REPLAN), sorted(times)
    rtts = [said[name]["rtt_ms"] for name in seated]
    late, answers, _ = slack.on_a_clock(measured, play, rtts, [begins[name] for name in seated], sizes=sizes,
                                        cycles=int(seconds / cycle_s))
    starts = [slack.on_a_clock(measured, play, rtts, begin, sizes=sizes, early=True)
              for begin in slack.starts(len(rtts), cycle_s)]
    # had every robot been seated: the clock for all of them, over its starts, as slack.in_time counts it
    every = [slack.on_a_clock(measured, play, [rtt for _, rtt in robots], begin, sizes=sizes, early=True)
             for begin in slack.starts(len(robots), cycle_s)]
    shares = [lt / n for lt, n, _ in every if n]
    return {"robots": said, "server": server,
            "clock": {"late": late, "answers": answers,
                      "late_share_over_starts": sum(lt / n for lt, n, _ in starts if n) / len(starts),
                      "every_robot": {"late_share_over_starts": sum(shares) / len(every), "worst": max(shares)}}}


def _in_time(late: int, answers: int) -> float:
    return 100.0 * (1 - late / answers) if answers else 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--start", type=int, default=7, help="which of slack.starts' starts the seated robots take")
    args = parser.parse_args()
    out = run(args.seconds, args.start)
    server, clock = out["server"], out["clock"]
    print(f"serve() at {HZ:g} Hz, {VALID} valid steps, the next asked for at step {REPLAN}; a batch of "
          f"{', '.join(f'{n}: {ms:g}' for n, ms in TIMES_MS.items())} ms in the stand-in, "
          f"{', '.join(f'{n}: {ms:g}' for n, ms in server['batch_ms'].items())} ms in the server. It seats "
          f"{server['seats']} robots 100 ms away; {server['robots']} were seated at the end.")
    live_late = live_answers = 0
    for name, robot in out["robots"].items():
        if robot["refused"]:
            print(f"  {name:7} {robot['rtt_ms']:4.0f} ms  refused: {robot['refused']}")
            continue
        live_late, live_answers = live_late + robot["late"], live_answers + robot["answers"]
        print(f"  {name:7} {robot['rtt_ms']:4.0f} ms  {robot['answers']} answers, {robot['late']} late: "
              f"{_in_time(robot['late'], robot['answers']):.1f}% in time")
    print(f"In time: {_in_time(live_late, live_answers):.1f}% of {live_answers} answers here; "
          f"{_in_time(clock['late'], clock['answers']):.1f}% of {clock['answers']} on slack's clock for the same "
          f"robots, starts and batch times, {100.0 * (1 - clock['late_share_over_starts']):.1f}% over its starts.")
    every = clock["every_robot"]
    print(f"Had every robot been seated: {100.0 * (1 - every['late_share_over_starts']):.1f}% in time over the clock's "
          f"starts, {100.0 * every['worst']:.0f}% late at the worst, where a fleet is seated only with none late in "
          f"any start.")
