# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
network.lines: one policy server over two or more of a robot's uplinks, against serve() on real ports - every act sent
on each line and run once, the first answer used, a slow line passed over while it carries the act before, a dead line
leaving the act to the other, one line at a time to a server that does not say "once", lines to two servers refused,
connect(url, lines=...) and its errors, and the request id's header the platform's.

    python -m pytest -q tests/network/test_lines.py
"""
from __future__ import annotations

import contextlib
import re
import socket
import sys
import threading
import time

import numpy as np
import pytest

import sequence_ai
from sequence_ai._core import rid
from sequence_ai.engine import serve as serving
from sequence_ai.engine.serve import App
from sequence_ai.errors import InvalidRequest
from sequence_ai.network import connect as network_connect
from sequence_ai.network.lines import Lines
from sequence_ai.network.link import ONCE, REQUEST_HEADER, Link


class _Counted:
    """An ASGI app's POST /act requests counted, with their request ids."""

    def __init__(self, app):
        self.app, self.acts, self.ids = app, 0, []

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"] == "/act":
            self.acts += 1
            self.ids.append(dict(scope["headers"]).get(REQUEST_HEADER.lower().encode()))
        await self.app(scope, receive, send)


@contextlib.contextmanager
def _serving(app):
    """`app` on uvicorn as serve() runs it, on 127.0.0.1 and a free port, in a thread; its URL."""
    server = serving._uvicorn(app, "warning")
    sockets = serving._listen("127.0.0.1", 0)
    thread = threading.Thread(target=server.run, kwargs={"sockets": sockets}, daemon=True)
    thread.start()
    try:
        for _ in range(500):
            if server.started:
                break
            time.sleep(0.01)
        yield serving._url(sockets[0])
    finally:
        server.should_exit = True
        thread.join(10)


@contextlib.contextmanager
def _relay(url: str, delay_s: float):
    """A TCP relay to `url`'s port that holds every byte `delay_s` in each direction: a slower uplink. Its URL."""
    target = int(url.rsplit(":", 1)[1])
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    done = threading.Event()

    def pump(a, b):
        try:
            while True:
                data = a.recv(65536)
                if not data:
                    break
                time.sleep(delay_s)
                b.sendall(data)
        except OSError:
            pass
        finally:
            with contextlib.suppress(OSError):
                b.shutdown(socket.SHUT_WR)

    def accept():
        while not done.is_set():
            try:
                client, _ = listener.accept()
            except OSError:
                return
            upstream = socket.create_connection(("127.0.0.1", target))
            threading.Thread(target=pump, args=(client, upstream), daemon=True).start()
            threading.Thread(target=pump, args=(upstream, client), daemon=True).start()

    threading.Thread(target=accept, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{listener.getsockname()[1]}"
    finally:
        done.set()
        listener.close()


def _policy(calls: list):
    def policy(observation):
        calls.append(float(observation["state"][0]))
        return {"actions": np.full((8, 7), observation["state"][0], dtype=np.float32)}
    return policy


def _lines(*urls) -> Lines:
    return Lines([Link(url, check=False, ping_every_s=60.0) for url in urls])


def test_the_request_id_header_is_the_platforms_and_serve_says_once():
    assert REQUEST_HEADER == rid.HEADER and ONCE == "once"


def test_every_act_goes_on_both_lines_runs_once_and_comes_back_with_one_request_id():
    calls: list = []
    app = _Counted(App(_policy(calls)))
    with _serving(app) as url, _lines(url, url) as robot:
        robot.wait_until_ready(timeout_s=10)
        assert robot.once
        got = [float(robot.act({"state": np.full(7, float(n), dtype=np.float32)})["actions"][0, 0]) for n in range(5)]
        time.sleep(0.2)                                   # the copies not used, read as they come
        stats = robot.stats()
    assert got == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert calls == got                                   # the policy ran once an act
    assert 5 < app.acts <= 10                             # its copies came on the other line
    assert sum(line["wins"] for line in stats["lines"]) == stats["counters"]["acts"] == 5
    sent = [i for i in app.ids if i is not None]
    assert len(sent) == app.acts and len(set(sent)) == 5 and all(i.startswith(b"r-") for i in sent)


def test_a_slower_line_answers_none_first_and_is_passed_over_while_it_carries_the_act_before():
    calls: list = []
    with _serving(App(_policy(calls))) as url, _relay(url, 0.15) as slow:
        with _lines(url, slow) as robot:
            for line in robot.links:                      # both ready: robot.wait_until_ready() returns on the first
                line.wait_until_ready(timeout_s=15)
            for n in range(4):
                robot.act({"state": np.full(7, float(n), dtype=np.float32)})
                if n < 2:
                    time.sleep(0.6)                       # the slow line's copy back before the next act; not after
            time.sleep(1.0)                               # the third, so the fourth finds it carrying the third
            stats = robot.stats()
    fast, slow_line = stats["lines"]
    assert (fast["wins"], slow_line["wins"]) == (4, 0) and len(calls) == 4
    assert slow_line["passed_over"] >= 1                 # it was still carrying the act before
    behind = stats["latency_us"]["behind_us"]
    assert behind["n"] >= 3 and behind["max"] > 200_000   # its answers came at least its round trip later


def test_a_dead_line_leaves_each_act_to_the_other():
    calls: list = []
    with _serving(App(_policy(calls))) as url, _lines("http://127.0.0.1:9", url) as robot:
        robot.wait_until_ready(timeout_s=10)
        for n in range(3):
            assert float(robot.act({"state": np.full(7, float(n), dtype=np.float32)})["actions"][0, 0]) == float(n)
        stats = robot.stats()
    dead, live = stats["lines"]
    assert live["wins"] == 3 and dead["wins"] == 0 and len(calls) == 3


def test_to_a_server_that_does_not_say_once_each_act_goes_on_one_line():
    calls: list = []
    old = App(_policy(calls))
    old._reads = b"jpeg, zstd, zlib"                      # a server from before ONCE: it runs each copy it gets
    app = _Counted(old)
    with _serving(app) as url, _lines(url, url) as robot:
        robot.wait_until_ready(timeout_s=10)
        assert not robot.once
        for n in range(4):
            robot.act({"state": np.full(7, float(n), dtype=np.float32)})
        stats = robot.stats()
    assert app.acts == 4 and len(calls) == 4 and stats["counters"]["acts_on_one_line"] == 4
    assert app.ids == [None] * 4                          # no request id where it would not be read


def test_lines_to_two_servers_are_refused_before_an_act_is_sent_to_each():
    calls: list = []
    with _serving(App(_policy(calls))) as a, _serving(App(_policy(calls))) as b, _lines(a, b) as robot:
        for line in robot.links:                          # each line's ping answered before its server is renamed
            line.wait_until_ready(timeout_s=10)
        for line, pid in zip(robot.links, (101, 202)):    # each ping names another process
            line.server = {**(line.server or {}), "pid": pid, "up_s": 5.0}
        with pytest.raises(ValueError, match="not one server"):
            robot.act({"state": np.zeros(7, dtype=np.float32)})
    assert calls == []


def test_connect_takes_two_interfaces_or_more_by_names_this_machine_has():
    with pytest.raises(ValueError, match="two network interfaces or more"):
        network_connect("http://127.0.0.1:9", lines=["lo0"])
    with pytest.raises(OSError, match="no network interface 'nope0'"):
        network_connect("http://127.0.0.1:9", lines=["nope0", "nope1"], check=False)


def test_sequence_ai_connect_takes_lines_for_an_address_and_refuses_them_for_a_name():
    with pytest.raises(InvalidRequest, match="lines= sends each act over each of the robot's uplinks"):
        sequence_ai.connect("my_policy_user", lines=["eth0", "wwan0"])


@pytest.mark.skipif(sys.platform != "darwin", reason="a Mac's lo0 and en0")
def test_on_a_mac_lines_by_the_loopback_and_the_wifi_reach_a_local_server_by_the_loopback_alone():
    """A line bound to en0 cannot reach 127.0.0.1, so the loopback's answers every act: the binding is each line's."""
    if "en0" not in {name for _, name in socket.if_nameindex()}:
        pytest.skip("no en0 here")
    calls: list = []
    with _serving(App(_policy(calls))) as url:
        robot = sequence_ai.connect(url, lines=["lo0", "en0"])
        try:
            robot.wait_until_ready(timeout_s=10)
            for n in range(2):
                robot.act({"state": np.full(7, float(n), dtype=np.float32)})
            stats = robot.stats()
        finally:
            robot.close()
    lo, en = stats["lines"]
    assert (lo["bind"], en["bind"]) == ("lo0", "en0") and lo["wins"] == 2 and en["wins"] == 0 and len(calls) == 2


def test_check_with_lines_says_each_lines_ping_that_copies_run_once_and_which_line_answered_first():
    from sequence_ai import check

    loopback = next(name for _, name in socket.if_nameindex() if name.startswith("lo"))
    calls: list = []
    with _serving(App(_policy(calls), example={"state": np.zeros(7, dtype=np.float32)})) as url:
        rows, _ = check.run(url, pings=3, lines=[loopback, loopback])
        refused, status = check.run(url, pings=2, lines=["nope0", "nope1"])
    by = {row.name: row for row in rows}
    assert by[f"line {loopback}"].status == "ok" and f"for a ping by {loopback} alone" in by[f"line {loopback}"].detail
    assert by["lines"].status == "ok" and '"once" in X-Seq-Reads' in by["lines"].detail
    answers = by["lines' answers"]
    assert answers.status == "ok" and "of 5 acts sent on every line, first by" in answers.detail
    assert re.search(r"each act sends [\d,]+ bytes on each line - at one act a second, [\d,]+ MB an hour", answers.detail)
    [no] = [row for row in refused if row.name == "lines"]
    assert no.status == "no" and "no network interface 'nope0'" in no.detail and status == 1
