# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
split: an act whose window has restarted, and whose body is one to two first flights long, goes in two halves at once,
each on a connection of its own, to a server that joins them (serve() says "split" in X-Seq-Reads) - against serve() on
a port of this machine, with what reached it recorded: each request's port, its half and its act.

On this machine's loopback a first flight is far larger than on a network, and a pause may not restart the window: each
test sets the link's segment size and its window's restart as a robot's on Linux without BBR would have them.
"""
import time

import numpy as np
import pytest

httpx = pytest.importorskip("httpx")
pytest.importorskip("uvicorn")

from sequence_ai.engine import serve as serving  # noqa: E402
from sequence_ai.engine.serve import App  # noqa: E402
from sequence_ai.network import connect, link as link_module  # noqa: E402
from sequence_ai.network.lines import Lines  # noqa: E402
from sequence_ai.network.link import Link  # noqa: E402
from tests.engine.test_serve import _serving  # noqa: E402
from tests.network.test_link import serving as plain_serving  # noqa: E402
from tests.network.test_link_defaults import Server  # noqa: E402


class Seen:
    """`app` with each request that reaches it recorded: (path, the robot's port, its X-Seq-Half, its X-Seq-Request-Id)."""

    def __init__(self, app) -> None:
        self.app, self.requests = app, []

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http":
            headers = dict(scope["headers"])
            self.requests.append((scope["path"], scope["client"][1], headers.get(b"x-seq-half"),
                                  headers.get(b"x-seq-request-id")))
        await self.app(scope, receive, send)

    def acts(self) -> list:
        return [r for r in self.requests if r[0] == "/act"]


def restarting(link, mss: int = 1000):
    """The link as a robot's on Linux without BBR: a pause restarts its window, and every act here follows one past its
    deadline; its connection's segment size `mss`, so that a first flight carries 9 * mss bytes of a body."""
    link._window_at_stake = True
    link._mss = mss
    return link


@pytest.fixture
def at_once(monkeypatch):
    """No silence to wait for: each send is past the deadline of the one before it."""
    monkeypatch.setattr(link_module, "FIRST_TRIP_S", 0.0)
    monkeypatch.setattr(link_module, "RTO_FLOOR_S", 0.0)


def body_of(n: int, seed: int = 0) -> dict:
    return {"noise": np.random.default_rng(seed).integers(0, 256, n, dtype=np.uint8)}


def policy(observation):
    return {"sum": int(observation["noise"].astype(np.int64).sum()), "n": int(observation["noise"].size)}


def ready_for_halves(link) -> None:
    """Until the second connection - opened once the server said it joins halves - has answered."""
    link.wait_until_ready()
    deadline = time.monotonic() + 5
    while not link._second_ready and time.monotonic() < deadline:
        time.sleep(0.01)
    assert link._second_ready


def test_an_act_past_its_windows_restart_goes_in_halves_on_two_connections_and_reaches_the_policy_whole(at_once):
    seen = Seen(App(policy, jpeg=False, compress=False))
    with _serving(seen) as url, restarting(connect(url, check=False, ping_every_s=60.0)) as link:
        ready_for_halves(link)
        sent = [body_of(12_000, seed) for seed in range(4)]                     # 9,000 < 12,0xx <= 18,000 bytes
        got = [link.act(observation) for observation in sent]
        counters = link.stats()["counters"]
    assert [g["sum"] for g in got] == [int(o["noise"].astype(np.int64).sum()) for o in sent]
    assert counters["acts"] == 4 and counters["acts_in_halves"] == 4 and counters["acts_whole_again"] == 0
    halves = [r for r in seen.acts() if r[2] is not None]
    firsts, seconds = [r for r in halves if r[2] == b"1/2"], [r for r in halves if r[2] == b"2/2"]
    assert len(firsts) == len(seconds) == 4 and len(seen.acts()) == 8
    assert len({r[1] for r in firsts}) == 1 and len({r[1] for r in seconds}) == 1      # the link's, and the second
    assert {r[1] for r in firsts} != {r[1] for r in seconds}
    assert sorted(r[3] for r in firsts) == sorted(r[3] for r in seconds) and len({r[3] for r in firsts}) == 4


def test_acts_go_whole_where_halves_would_save_nothing(at_once):
    """A body a first flight carries, or more than two of them; a window a pause does not restart (BBR, macOS); a
    window a ping has kept; split=False: whole, on the link's own connection."""
    with _serving(App(policy, jpeg=False, compress=False)) as url:
        with restarting(connect(url, check=False, ping_every_s=60.0)) as link:
            ready_for_halves(link)
            for size in (8_000, 20_000):                                         # one first flight; more than two
                link.act(body_of(size))
            assert link.stats()["counters"]["acts_in_halves"] == 0
            link._window_at_stake = False
            link.act(body_of(12_000))
            assert link.stats()["counters"]["acts_in_halves"] == 0
        with restarting(connect(url, check=False, ping_every_s=60.0, split=False)) as link:
            link.wait_until_ready()
            link.act(body_of(12_000))
            assert link.stats()["counters"]["acts_in_halves"] == 0 and link._second is None


def test_a_ping_after_a_silence_past_its_deadline_leaves_the_next_act_a_restarted_window():
    """With a ping's round trip and 200 ms as the deadline: an act soon after the last send keeps its window, unless a
    ping since the last act went after a silence past it - that ping restarted the window, and the act finds it so."""
    link = restarting(connect("http://127.0.0.1:9", check=False))
    now = time.perf_counter()
    link._act_sent(now - 0.6, now - 0.6)
    link._ping_done(now - 0.05, now - 0.03)                       # 550 ms after the act: past 20 ms and 200 ms
    assert link._restarts_now()
    link._act_sent(now, now)                                      # the act's own upload grows the window again
    assert not link._restarts_now()
    link._ping_done(now + 0.1, now + 0.12)                        # 100 ms after it: the window kept
    assert not link._restarts_now()
    link.close()


def test_an_old_server_that_does_not_say_it_joins_halves_gets_every_act_whole_on_one_connection(at_once):
    """A server that does not name "split" in X-Seq-Reads - an older serve(), or another program - is sent each act
    whole, on the link's one connection, and the second connection is never opened."""
    old = Seen(Server(reads="zstd, zlib, once"))
    with plain_serving(old) as url, restarting(connect(url, check=False, ping_every_s=60.0, jpeg=False,
                                                       compress=False)) as link:
        link.wait_until_ready()
        for seed in range(3):
            link.act({"image": np.random.default_rng(seed).integers(0, 256, (80, 50, 3), dtype=np.uint8)})
        counters = link.stats()["counters"]
        opened = link._second_sent
    assert counters["acts"] == 3 and counters["acts_in_halves"] == 0 and opened is None
    assert len({r[1] for r in old.requests}) == 1 and all(r[2] is None for r in old.requests)


def test_a_second_half_that_does_not_reach_the_server_sends_the_act_whole_and_the_next_acts_whole_for_a_while(
        at_once, monkeypatch):
    """The second connection fails: the server answers the first half 424 once its wait is over, and the act goes whole
    on the link's connection; the acts after it go whole for WHOLE_AGAIN_S, then the second connection is opened again
    and acts go in halves again."""
    monkeypatch.setattr(serving, "HALF_WAIT_S", 0.2)
    calls: list = []

    def counting(observation):
        calls.append(1)
        return policy(observation)

    with _serving(App(counting, jpeg=False, compress=False)) as url, \
            restarting(connect(url, check=False, ping_every_s=60.0)) as link:
        ready_for_halves(link)
        real = link._second.post

        def lost(*args, **kwargs):
            raise httpx.ConnectError("the second connection is gone")

        link._second.post = lost
        observation = body_of(12_000, 7)
        began = time.perf_counter()
        got = link.act(observation)
        took = time.perf_counter() - began
        counters = link.stats()["counters"]
        assert got["sum"] == int(observation["noise"].astype(np.int64).sum()) and len(calls) == 1
        assert counters["acts_in_halves"] == 1 and counters["acts_whole_again"] == 1 and 0.2 <= took < 5
        link.act(body_of(12_000, 8))                                             # within WHOLE_AGAIN_S: whole
        assert link.stats()["counters"]["acts_in_halves"] == 1 and len(calls) == 2
        link._second.post, link._whole_until = real, 0.0                         # WHOLE_AGAIN_S over
        link.act(body_of(12_000, 9))                                             # opens the second connection again
        deadline = time.monotonic() + 5
        while not link._second_ready and time.monotonic() < deadline:
            time.sleep(0.01)
        link.act(body_of(12_000, 10))
        assert link.stats()["counters"]["acts_in_halves"] == 2 and len(calls) == 4


def test_lines_whose_links_send_halves_run_each_act_once(at_once):
    """network.lines sends each act on every line with one request id: lines that send it in halves, and lines that send
    it whole, run it once all the same."""
    calls: list = []

    def counting(observation):
        calls.append(1)
        return policy(observation)

    with _serving(App(counting, jpeg=False, compress=False)) as url:
        links = [restarting(Link(url, check=False, ping_every_s=60.0)) for _ in range(2)]
        links[1]._split = False                                                  # one line in halves, one whole
        robot = Lines(links)
        try:
            robot.wait_until_ready()
            ready_for_halves(links[0])
            sent = [body_of(12_000, seed) for seed in range(4)]
            got = [robot.act(observation) for observation in sent]
            time.sleep(0.2)                                                      # the slower copies answered too
            halves = links[0].stats()["counters"]["acts_in_halves"]
        finally:
            robot.close()
    assert [g["sum"] for g in got] == [int(o["noise"].astype(np.int64).sum()) for o in sent]
    assert len(calls) == 4 and halves >= 1


def test_close_stops_the_second_connections_thread():
    link = connect("http://127.0.0.1:9", check=False)
    link.close()
    with pytest.raises(RuntimeError):
        link._second_thread.submit(lambda: None)
