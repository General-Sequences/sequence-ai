# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
connect(): the client-side components on one connection, against a policy server on this machine that records what
reached it: each request's path, the port it came from, its size, and when it started and ended.
"""
import asyncio
import contextlib
import json
import threading
import time

import numpy as np
import pytest

httpx = pytest.importorskip("httpx")
uvicorn = pytest.importorskip("uvicorn")

from sequence_ai.network import compressed, connect, msgpack_io  # noqa: E402
from sequence_ai.network.link import _server_seconds  # noqa: E402


class Policy:
    def __init__(self, act_s: float = 0.0, as_json: bool = False) -> None:
        self.act_s, self.as_json, self.seen = act_s, as_json, []

    async def __call__(self, scope, receive, send) -> None:
        body = b""
        while True:
            message = await receive()
            body += message.get("body", b"")
            if not message.get("more_body"):
                break
        path, port, start = scope["path"], scope["client"][1], time.perf_counter()
        status, kind, response = 200, "text/plain", b""
        if path == "/boom":
            status = 500
        elif path != "/ping":
            observation, _ = compressed.expand(msgpack_io.unpack(body), limit=64 << 20)
            await asyncio.sleep(self.act_s)
            actions = np.full((8, 7), float(observation["image"].mean()), dtype=np.float32)
            if self.as_json:
                kind, response = "application/json", json.dumps({"actions": actions.tolist()}).encode()
            else:
                kind = msgpack_io.MEDIA_TYPE
                response = msgpack_io.pack({"actions": actions, "state": observation["state"]})
        self.seen.append({"path": path, "port": port, "size": len(body), "start": start, "end": time.perf_counter()})
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", kind.encode()), (b"content-length", str(len(response)).encode()),
                                (b"server-timing", f"act;dur={self.act_s * 1000:.1f}".encode())]})
        await send({"type": "http.response.body", "body": response})

    def paths(self, path: str) -> list:
        return [r for r in self.seen if r["path"] == path]


@contextlib.contextmanager
def serving(app):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error", lifespan="off"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 5
    while not server.started and time.time() < deadline:
        time.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}"
    finally:
        server.should_exit = True
        thread.join(5)


def frame(step: int = 0) -> np.ndarray:
    yy, xx = np.mgrid[0:240, 0:320]
    return np.stack([(xx // 8) * 9 + step, (yy // 6) * 7, ((xx + yy) // 16) * 11], -1).astype(np.uint8)


def test_a_request_is_uploaded_and_returns_arrays_and_every_request_is_on_one_connection():
    policy = Policy()
    with serving(policy) as url, connect(url, ping_every_s=0.03, check=False) as link:
        for step in range(3):
            state = np.arange(7, dtype=np.float32) + step
            response = link.act({"image": frame(step), "state": state})
            assert response["actions"].shape == (8, 7) and response["actions"].dtype == np.float32
            assert np.array_equal(response["state"], state)
            time.sleep(0.1)                                   # idle: pings are sent
        snap = link.stats()
    assert snap["counters"]["acts"] == 3 and snap["counters"]["pings"] >= 3
    assert snap["latency_us"]["act_us"]["n"] == 3
    assert len(policy.paths("/act")) == 3 and len(policy.paths("/ping")) >= 3
    assert len({r["port"] for r in policy.seen}) == 1         # requests and pings alike: one connection, never a second


def test_observe_then_act_uploads_the_newest_observation_and_act_with_neither_is_refused():
    policy = Policy()
    with serving(policy) as url, connect(url, ping_path=None, check=False) as link:
        with pytest.raises(ValueError, match="observe\\(\\) has not been called"):
            link.act()
        assert policy.paths("/act") == []                         # refused here: nothing was sent
        link.observe({"image": frame(0), "state": np.zeros(7, np.float32)})
        link.observe({"image": frame(40), "state": np.ones(7, np.float32)})
        assert policy.paths("/act") == []                         # held: an observation is not a request
        response = link.act()
        assert len(policy.paths("/act")) == 1                     # one upload, of the newest
        assert response["state"].tolist() == [1.0] * 7 and response["actions"][0, 0] == np.float32(frame(40).mean())
        given = link.act({"image": frame(0), "state": np.zeros(7, np.float32)})     # handed to act(): sent as before
        assert given["state"].tolist() == [0.0] * 7 and len(policy.paths("/act")) == 2


def test_pings_are_sent_while_the_connection_is_idle_and_none_reaches_the_server_during_a_request():
    policy = Policy(act_s=0.25)
    with serving(policy) as url, connect(url, ping_every_s=0.03, check=False) as link:
        for _ in range(3):
            link.act({"image": frame(), "state": np.zeros(7, np.float32)})
            time.sleep(0.2)
    requests, pings = policy.paths("/act"), policy.paths("/ping")
    assert len(requests) == 3 and len(pings) >= 6
    for request in requests:
        inside = [p for p in pings if request["start"] < p["start"] < request["end"]]
        assert not inside                                     # a request in flight: the connection is not idle
    assert len({r["port"] for r in policy.seen}) == 1


def test_with_compress_less_is_uploaded_and_the_server_receives_the_same_array():
    plain, squeezed = Policy(), Policy()
    image = frame()
    with serving(plain) as url, connect(url, ping_path=None, check=False) as link:
        want = link.act({"image": image, "state": np.zeros(7, np.float32)})["actions"]
    with serving(squeezed) as url, connect(url, ping_path=None, compress=True, check=False) as link:
        got = link.act({"image": image, "state": np.zeros(7, np.float32)})["actions"]
    assert np.array_equal(want, got)                          # the mean of the frame the server received
    assert plain.paths("/act")[0]["size"] > image.nbytes
    assert squeezed.paths("/act")[0]["size"] < image.nbytes / 2


def test_a_json_response_is_read_as_json_and_an_error_status_raises():
    policy = Policy(as_json=True)
    with serving(policy) as url, connect(url, ping_path=None, check=False) as link:
        response = link.act({"image": frame(), "state": np.zeros(7, np.float32)})
        assert isinstance(response["actions"], list) and len(response["actions"]) == 8
        with pytest.raises(httpx.HTTPStatusError):
            link.act({"image": frame(), "state": np.zeros(7, np.float32)}, path="/boom")
    assert not policy.paths("/ping")                          # ping_path=None: no pings at all


def test_close_stops_the_pings():
    policy = Policy()
    with serving(policy) as url:
        link = connect(url, ping_every_s=0.03, check=False)
        link.act({"image": frame(), "state": np.zeros(7, np.float32)})
        time.sleep(0.15)
        link.close()
        time.sleep(0.1)                                       # a ping in flight when it closed has completed by now
        before = len(policy.paths("/ping"))
        time.sleep(0.2)
        assert before >= 2 and len(policy.paths("/ping")) == before


def test_the_servers_own_time_is_read_from_server_timing():
    assert _server_seconds("act;dur=276.0") == pytest.approx(0.276)
    assert _server_seconds("db;desc=x, act;dur=5") == pytest.approx(0.005)
    assert _server_seconds(None) == 0.0 and _server_seconds("act;dur=soon") == 0.0


def test_no_server_at_the_address_raises_with_what_to_check():
    import socket

    with socket.socket() as probe:                     # a port nothing listens on
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with connect(f"http://127.0.0.1:{port}", ping_path=None, check=False) as link:
        with pytest.raises(httpx.ConnectError) as refused:
            link.act({"state": [0.0]})
    message = str(refused.value)
    assert f"no response from http://127.0.0.1:{port}" in message and "0.0.0.0" in message


def test_without_httpx_connect_raises_with_what_to_install(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "httpx", None)    # as on a machine where it is not installed
    with pytest.raises(ImportError, match=r'pip install httpx'):
        connect("http://127.0.0.1:9", check=False)
