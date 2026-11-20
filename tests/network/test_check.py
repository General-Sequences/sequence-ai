# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
python -m sequence_ai.check against serve() on this machine, reached as a robot reaches it - sequence_ai.connect(url),
each act an observation handed over and the act after it, on the robot's stream: a server that runs its model on a
thread of its own and hands out an example observation, one that runs its model in its event loop and says nothing of
how long it took, one whose ping does not say what it is, one that takes no robot's stream - and against no server, a
server that wants a token, and a string that is no address.
"""
from __future__ import annotations

import contextlib
import os
import socket
import sys
import threading
import time

import numpy as np
import pytest

pytest.importorskip("httpx")
pytest.importorskip("uvicorn")
pytest.importorskip("sequence_base")         # the robot's stream: serve() takes it where sequence-base is installed

from sequence_ai import check  # noqa: E402
from sequence_ai.engine import serve as serving  # noqa: E402

MODEL_S = 0.15
ACTS = ("pings in inference", "server time", "first act")      # the lines that send an act


def model(observation: dict) -> dict:
    """The policy: MODEL_S of work, then a chunk of actions."""
    time.sleep(MODEL_S)
    return {"actions": np.zeros((8, 7), np.float32)}


def observation() -> dict:
    """A robot's observation: a camera frame, channels first, and the arm's state."""
    return {"image": np.zeros((3, 96, 96), np.uint8), "state": np.arange(7, dtype=np.float32)}


class Silent(serving.App):
    """serve() as a server of another kind may be written: its model run in its event loop, so nothing else - a ping,
    anything on a robot's stream - is answered while it runs, and its answers say nothing of how long it took (no
    handler_ms in their perf)."""

    async def _run(self, items: list) -> list:
        dispatched = time.perf_counter()
        return [(output, took, dispatched, len(items)) for output, took in self._call(items)]

    async def _policy_on(self, observation, **said) -> tuple:
        return await super()._policy_on(observation, **{**said, "began": None})


class Anonymous:
    """serve() behind a ping of another program's: 200, and nothing said of what it is or of the robot's stream."""

    def __init__(self, app: serving.App) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http" and scope["path"] == check.PING:
            await send({"type": "http.response.start", "status": 200,
                        "headers": [(b"content-type", b"text/plain"), (b"content-length", b"4")]})
            await send({"type": "http.response.body", "body": b"pong"})
            return
        await self.app(scope, receive, send)

    def close(self) -> None:
        self.app.close()


@contextlib.contextmanager
def serving_on(app):
    """`app` on a port of this machine, as serve() runs it: its address while it runs."""
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
        app.close()


@pytest.fixture(autouse=True)
def this_machine(monkeypatch):
    """The window's and the Wi-Fi's lines read this machine, which differs between machines: fixed here."""
    monkeypatch.setattr(check, "_window_rows", lambda: [check.Row("ok", "send window", "kept")])
    monkeypatch.setattr(check, "_wifi_row", lambda: check.Row("--", "Wi-Fi power saving", "not read"))


def by_name(rows) -> dict:
    return {row.name: row for row in rows}


def test_a_server_that_does_it_all_has_every_line_ok_with_its_own_example():
    with serving_on(serving.App(model, example=observation())) as url:
        rows, status = check.run(url, pings=5)
    got = by_name(rows)
    for name in ("server", "round trip", "pings when idle", *ACTS):
        assert got[name].status == "ok", got[name]
    server, idle = got["server"].detail, got["pings when idle"].detail
    assert server.startswith(f"sequence-ai {serving._version()}, pid {os.getpid()}, up ")
    # the speed-ups its warm-up kept (engine.optimize): none, a model with no torch module and no jax.jit function
    assert server.endswith(", policy model, no speed-up on; it answers /ping with 200 and takes the robot's stream "
                           "(x-seq-stream/3)")
    assert "after each 100.0 ms of idle time" in idle and "until 60 s after its last observation or act" in idle
    assert "the server says it took" in got["server time"].detail and "GET /example" in got["first act"].detail
    assert not {"pictures as JPEG", "compression"} & set(got) and status == 0


def test_a_server_that_runs_its_model_in_its_event_loop_and_says_nothing_has_those_lines_no():
    with serving_on(Silent(model)) as url:
        rows, status = check.run(url, observation=observation(), pings=5)
    got = by_name(rows)
    assert got["pings in inference"].status == "no" and "held while the model ran" in got["pings in inference"].detail
    assert got["server time"].status == "no" and "no handler_ms" in got["server time"].detail
    assert got["server"].status == "ok" and "--observation" in got["first act"].detail and status == 1


def test_no_observation_and_no_example_says_what_the_act_lines_need():
    with serving_on(serving.App(model)) as url:
        rows, status = check.run(url, pings=2)
    got = by_name(rows)
    for name in ACTS:
        assert got[name].status == "--" and "--observation" in got[name].detail
    assert status == 0


def test_no_server_a_server_that_wants_a_token_and_no_address_end_with_status_2_and_say_why():
    rows, status = check.run("http://127.0.0.1:9", wait_s=0.3)
    assert status == 2 and rows == [check.Row("no", "server", "http://127.0.0.1:9 was not ready in 0.3 s: nothing "
                                                              "answered")]
    with serving_on(serving.App(model, token="s3cret", example=observation())) as url:
        began = time.monotonic()
        rows, status = check.run(url, wait_s=30.0)
        assert status == 2 and rows[0].detail == "refused this machine's ping (401): give it with --token"
        assert time.monotonic() - began < 5.0                # said at once: a refusal is not waited out
        rows, status = check.run(url, token="not-it", wait_s=30.0)
        assert status == 2 and rows[0].detail == "refused this machine's ping (401)"
        rows, status = check.run(url, token="s3cret", pings=2)
        assert status == 0 and by_name(rows)["first act"].status == "ok", rows
    rows, status = check.run("127.0.0.1:8000")
    assert status == 2 and rows[0].detail == "127.0.0.1:8000 is not an address: give it as http://<host>:<port>"


def test_a_refusal_is_said_as_soon_as_the_server_listens_not_at_the_end_of_the_wait():
    with socket.socket() as free:                       # a port with nothing on it: a connect there is refused at once
        free.bind(("127.0.0.1", 0))
        port = free.getsockname()[1]
    app = serving.App(model, token="s3cret")
    server = serving._uvicorn(app, "warning")
    starts = threading.Timer(1.0, lambda: server.run(sockets=serving._listen("127.0.0.1", port)))
    starts.daemon = True
    starts.start()
    try:
        began = time.monotonic()
        rows, status = check.run(f"http://127.0.0.1:{port}", wait_s=30.0)
        assert status == 2 and rows[0].detail == "refused this machine's ping (401): give it with --token"
        assert 1.0 <= time.monotonic() - began < 10.0        # after the server came up, long before the wait's end
    finally:
        server.should_exit = True
        starts.join(10)
        app.close()


def test_main_prints_one_line_a_check_and_exits_with_its_status(capsys):
    with serving_on(serving.App(model, example=observation())) as url:
        with pytest.raises(SystemExit) as ended:
            check.main([url])
    out = capsys.readouterr().out
    assert ended.value.code == 0 and out.startswith(f"sequence-ai check: {url}")
    assert "  ok  pings in inference" in out


def test_the_flags_of_the_http_path_are_gone(capsys):
    for gone in (["--lines", "eth0,wwan0"], ["--act-path", "/act"], ["--ping-path", "/ping"]):
        with pytest.raises(SystemExit) as ended:
            check.main(["http://127.0.0.1:9", *gone])
        assert ended.value.code == 2 and "unrecognized arguments" in capsys.readouterr().err


def test_a_server_that_answers_its_ping_but_does_not_say_it_is_serve_may_be_another_program():
    with serving_on(Anonymous(serving.App(model))) as url:
        rows, status = check.run(url, observation=observation(), pings=2)
    server = by_name(rows)["server"]
    assert server.status == "no" and "is not one serve() runs" in server.detail and status == 2


@pytest.mark.filterwarnings("ignore::UserWarning")         # wait_until_ready() warns that the stream did not open
class _NoStream:
    """A server that answers /ping as serve() does and takes no robot's stream: X-Seq-Server on its answers, no
    X-Seq-Act-Formats."""

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return
        if scope["path"] == "/example":                              # its example, as serve() hands one out
            from sequence_ai.network import msgpack_io

            body, kind = msgpack_io.pack(observation()), msgpack_io.MEDIA_TYPE.encode()
        else:
            body, kind = b'{"server": "sequence-ai", "version": "0", "policy": "model"}', b"application/json"
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", kind), (b"content-length", str(len(body)).encode()),
                                (b"x-seq-server", b"sequence-ai/0")]})
        await send({"type": "http.response.body", "body": body})

    def close(self):
        pass


def test_a_server_that_takes_no_robots_stream_says_so_and_sends_no_act():
    with serving_on(_NoStream()) as url:
        rows, status = check.run(url, pings=2)
    got, said = by_name(rows), "does not take the robot's stream (no x-seq-stream/3 in X-Seq-Act-Formats)"
    assert got["server"].status == "no" and said in got["server"].detail
    for name in ACTS:
        assert got[name] == check.Row("--", name, "needs a server that takes the robot's stream")
    assert status == 1


def test_on_macos_the_wi_fi_line_says_it_was_not_checked_and_whether_the_machine_is_on_wi_fi(monkeypatch):
    monkeypatch.undo()                                   # the real _wifi_row, not the fixture's
    from sequence_ai.network import wifi_power

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(wifi_power, "mac_wifi_iface", lambda: "en0")
    row = check._wifi_row()
    assert row.status == "--" and "not checked" in row.detail and "(en0)" in row.detail
    monkeypatch.setattr(wifi_power, "mac_wifi_iface", lambda: None)
    assert "not on Wi-Fi" in check._wifi_row().detail
