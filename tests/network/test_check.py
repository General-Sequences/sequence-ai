# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
python -m sequence_ai.check against two servers on this machine: one that runs its model on a thread of its own, says
what it restores and how long its model took and hands out an example observation, and one that runs its model in its
event loop and says nothing - and against no server, and a server that wants a token.
"""
import asyncio
import json
import re
import sys
import time

import numpy as np
import pytest

pytest.importorskip("httpx")
pytest.importorskip("uvicorn")
pytest.importorskip("PIL")

from sequence_ai import check  # noqa: E402
from sequence_ai.network import compressed, frames, msgpack_io  # noqa: E402
from tests.network.test_link import frame, serving  # noqa: E402

MODEL_S = 0.15


class Server:
    def __init__(self, *, good: bool, token=None, ping: int = 404, says_who: bool = True) -> None:
        self.good, self.token, self.ping, self.says_who = good, token, ping, says_who

    async def __call__(self, scope, receive, send) -> None:
        body = b""
        while True:
            message = await receive()
            body += message.get("body", b"")
            if not message.get("more_body"):
                break
        path, headers, status, content = scope["path"], [], 200, b""
        if self.good:
            headers.append((b"x-seq-reads", b"jpeg, zstd, zlib"))
            if self.says_who:
                headers.append((b"x-seq-server", b"sequence-ai/0.63.0"))
        if self.token and dict(scope["headers"]).get(b"authorization") != f"Bearer {self.token}".encode():
            status = 401
        elif path == "/ping":
            status = 200 if self.good else self.ping
            if self.good and self.says_who:
                headers.append((b"content-type", b"application/json"))
                content = json.dumps({"server": "sequence-ai", "version": "0.63.0", "up_s": 120.0, "pid": 4242,
                                      "policy": "toy"}).encode()
        elif path == "/example" and self.good:
            content = msgpack_io.pack(observation())
        elif path == "/act":
            msgpack = frames.unpack_pictures(compressed.expand(msgpack_io.unpack(body), limit=64 << 20)[0])
            if self.good:
                await asyncio.to_thread(time.sleep, MODEL_S)         # the model on a thread of its own
                headers.append((b"server-timing", f"act;dur={MODEL_S * 1000:.1f}".encode()))
            else:
                time.sleep(MODEL_S)                                  # the model in the event loop: nothing answers
            content = msgpack_io.pack({"actions": np.zeros((8, 7), np.float32), "seen": list(msgpack)})
        else:
            status = 404
        await send({"type": "http.response.start", "status": status,
                    "headers": headers + [(b"content-length", str(len(content)).encode())]})
        await send({"type": "http.response.body", "body": content})


def observation() -> dict:
    return {"image": frame(), "state": np.arange(7, dtype=np.float32)}


@pytest.fixture(autouse=True)
def this_machine(monkeypatch):
    """The window's and the Wi-Fi's lines read this machine, which differs between machines: fixed here."""
    monkeypatch.setattr(check, "_window_rows", lambda: [check.Row("ok", "send window", "kept")])
    monkeypatch.setattr(check, "_wifi_row", lambda: check.Row("--", "Wi-Fi power saving", "not read"))


def by_name(rows) -> dict:
    return {row.name: row for row in rows}


def test_a_server_that_does_it_all_has_every_line_ok_with_its_own_example():
    with serving(Server(good=True)) as url:
        rows, status = check.run(url, pings=5)
    got = by_name(rows)
    for name in ("server", "round trip", "pings when idle", "pings in inference", "pictures as JPEG", "compression",
                 "server time", "first act"):
        assert got[name].status == "ok", got[name]
    assert "jpeg, zlib, zstd" in got["server"].detail and "GET /example" in got["first act"].detail
    assert got["server"].detail.startswith("sequence-ai 0.63.0, pid 4242, up 2.0 min, policy toy;")
    assert "1 in this observation" in got["pictures as JPEG"].detail
    assert status == 0


def test_a_server_that_runs_its_model_in_its_event_loop_and_says_nothing_has_those_lines_no():
    with serving(Server(good=False)) as url:
        rows, status = check.run(url, observation=observation(), pings=5)
    got = by_name(rows)
    assert got["pings in inference"].status == "no" and "held while the model ran" in got["pings in inference"].detail
    for name in ("server", "pings when idle", "pictures as JPEG", "compression", "server time"):
        assert got[name].status == "no", got[name]
    assert re.search(r"would go from 230,\d{3} to \d", got["pictures as JPEG"].detail) and status == 1


def test_no_observation_and_no_example_says_what_the_act_lines_need():
    with serving(Server(good=False)) as url:
        rows, _ = check.run(url, pings=2)
    got = by_name(rows)
    assert got["pings in inference"].status == "--" and "--observation" in got["pings in inference"].detail


def test_no_server_and_a_server_that_wants_a_token_end_with_status_2_and_say_why():
    rows, status = check.run("http://127.0.0.1:9", wait_s=0.3)
    assert status == 2 and rows[0].name == "server" and "not ready" in rows[0].detail
    with serving(Server(good=True, token="s3cret")) as url:
        rows, status = check.run(url, wait_s=5.0)
        assert status == 2 and "--token" in rows[0].detail
        rows, status = check.run(url, token="s3cret", pings=2)
        assert status == 0


def test_main_prints_one_line_a_check_and_exits_with_its_status(capsys):
    with serving(Server(good=True)) as url:
        with pytest.raises(SystemExit) as ended:
            check.main([url])
    out = capsys.readouterr().out
    assert ended.value.code == 0 and out.startswith(f"sequence-ai check: {url}")
    assert "  ok  pings in inference" in out


def test_a_server_that_answers_its_ping_but_does_not_say_it_is_serve_may_be_another_program():
    with serving(Server(good=True, says_who=False)) as url:
        rows, status = check.run(url, pings=2)
    server = by_name(rows)["server"]
    assert server.status == "no" and "another program answers at this address" in server.detail and status == 1


def test_on_macos_the_wi_fi_line_says_it_was_not_checked_and_whether_the_machine_is_on_wi_fi(monkeypatch):
    monkeypatch.undo()                                   # the real _wifi_row, not the fixture's
    from sequence_ai.network import wifi_power

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(wifi_power, "mac_wifi_iface", lambda: "en0")
    row = check._wifi_row()
    assert row.status == "--" and "not checked" in row.detail and "(en0)" in row.detail
    monkeypatch.setattr(wifi_power, "mac_wifi_iface", lambda: None)
    assert "not on Wi-Fi" in check._wifi_row().detail
