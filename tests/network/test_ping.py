# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
ping: the options make a server ping. A client written here (a WebSocket handshake over a plain socket, with each ping
answered by a pong as a robot's library would answer it) counts the ping frames that reach it, so the count reflects the
server and no client library. The pong matters: a server sends its next ping one interval after the previous pong
arrived.

A server given the options is waited on for PINGS pings, for as long as WITHIN_S - a quarter of the libraries' own
interval: pings that come sooner are the options' doing, however slowly a busy machine runs this process. A count in a
fixed 0.6 s measured that machine as well: each ping takes three wake-ups here (the server's at its interval, the
client's at the ping, the server's at the pong), and under a whole suite's load the window once held 4 pings where a
quiet machine sees 11. A server without them is watched for 0.4 s, and sends none however busy the machine.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import os
import socket
import threading
import time

import pytest

from sequence_ai.network import ping

PINGS = 6                  # the pings a server given the options is waited on for
DEFAULT_S = 20.0           # websockets' and uvicorn's own ping interval, without the options
WITHIN_S = DEFAULT_S / 4   # how long those pings are waited on for: far sooner than either library's own interval


def _pings_in(port: int, seconds: float, until: int | None = None) -> int:
    """The pings that reach a connection to `port` in `seconds`, each answered by its pong; with `until`, no more than
    that many - it returns as soon as they have come."""
    key = base64.b64encode(os.urandom(16)).decode()
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    s.sendall((f"GET / HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
               f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
    buf = b""
    while b"\r\n\r\n" not in buf:
        buf += s.recv(4096)
    assert buf.startswith(b"HTTP/1.1 101"), buf[:80]
    buf = buf.split(b"\r\n\r\n", 1)[1]
    s.settimeout(0.05)
    end, pings = time.time() + seconds, 0
    while time.time() < end and (until is None or pings < until):
        try:
            buf += s.recv(4096)
        except (socket.timeout, TimeoutError):
            pass
        while len(buf) >= 2 and (until is None or pings < until):     # a ping past `until` is left uncounted
            opcode, length = buf[0] & 0x0F, buf[1] & 0x7F     # a server's frames are unmasked; a ping's are short
            if len(buf) < 2 + length:
                break
            if opcode == 0x9:
                pings += 1
                mask, payload = os.urandom(4), buf[2:2 + length]
                s.sendall(bytes([0x8A, 0x80 | length]) + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload)))
            buf = buf[2 + length:]
    s.close()
    return pings


def test_the_options_are_those_each_server_accepts_and_a_bad_interval_is_refused():
    assert ping.websockets_options() == {"ping_interval": 0.1, "ping_timeout": 20.0}
    assert ping.uvicorn_options(0.05, 3) == {"ws_ping_interval": 0.05, "ws_ping_timeout": 3.0}
    for bad in (0, -1):
        with pytest.raises(ValueError):
            ping.websockets_options(bad)
        with pytest.raises(ValueError):
            ping.uvicorn_options(0.1, bad)


@contextlib.contextmanager
def _websockets_server(serve=None, **options):
    websockets = pytest.importorskip("websockets")
    ready, box = threading.Event(), {}

    async def handler(ws):
        with contextlib.suppress(Exception):
            async for _ in ws:
                pass

    async def main():
        box["stop"] = asyncio.get_running_loop().create_future()
        async with (serve or websockets.serve)(handler, "127.0.0.1", 0, **options) as server:
            box["port"], box["loop"] = server.sockets[0].getsockname()[1], asyncio.get_running_loop()
            ready.set()
            await box["stop"]

    thread = threading.Thread(target=lambda: asyncio.run(main()), daemon=True)
    thread.start()
    assert ready.wait(5)
    try:
        yield box["port"]
    finally:
        box["loop"].call_soon_threadsafe(box["stop"].set_result, None)
        thread.join(5)


def test_a_websockets_server_with_the_options_pings_every_interval_and_one_without_does_not():
    with _websockets_server(**ping.websockets_options(0.05)) as port:
        assert _pings_in(port, WITHIN_S, until=PINGS) == PINGS
    with _websockets_server() as port:
        assert _pings_in(port, 0.4) == 0                      # the library's default interval is 20 s


def test_serve_is_websockets_own_server_pinging_every_interval_and_an_option_passed_wins():
    with _websockets_server(ping.serve, every_s=0.05) as port:
        assert _pings_in(port, WITHIN_S, until=PINGS) == PINGS
    with _websockets_server(ping.serve, ping_interval=None) as port:
        assert _pings_in(port, 0.4) == 0                      # the caller's own option: no pings
    with pytest.raises(ValueError):
        ping.serve(None, "127.0.0.1", 0, every_s=0)


@pytest.mark.parametrize("implementation", ["auto", "websockets-sansio", "websockets"])
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_a_uvicorn_server_with_the_options_pings_its_websocket_connections(implementation):
    uvicorn = pytest.importorskip("uvicorn")
    pytest.importorskip("websockets")
    if implementation not in uvicorn.config.WS_PROTOCOLS:
        pytest.skip(f"this uvicorn has no {implementation} implementation")

    async def app(scope, receive, send):
        if scope["type"] != "websocket":
            return
        await receive()
        await send({"type": "websocket.accept"})
        while (await receive())["type"] != "websocket.disconnect":
            pass

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error", lifespan="off",
                                           ws=implementation, **ping.uvicorn_options(0.05)))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 5
    while not server.started and time.time() < deadline:
        time.sleep(0.01)
    try:
        pings = _pings_in(server.servers[0].sockets[0].getsockname()[1], WITHIN_S, until=PINGS)
    finally:
        server.should_exit = True
        thread.join(5)
    if implementation == "websockets-sansio" and pings == 0:
        pytest.skip(f"uvicorn {uvicorn.__version__}'s websockets-sansio sends no pings (0.54's does)")
    assert pings == PINGS
