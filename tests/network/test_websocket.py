# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
network.websocket on a real uvicorn server: the upgrade answered 101 with the accept key RFC 6455 gives and the app's
headers, or 403 when the app closes first; a message the client sends in fragments handed to the app whole; a ping
answered with a pong; a message over the size the server reads closing the connection with 1009.

    python -m pytest -q tests/network/test_websocket.py
"""
from __future__ import annotations

import base64
import contextlib
import os
import socket
import struct
import threading
import time

import pytest

uvicorn = pytest.importorskip("uvicorn")

from sequence_ai.network import websocket  # noqa: E402


async def _echo(scope, receive, send) -> None:
    """Accepts at /echo with a header of its own and sends each message back; closes before accepting anywhere else."""
    if scope["type"] != "websocket":
        return
    await receive()
    if scope["path"] != "/echo":
        await send({"type": "websocket.close", "code": 1008})
        return
    await send({"type": "websocket.accept", "headers": [(b"x-said", b"hello")]})
    while True:
        message = await receive()
        if message["type"] == "websocket.disconnect":
            return
        await send({"type": "websocket.send", "bytes": message.get("bytes") or message.get("text", "").encode()})


@contextlib.contextmanager
def _serving(app, **config):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error", lifespan="off",
                                           ws=websocket.protocol(), **config))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(500):
        if server.started:
            break
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield port
    finally:
        server.should_exit = True
        thread.join(10)


def _upgrade(port: int, path: str) -> tuple[socket.socket, bytes, bytes]:
    key = base64.b64encode(os.urandom(16)).decode()
    sock = socket.create_connection(("127.0.0.1", port), timeout=5)
    sock.sendall((f"GET {path} HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                  f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
    got = b""
    while b"\r\n\r\n" not in got:
        chunk = sock.recv(4096)
        if not chunk:
            break
        got += chunk
    head, _, rest = got.partition(b"\r\n\r\n")
    return sock, head, rest


def _client_frame(opcode: int, payload: bytes, fin: bool = True) -> bytes:
    mask = os.urandom(4)
    n = len(payload)
    first = (0x80 if fin else 0) | opcode
    head = struct.pack(">BB", first, 0x80 | n) if n < 126 else struct.pack(">BBH", first, 0x80 | 126, n)
    return head + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload))


def _read_frame(sock: socket.socket, buf: bytearray) -> tuple[int, bytes]:
    while True:
        for _, opcode, payload in websocket.read_frames(buf):
            return opcode, payload
        chunk = sock.recv(65536)
        if not chunk:
            raise ConnectionError("closed")
        buf += chunk


def test_the_upgrade_is_answered_101_with_the_accept_key_and_the_apps_headers():
    with _serving(_echo) as port:
        key = base64.b64encode(os.urandom(16)).decode()
        sock = socket.create_connection(("127.0.0.1", port), timeout=5)
        sock.sendall((f"GET /echo HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                      f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        got = b""
        while b"\r\n\r\n" not in got:
            got += sock.recv(4096)
        head = got.split(b"\r\n\r\n")[0].decode().lower()
        sock.close()
    assert head.startswith("http/1.1 101")
    assert f"sec-websocket-accept: {websocket.accept_key(key)}".lower() in head and "x-said: hello" in head
    # the key RFC 6455 section 1.3 gives for its example
    assert websocket.accept_key("dGhlIHNhbXBsZSBub25jZQ==") == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="


def test_an_app_that_closes_before_it_accepts_answers_403():
    with _serving(_echo) as port:
        sock, head, _ = _upgrade(port, "/elsewhere")
        sock.close()
    assert head.startswith(b"HTTP/1.1 403")


def test_a_message_in_fragments_reaches_the_app_whole_and_a_ping_is_answered_with_a_pong():
    with _serving(_echo) as port:
        sock, head, rest = _upgrade(port, "/echo")
        assert head.startswith(b"HTTP/1.1 101")
        buf = bytearray(rest)
        sock.sendall(_client_frame(0x2, b"hello, ", fin=False) + _client_frame(0x0, b"world", fin=True))
        assert _read_frame(sock, buf) == (0x2, b"hello, world")
        sock.sendall(_client_frame(0x9, b"are you there"))
        assert _read_frame(sock, buf) == (0xA, b"are you there")
        big = os.urandom(70_000)                      # past the 16-bit length: a 64-bit one each way
        mask = os.urandom(4)
        frame = struct.pack(">BBQ", 0x82, 0x80 | 127, len(big)) + mask + websocket._unmasked(big, mask)
        sock.sendall(frame)
        assert _read_frame(sock, buf) == (0x2, big)
        sock.close()


def test_a_message_over_the_size_the_server_reads_closes_the_connection_1009():
    with _serving(_echo, ws_max_size=1000) as port:
        sock, head, rest = _upgrade(port, "/echo")
        assert head.startswith(b"HTTP/1.1 101")
        buf = bytearray(rest)
        sock.sendall(_client_frame(0x2, b"x" * 2000))
        opcode, payload = _read_frame(sock, buf)
        sock.close()
    assert opcode == 0x8 and struct.unpack(">H", payload[:2])[0] == 1009
