# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
Each HTTP response leaves a uvicorn server in one write (sequence_ai/network/one_write.py). Written in pieces, as
uvicorn writes one, a tunnel in front of the server held every piece after the first for one round trip: headers then
body took a GET p50 of 189.7 ms against a TCP connect of 89.5 ms; the same response in one write, 106.7 ms.

    python -m pytest -q tests/network/test_one_write.py
"""
from __future__ import annotations

import asyncio

import pytest
import uvicorn
from uvicorn.protocols.http.auto import AutoHTTPProtocol
from uvicorn.server import ServerState

from sequence_ai.network.one_write import OneWriteTransport, one_write_http


class _Transport:
    """What uvicorn's protocol calls on a transport, recording every write and the close."""

    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self.closed = self.aborted = False

    def write(self, data: bytes) -> None:
        self.writes.append(bytes(data))

    def close(self) -> None:
        self.closed = True

    def abort(self) -> None:
        self.aborted = True

    def is_closing(self) -> bool:
        return self.closed or self.aborted

    def get_extra_info(self, name, default=None):
        return default

    def get_write_buffer_size(self) -> int:
        return 0

    def pause_reading(self) -> None:
        pass

    def resume_reading(self) -> None:
        pass


async def _json_app(scope, receive, send):
    """An ASGI app that responds as a policy server's /act does: a start, then the whole body."""
    if scope["type"] != "http":
        return
    body = b'{"actions": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]}'
    await send({"type": "http.response.start", "status": 200,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


def _serve_one(protocol_class) -> list[bytes]:
    """One keep-alive GET through `protocol_class` on uvicorn's own send path; the writes its transport received."""
    async def run() -> list[bytes]:
        config = uvicorn.Config(_json_app, log_level="warning", lifespan="off", http=protocol_class)
        config.load()
        proto = protocol_class(config=config, server_state=ServerState(), app_state={})
        transport = _Transport()
        proto.connection_made(transport)
        proto.data_received(b"GET /ping HTTP/1.1\r\nHost: server\r\n\r\n")
        for _ in range(50):                      # until it is sent: its task and the flush each take an iteration
            await asyncio.sleep(0)
            if any(b"actions" in w for w in transport.writes):
                break
        await asyncio.sleep(0)
        proto.connection_lost(None)
        return transport.writes

    return asyncio.run(run())


def test_a_response_leaves_in_one_write():
    writes = _serve_one(one_write_http())
    assert len(writes) == 1
    assert writes[0].startswith(b"HTTP/1.1 200 OK\r\n") and writes[0].endswith(b'1.0]]}')


def test_uvicorns_own_protocol_writes_a_response_in_pieces():
    """What the tunnel held back: uvicorn by itself writes the status line and headers, then the body. This shows that
    the transport is needed, and that the previous test would fail without it (uvicorn 0.54: h11_impl.send writes the
    headers at http.response.start and the body at http.response.body)."""
    writes = _serve_one(AutoHTTPProtocol)
    assert len(writes) >= 2 and writes[0].startswith(b"HTTP/1.1 200 OK\r\n") and b"actions" not in writes[0]


def test_writes_in_one_iteration_are_sent_together_and_a_later_write_is_sent_alone():
    async def run():
        inner = _Transport()
        t = OneWriteTransport(inner, asyncio.get_running_loop())
        t.write(b"head ")
        t.write(b"")                             # h11's end of a Content-Length body: nothing to send
        t.writelines([b"bo", b"dy"])
        assert inner.writes == []                # nothing is sent within the iteration
        await asyncio.sleep(0)
        assert inner.writes == [b"head body"]
        t.write(b"next")                         # written one iteration later, as a streamed response's next message
        await asyncio.sleep(0)
        assert inner.writes == [b"head body", b"next"]

    asyncio.run(run())


def test_a_close_sends_what_was_written_first_and_an_abort_drops_it():
    async def run():
        inner = _Transport()
        t = OneWriteTransport(inner, asyncio.get_running_loop())
        t.write(b"HTTP/1.1 400 Bad Request\r\n\r\n")
        assert t.get_write_buffer_size() == len(b"HTTP/1.1 400 Bad Request\r\n\r\n")   # flow control counts it
        t.close()
        assert inner.writes == [b"HTTP/1.1 400 Bad Request\r\n\r\n"] and inner.closed
        await asyncio.sleep(0)                   # the flush scheduled by the write finds nothing left
        assert len(inner.writes) == 1

        inner2 = _Transport()
        t2 = OneWriteTransport(inner2, asyncio.get_running_loop())
        t2.write(b"partial")
        t2.abort()
        await asyncio.sleep(0)
        assert inner2.writes == [] and inner2.aborted

    asyncio.run(run())


def test_a_connection_that_closed_meanwhile_is_not_written_to():
    async def run():
        inner = _Transport()
        t = OneWriteTransport(inner, asyncio.get_running_loop())
        t.write(b"late")
        inner.closed = True                      # the peer disconnected before the iteration ended
        await asyncio.sleep(0)
        assert inner.writes == []

    asyncio.run(run())


def test_everything_else_is_the_wrapped_transports():
    inner = _Transport()
    t = OneWriteTransport(inner, asyncio.new_event_loop())
    assert t.get_extra_info("peername", "x") == "x" and t.is_closing() is False
    with pytest.raises(AttributeError):
        t.no_such_method  # noqa: B018


# ── an app with middleware layers: they pass a response's start and body on one iteration apart (WholeResponse) ───────


def test_a_held_start_is_sent_with_its_body_and_a_start_never_sent_leaves_room_for_a_500():
    from sequence_ai.network.one_write import WholeResponse

    sent = []

    async def send(message):
        sent.append(message["type"])

    async def streamed(scope, receive, send_):
        await send_({"type": "http.response.start", "status": 200, "headers": []})
        await asyncio.sleep(0)                   # one iteration apart, as the layers pass them on
        assert sent == []                        # the start is held
        await send_({"type": "http.response.body", "body": b"x"})

    asyncio.run(WholeResponse(streamed)({"type": "http"}, None, send))
    assert sent == ["http.response.start", "http.response.body"]

    async def raises(scope, receive, send_):
        await send_({"type": "http.response.start", "status": 200, "headers": []})
        raise RuntimeError("after its start")

    sent.clear()
    with pytest.raises(RuntimeError):
        asyncio.run(WholeResponse(raises)({"type": "http"}, None, send))
    assert sent == []                            # nothing was sent: the server can still send its error response
