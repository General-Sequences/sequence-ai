# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
Each client connection's transport-level state, read from the server's own kernel (sequence_ai/network/tcp_link.py): its
TCP_INFO (the smoothed RTT and its variance, the retransmitted segments, the congestion window, the delivery rate) and a
request's upload time, from the bytes that carried its head to those that carried the last part of its body. A client's
window summarizes them over its recent requests (the RTT's p50 and p95, retransmitted segments per minute, the share of
its uploads that took two flights). Where the kernel provides none the fields are omitted.

    python -m pytest -q tests/network/test_tcp_link.py
"""
from __future__ import annotations

import asyncio
import collections
import socket
import struct
import sys

import pytest

LINUX = sys.platform.startswith("linux")


@pytest.fixture
def link(monkeypatch):
    """sequence_ai.network.tcp_link, with no connection left registered by another test."""
    from sequence_ai.network import tcp_link

    monkeypatch.setattr(tcp_link, "_peers", collections.OrderedDict())
    return tcp_link


def _tcp_info(size: int = 232) -> bytes:
    """A struct tcp_info as Linux returns one: 2 retransmitted segments unacknowledged and 7 in total, an RTT of 12.345
    ms with a variance of 0.678 ms, a congestion window of 10 and a delivery rate of 1.25 MB per second, truncated to
    `size` bytes."""
    raw = bytearray(232)
    for at, value in ((36, 2), (68, 12_345), (72, 678), (80, 10), (100, 7)):
        struct.pack_into("=I", raw, at, value)
    struct.pack_into("=Q", raw, 160, 1_250_000)
    return bytes(raw[:size])


class _Sock:
    def getsockopt(self, level, opt, size):
        return _tcp_info()[:size]


class _Transport:
    """An asyncio transport as tcp_link sees one: its peer, its socket, whether it is closing."""

    def __init__(self, peer, sock=None) -> None:
        self.peer, self.sock, self.closing = peer, sock, False

    def get_extra_info(self, name):
        return {"peername": self.peer, "socket": self.sock}.get(name)

    def is_closing(self) -> bool:
        return self.closing


def test_the_fields_are_read_from_tcp_info_and_a_shorter_struct_keeps_those_it_holds(link):
    assert link.parse(_tcp_info()) == {"rtt_ms": 12.345, "rttvar_ms": 0.678, "retrans": 2, "total_retrans": 7,
                                       "cwnd": 10, "delivery_kbps": 10_000.0}
    # a kernel older than 4.9, which has no delivery_rate
    assert link.parse(_tcp_info(104)) == {"rtt_ms": 12.345, "rttvar_ms": 0.678, "retrans": 2, "total_retrans": 7,
                                          "cwnd": 10}
    assert link.parse(_tcp_info(72)) == {"rtt_ms": 12.345, "retrans": 2}
    assert link.parse(_tcp_info(71)) is None


@pytest.mark.skipif(not LINUX, reason="TCP_INFO is Linux only")
def test_a_connections_tcp_info_is_read_from_its_socket_and_none_once_it_is_closed(link):
    with socket.create_server(("127.0.0.1", 0)) as listening:
        robot = socket.create_connection(listening.getsockname()[:2])
        conn, _ = listening.accept()
        try:
            got = link.read(conn)
        finally:
            conn.close()
            robot.close()
    assert got is not None and got["retrans"] == 0 and got["total_retrans"] == 0, got
    assert 0.0 <= got["rtt_ms"] < 20.0 and got["cwnd"] >= 1, got        # loopback: the RTT of its handshake
    assert link.read(conn) is None and link.read(None) is None


def test_a_kernel_that_provides_no_tcp_info_yields_no_fields(link, monkeypatch):
    class Refusing:
        def getsockopt(self, *a):
            raise OSError(92, "Protocol not available")       # a sandbox that withholds the counters

    monkeypatch.setattr(link, "_LINUX", True, raising=False)
    monkeypatch.setattr(socket, "TCP_INFO", getattr(socket, "TCP_INFO", 11), raising=False)
    assert link.read(_Sock())["rtt_ms"] == 12.345
    assert link.read(Refusing()) is None
    monkeypatch.delattr(socket, "TCP_INFO")                 # not Linux
    assert link.read(_Sock()) is None


def test_a_macs_python_that_defines_tcp_info_reads_no_fields(link, monkeypatch):
    """A Mac's own Python 3.9 defines socket.TCP_INFO (0x200) for Apple's struct tcp_info, which is laid out
    differently: read as Linux's, every field would be wrong. Off Linux nothing is read."""
    monkeypatch.setattr(link, "_LINUX", False, raising=False)
    monkeypatch.setattr(socket, "TCP_INFO", 0x200, raising=False)
    assert link.read(_Sock()) is None


def test_a_request_finds_its_connection_by_its_peer_and_a_tunnels_hop_is_marked(link, monkeypatch):
    monkeypatch.setattr(link, "_LINUX", True, raising=False)
    monkeypatch.setattr(socket, "TCP_INFO", getattr(socket, "TCP_INFO", 11), raising=False)
    tunnel = _Transport(("10.0.0.7", 5000), _Sock())
    for transport in (_Transport(("8.8.8.8", 6000), _Sock()), tunnel,
                      _Transport(("::ffff:8.8.8.8", 6001, 0, 0), _Sock())):
        assert isinstance(link.register(transport), link.Connection)
    assert link.of(("8.8.8.8", 6000)) == {"rtt_ms": 12.345, "rttvar_ms": 0.678, "retrans": 2, "total_retrans": 7,
                                          "cwnd": 10, "delivery_kbps": 10_000.0}
    assert link.of(("10.0.0.7", 5000))["peer_private"] is True
    assert "peer_private" not in link.of(("::ffff:8.8.8.8", 6001))          # a dual-stack socket's IPv4 peer
    assert link.of(("8.8.8.8", 6002)) is None and link.of(None) is None and link.of(("testclient", "x")) is None
    tunnel.closing = True
    assert link.of(("10.0.0.7", 5000)) is None
    link.register(_Transport(("8.8.4.4", 7000)))                              # a closed one is removed by the next
    assert ("10.0.0.7", 5000) not in link._peers
    assert link.register(_Transport("/run/server.sock")) is None             # a Unix socket's peer has no address


def test_a_robots_window_summarizes_its_rtt_its_retransmissions_per_minute_and_its_two_flight_uploads(link):
    w = link.Window()
    first, again = ("8.8.8.8", 40000), ("8.8.8.8", 40001)       # the robot's connection, then its next one
    for now, rtt, total, up, client in ((0.0, 10.0, 0, 0.0, first), (1.0, 20.0, 0, 0.3, first),
                                        (2.0, 30.0, 2, 20.0, first), (3.0, 40.0, 5, 0.0, first),
                                        (4.0, 100.0, 1, 60.0, again)):
        w.add("a1b2c3d4", now, {"rtt_ms": rtt, "total_retrans": total}, up, client)
    # retransmitted: 2 then 3 on its first connection, 1 on its next (counted from its start): 6 in 4 s, over one
    # minute; two flights: 20 ms at an RTT of 30 ms and 60 at 100, not 0.3 at 20
    assert w.summary("a1b2c3d4", 4.0) == {"acts": 5, "window_s": 4.0, "rtt_p50_ms": 30.0, "rtt_p95_ms": 100.0,
                                          "retrans_per_min": 6.0, "two_trip_share": 0.4}
    near = ("192.168.1.20", 50000)
    w.add("e5f6a7b8", 0.0, {"rtt_ms": 0.05, "total_retrans": 0, "peer_private": True}, 0.6, near)
    w.add("e5f6a7b8", 120.0, {"rtt_ms": 0.05, "total_retrans": 4, "peer_private": True}, 1.5, near)
    got = w.summary("e5f6a7b8", 120.0)
    # 4 over two minutes; at an RTT of 0.05 ms an upload of 0.6 ms is event-loop jitter, one of 1.5 ms a second flight
    assert (got["retrans_per_min"], got["two_trip_share"], got["peer_private"]) == (2.0, 0.5, True)
    w.add("c0ffee00", 5.0, None, 12.0, first)                    # the kernel reported nothing for its connection
    assert w.summary("c0ffee00", 5.0) is None and w.summary(None, 5.0) is None
    small = link.Window(acts=3, robots=2)
    for i in range(5):
        small.add("r1", float(i), {"rtt_ms": 1.0}, 0.0)
    assert small.summary("r1", 4.0)["acts"] == 3
    small.add("r2", 5.0, {"rtt_ms": 1.0}, 0.0)
    small.add("r3", 6.0, {"rtt_ms": 1.0}, 0.0)
    assert small.summary("r1", 6.0) is None and small.summary("r3", 6.0)["acts"] == 1


def test_an_upload_is_timed_by_when_its_bytes_arrived_not_by_when_the_server_read_them(link):
    conn = link.register(_Transport(("8.8.8.8", 41000)))
    got = {}

    async def receive():
        return {"type": "http.request", "body": b"{}", "more_body": False}

    def app(late: float | None, key: str):
        async def inner(scope, receive_, send):
            if late is not None:
                conn.last_in = late             # the last part of its body arrived a flight after its head ...
            await asyncio.sleep(0.05)           # ... and the server, busy with other requests, read it later
            await receive_()
            got[key] = link.upload_ms(scope)
        return link.Arrival(inner)

    async def run():
        conn.last_in = 1.0                      # when the bytes of its head arrived
        await app(1.08, "two flights")({"type": "http", "client": ("8.8.8.8", 41000)}, receive, None)
        conn.last_in = 2.0
        await app(None, "with its head")({"type": "http", "client": ("8.8.8.8", 41000)}, receive, None)
        await app(None, "not registered")({"type": "http", "client": ("8.8.8.8", 41001)}, receive, None)

    asyncio.run(run())
    assert got == {"two flights": 80.0, "with its head": 0.0, "not registered": None}
    assert link.upload_ms({"type": "http", "state": {"sequence_ai_head_in": 1.0}}) is None    # its body was not read
