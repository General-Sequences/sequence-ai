# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
openpi's own client, with the client-side components on the connection it already holds.

    from openpi_client import websocket_client_policy
    from sequence_ai.network import openpi

    policy = websocket_client_policy.WebsocketClientPolicy(host, port)
    kept = openpi.keep(policy)             # BBR and TCP keepalive on its socket; a WebSocket ping while it is idle
    action = policy.infer(observation)     # as before: the client, its protocol and its responses are openpi's own
    kept.close()

In one sentence: keep() takes the WebSocket connection the client opened → requests BBR on its socket and enables TCP
keepalive (window.keep, window.keepalive_options), each where this machine allows it → pings the connection after each
`every_s` of idle time and waits for the pong (window.Warm), so the robot's kernel never counts it idle and a dead
connection is detected before the next observation is sent on it → marks each infer() as a use.

WHY on the client's own connection: nothing of openpi's is replaced. The frames, the msgpack layout and the server are
unchanged; only the socket's options and the idle time between two infer() calls change.

On the server side the same is one option of the websockets library openpi's server is built on: `ping_interval=0.1` in
its `serve(...)` call (ping.websockets_options() returns it). That needs a change in openpi's
websocket_policy_server.py, which we have not tested with it; keep() needs none.

Verified against openpi-client 0.1.0 on a local connection (tests/network/test_openpi.py). Measured on a WebSocket
connection of the kind openpi's client holds (the websockets library on both ends, a stand-in for the client, two cloud
machines; bench/ws_bench.py, docs/measurements.md): a 100 KB upload 0.53 s after each response took 37.6, 44.4, 82.4 and
163.8 ms at RTTs of 11, 21, 41 and 81 ms, and 12.7, 22.7, 41.8 and 82.6 ms with keep(), 40 uploads per cell. With
openpi's own server and client (π0.5 on an L4, the client in a Linux virtual machine on a home connection, 301 KB per
request, one 45 s episode each): the time a request spent outside inference had a median of 181 and 152 ms without
keep(); with it, 92 ms, and 157 ms in an episode that lost packets; with the BBR module loaded as well, 86 and 91 ms.
bench/ measures your link, and window.py documents where the same mechanism helped and where it did not.

Reads the client's `_ws`, the connection it keeps: a client without one (another version, another transport) raises
TypeError with that reason.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from . import window

PONG_S = 5.0       # how long a pong may take: past it the connection is considered dead and the pings stop


@dataclass
class Kept:
    """What keep() applied to a client's connection: whether its socket accepted BBR, the congestion control it now
    uses, the keepalive options it accepted, and the pings that keep it warm (`warm.pings`: those answered so far)."""

    bbr: bool
    congestion: Optional[str]
    keepalive: list
    warm: window.Warm

    def close(self) -> None:
        """Stop the pings. The connection is the client's, and stays open."""
        self.warm.close()


def keep(policy: Any, *, every_s: float = 0.1, for_s: float = 60.0) -> Kept:
    """Apply the client-side components to `policy`'s connection (the module's docstring). `every_s`: the idle time
    after which a ping is sent; `for_s`: how long the pings continue after the last infer()."""
    ws = getattr(policy, "_ws", None)
    sock = getattr(ws, "socket", None)
    if ws is None or sock is None or not callable(getattr(ws, "ping", None)):
        raise TypeError("this client keeps no WebSocket connection at `_ws` (openpi_client's WebsocketClientPolicy "
                        "does): nothing to keep warm")
    bbr = window.keep(sock)
    took = []
    for level, name, value in window.keepalive_options():
        try:
            sock.setsockopt(level, name, value)
        except OSError:
            continue
        took.append((level, name, value))

    def ping() -> None:
        if not ws.ping().wait(PONG_S):
            raise TimeoutError(f"no pong in {PONG_S:g} s")

    warm = window.Warm(ping, every_s=every_s, for_s=for_s)
    infer = policy.infer

    def infer_and_mark(obs):
        try:
            return infer(obs)
        finally:
            warm.used()

    policy.infer = infer_and_mark
    warm.used()                    # the connection was just used to connect: idle time is counted from now
    return Kept(bbr=bbr, congestion=window.congestion_of(sock), keepalive=took, warm=warm)
