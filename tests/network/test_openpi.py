# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
openpi.keep(): openpi's own client (openpi_client, installed from its repository) against a server on this machine that
speaks its protocol (metadata on connect, then a response for each observation, both in its msgpack layout). Skipped
where openpi_client is not installed.
"""
import socket
import threading
import time

import numpy as np
import pytest

from sequence_ai.network import openpi

pytest.importorskip("websockets")
openpi_client = pytest.importorskip("openpi_client")

from openpi_client import msgpack_numpy, websocket_client_policy  # noqa: E402
from websockets.sync.server import serve  # noqa: E402


@pytest.fixture()
def server_port():
    def handler(ws):
        ws.send(msgpack_numpy.packb({"robot": "test"}))
        for message in ws:
            observation = msgpack_numpy.unpackb(message)
            ws.send(msgpack_numpy.packb({"actions": observation["state"] * 2}))

    server = serve(handler, "127.0.0.1", 0, compression=None, max_size=None)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.socket.getsockname()[1]
    server.shutdown()


def test_the_clients_own_responses_are_unchanged_and_its_connection_is_pinged_while_idle(server_port):
    policy = websocket_client_policy.WebsocketClientPolicy("127.0.0.1", server_port)
    kept = openpi.keep(policy, every_s=0.03)
    try:
        assert policy.get_server_metadata() == {"robot": "test"}
        state = np.arange(7, dtype=np.float32)
        assert np.array_equal(policy.infer({"state": state})["actions"], state * 2)
        time.sleep(0.3)                                       # idle: each ping waits for its pong
        assert kept.warm.pings >= 3
        assert np.array_equal(policy.infer({"state": state + 1})["actions"], (state + 1) * 2)
        sock = policy._ws.socket
        assert sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE) != 0
        assert (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1) in kept.keepalive
        assert isinstance(kept.bbr, bool) and (kept.congestion == "bbr") == kept.bbr
    finally:
        kept.close()
    before = kept.warm.pings
    time.sleep(0.15)
    assert kept.warm.pings == before                          # closed: no more pings; the connection stays the client's
    assert np.array_equal(policy.infer({"state": np.ones(3, np.float32)})["actions"], np.full(3, 2, np.float32))


def test_a_client_that_keeps_no_websocket_connection_is_refused():
    class Other:
        def infer(self, obs):
            return obs

    with pytest.raises(TypeError, match="_ws"):
        openpi.keep(Other())
