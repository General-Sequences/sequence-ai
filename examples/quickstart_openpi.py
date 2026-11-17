# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
Quickstart for openpi users: openpi's own client, with one line of sequence-ai on the connection it holds.

    python examples/quickstart_openpi.py

Needs openpi-client (packages/openpi-client in openpi's repository) and websockets. It starts a stand-in server on this
machine that speaks openpi's protocol (metadata on connect, then a response for each observation, both in openpi's
msgpack layout), so the line that matters can be seen working without a GPU or a checkpoint. Against a real openpi
policy server the client side is the same two lines: examples/openpi_robot.py takes its host and port.
"""
import threading
import time

import numpy as np
from openpi_client import msgpack_numpy, websocket_client_policy
from websockets.sync.server import serve

from sequence_ai.network import openpi


def stand_in(websocket) -> None:
    """What openpi's WebsocketPolicyServer does, with zeros in place of a model's actions."""
    websocket.send(msgpack_numpy.packb({"stand_in": True}))
    for message in websocket:
        observation = msgpack_numpy.unpackb(message)
        actions = np.zeros((10, 7), dtype=np.float32) + observation["observation/state"][0]
        websocket.send(msgpack_numpy.packb({"actions": actions}))


server = serve(stand_in, "127.0.0.1", 0, compression=None, max_size=None)
threading.Thread(target=server.serve_forever, daemon=True).start()

policy = websocket_client_policy.WebsocketClientPolicy("127.0.0.1", server.socket.getsockname()[1])
kept = openpi.keep(policy)             # the line this repository adds: BBR, TCP keepalive, a ping when idle
print(f"BBR on its socket: {kept.bbr}; TCP keepalive options accepted: {len(kept.keepalive)}")

observation = {"observation/state": np.ones(7, dtype=np.float32), "prompt": "pick up the block"}
for step in range(3):
    actions = policy.infer(observation)["actions"]         # openpi's own call, unchanged
    print(f"infer {step}: a chunk of {actions.shape[0]} actions")
    time.sleep(0.5)                                        # the robot executes the chunk: the connection is idle
print(f"pings answered while it was idle: {kept.warm.pings}")
kept.close()
server.shutdown()
