# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
openpi's own client with the client-side components on its connection. Needs openpi-client and a running openpi policy
server:

    python examples/openpi_robot.py HOST PORT

Nothing of openpi's is replaced: the client, its protocol and its responses are openpi's own.
"""
import sys

from openpi_client import websocket_client_policy

from sequence_ai.network import openpi

policy = websocket_client_policy.WebsocketClientPolicy(sys.argv[1], int(sys.argv[2]))
kept = openpi.keep(policy)             # BBR and TCP keepalive on its socket; a WebSocket ping while it is idle
print(f"BBR: {kept.bbr} (congestion control {kept.congestion}); keepalive options accepted: {len(kept.keepalive)}")
print("server metadata:", policy.get_server_metadata())
# action = policy.infer(observation)   # as before
kept.close()
