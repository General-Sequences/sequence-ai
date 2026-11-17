# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
window, alone: the socket options this machine accepts, and a ping while a connection is idle.

    python examples/components/window.py
"""
import socket
import time

from sequence_ai.network import window

print("socket options this machine accepts:", window.socket_options())
print("restarts a window after a pause:", window.restarts_after_idle())

server = socket.create_server(("127.0.0.1", 0))
sock = socket.create_connection(server.getsockname())
print(f"BBR on your own socket: {window.keep(sock)}; its congestion control: {window.congestion_of(sock)}")

pings = []
warm = window.Warm(ping=lambda: pings.append(time.perf_counter()), every_s=0.05)     # your connection's own ping here
warm.used()                    # you just sent something
time.sleep(0.3)                # the robot executes its actions: the connection is idle
warm.close()
print(f"pings in 0.3 s of idle time: {len(pings)}")
sock.close()
server.close()
