# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
tcp_link, alone: a connection's RTT, retransmissions and congestion window, read from the kernel.

    python examples/components/tcp_link.py          # prints the fields on Linux; None elsewhere
"""
import socket

from sequence_ai.network import tcp_link

server = socket.create_server(("127.0.0.1", 0))
client = socket.create_connection(server.getsockname())
conn, _ = server.accept()
client.sendall(b"x" * 20_000)
conn.recv(65536)
print("this connection, as the server's kernel reports it:", tcp_link.read(conn))
for s in (client, conn, server):
    s.close()
