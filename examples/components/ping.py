# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
ping, alone: a WebSocket server that pings each connection every 100 ms, so a robot's connection is never idle.

    python examples/components/ping.py              # needs websockets; serves briefly with one client connected
"""
import asyncio

import websockets

from sequence_ai.network import ping


async def handler(ws):
    async for message in ws:
        await ws.send(message)                 # your policy's response here


async def main():
    async with websockets.serve(handler, "127.0.0.1", 0, **ping.websockets_options()) as server:
        port = server.sockets[0].getsockname()[1]
        async with websockets.connect(f"ws://127.0.0.1:{port}") as robot:       # any client: its library answers pings
            await robot.send("observation")
            print("response:", await robot.recv())
            await asyncio.sleep(0.5)           # idle: the server pings, the client's library pongs
            for connection in server.connections:
                print("the server measured the connection's RTT from its pings: "
                      f"{connection.latency * 1000:.2f} ms")


asyncio.run(main())
