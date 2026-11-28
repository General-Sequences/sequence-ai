#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
Measures a WebSocket policy link: one upload after each pause, on one connection.

A robot keeps one such connection open to a WebSocket policy server (openpi's client and server are built on the same
`websockets` library).

    python3 bench/ws_bench.py server PORT [--ping]
    python3 bench/ws_bench.py client HOST PORT VARIANT [--size 100000] [--pause 0.53] [--samples 40]

The server responds to each message with two bytes; `--ping` makes it ping each connection (ping.websockets_options()).
The client's VARIANT selects what the client side does:
    plain    nothing: the `websockets` client as it is
    bbr      BBR alone on the socket (window.keep); False in the result where the machine does not allow it

What runs: the client connects, uploads 1 MB once (which opens the window as a first large observation would), waits one
second, then uploads `size` bytes and waits for the response, with `pause` seconds after each response: 3 times
uncounted, then `samples` times counted. It prints one line, `RESULT {json}`: the median and the p95 of the counted
uploads, and every sample, in milliseconds.

The pause is counted from the response, as bench/client.py counts it: the connection has then sent nothing for the pause
plus one RTT.

Needs websockets: pip install websockets.
"""
import argparse
import asyncio
import json
import os
import statistics
import time

WARMUP = 3                      # uncounted uploads first: the window reaches its steady state for this pause
WIDEN = 1_000_000               # one upload of this size first, as a client that widens its window on connect does


def server(port: int, ping: bool) -> None:
    """Respond to every message on `port`; with `ping`, ping each connection with ping.websockets_options()."""
    import websockets

    from sequence_ai.network import ping as ping_options

    async def handler(ws):
        await ws.send(b"meta")                     # openpi's server sends its metadata first; a client reads it
        async for _ in ws:
            await ws.send(b"ok")

    async def main():
        options = ping_options.websockets_options() if ping else {}
        async with websockets.serve(handler, "0.0.0.0", port, compression=None, max_size=None, **options):
            await asyncio.get_running_loop().create_future()

    asyncio.run(main())


def client(host: str, port: int, variant: str, size: int = 100_000, pause: float = 0.53, samples: int = 40) -> dict:
    """Upload `size` bytes `pause` seconds after each response, and return the timings (the module's docstring)."""
    import websockets.sync.client as sync_client

    from sequence_ai.network import window

    # as a context manager: websockets 17 deprecates a sync connection opened without one
    with sync_client.connect(f"ws://{host}:{port}", compression=None, max_size=None) as ws:
        ws.recv()

        # one request and its reply on the connection, as a policy's client sends an observation
        class Policy:
            def __init__(self) -> None:
                self._ws = ws

            def infer(self, observation: bytes) -> bytes:
                self._ws.send(observation)
                return self._ws.recv()

        policy = Policy()
        bbr = None
        if variant == "bbr":
            bbr = window.keep(ws.socket)
        elif variant != "plain":
            raise SystemExit(f"VARIANT is plain or bbr; got {variant!r}")
        payload = os.urandom(size)
        policy.infer(os.urandom(WIDEN))
        time.sleep(1.0)
        rows = []
        for i in range(WARMUP + samples):
            started = time.perf_counter()
            policy.infer(payload)
            if i >= WARMUP:
                rows.append(round((time.perf_counter() - started) * 1000, 2))
            time.sleep(pause)
        p95 = sorted(rows)[int(0.95 * (len(rows) - 1))]
        result = {"bench": "ws", "variant": variant, "port": port, "size": size, "pause_s": pause, "n": len(rows),
                  "p50_ms": round(statistics.median(rows), 2), "p95_ms": round(p95, 2), "bbr": bbr,
                  "congestion": window.congestion_of(ws.socket), "samples_ms": rows}
    return result


def main(argv: "list[str] | None" = None) -> "dict | None":
    """The command line: `server PORT [--ping]` runs until stopped; `client ...` prints and returns its result."""
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    roles = parser.add_subparsers(dest="role", required=True)
    server_args = roles.add_parser("server")
    server_args.add_argument("port", type=int)
    server_args.add_argument("--ping", action="store_true", help="ping each connection every 100 ms")
    client_args = roles.add_parser("client")
    client_args.add_argument("host")
    client_args.add_argument("port", type=int)
    client_args.add_argument("variant", choices=["plain", "bbr"])
    client_args.add_argument("--size", type=int, default=100_000, help="bytes of each upload")
    client_args.add_argument("--pause", type=float, default=0.53, help="seconds of silence after each response")
    client_args.add_argument("--samples", type=int, default=40)
    args = parser.parse_args(argv)
    if args.role == "server":
        server(args.port, args.ping)
        return None
    result = client(args.host, args.port, args.variant, args.size, args.pause, args.samples)
    print("RESULT " + json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    main()
