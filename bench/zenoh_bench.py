#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
Measures a Zenoh link: one request with a payload after each pause, between two Zenoh peers over TCP.

This is the transport that LeRobot 0.7's remote inference uses.

    python3 bench/zenoh_bench.py server PORT [--keep]
    python3 bench/zenoh_bench.py client HOST PORT [--keep] [--size 100000] [--pause 0.53] [--samples 40]

`--keep` opens the session with the file lerobot.keep_alive_file() writes: an idle link then carries a frame about every
100 ms. Pass it to both ends, as you would give the file to both LeRobot's server and its robot.

What runs: the server declares a queryable that replies with two bytes. The client connects to it, sends one 1 MB
request (which opens the window as a first large observation would), waits one second, then sends a request of `size`
bytes and waits for the reply, with `pause` seconds after each reply: 3 times uncounted, then `samples` times counted.
It prints one line, `RESULT {json}`: the median and the p95 of the counted requests, and every sample, in milliseconds.

These are two Zenoh peers, not LeRobot's policy server and robot: what is measured is the transport under them.

Needs Zenoh's Python package: pip install eclipse-zenoh.
"""
import argparse
import json
import os
import statistics
import tempfile
import time

WARMUP = 3                      # uncounted requests first: the window reaches its steady state for this pause
WIDEN = 1_000_000               # one request of this size first
KEY = "bench/echo"


def config(keep: bool, **keys: object):
    """A peer's configuration, with lerobot.keep_alive_file()'s settings when `keep` is set; scouting is off: one known
    peer."""
    import zenoh

    from sequence_ai.network import lerobot

    if keep:
        path = lerobot.keep_alive_file(os.path.join(tempfile.mkdtemp(), "zenoh-keep-alive.json5"))
        conf = zenoh.Config.from_file(path)
    else:
        conf = zenoh.Config()
    settings = {"mode": "peer", "scouting/multicast/enabled": False, "scouting/gossip/enabled": False, **keys}
    for key, value in settings.items():
        conf.insert_json5(key, json.dumps(value))
    return conf


def server(port: int, keep: bool) -> None:
    """Reply to every request on `port` until stopped."""
    import zenoh

    session = zenoh.open(config(keep, **{"listen/endpoints": [f"tcp/0.0.0.0:{port}"]}))

    def reply(query) -> None:
        query.reply(query.key_expr, b"ok")

    session.declare_queryable(KEY, reply)
    while True:
        time.sleep(1)


def client(host: str, port: int, keep: bool, size: int = 100_000, pause: float = 0.53, samples: int = 40) -> dict:
    """Send a request of `size` bytes `pause` seconds after each reply, and return the timings."""
    import zenoh

    session = zenoh.open(config(keep, **{"connect/endpoints": [f"tcp/{host}:{port}"]}))
    time.sleep(1.0)                                # time for the two peers to connect

    def request(payload: bytes) -> None:
        replies = list(session.get(KEY, payload=payload))
        if not replies or replies[0].ok is None:
            raise RuntimeError(f"no reply from tcp/{host}:{port}: is `zenoh_bench.py server {port}` running there?")

    payload = os.urandom(size)
    request(os.urandom(WIDEN))
    time.sleep(1.0)
    rows = []
    for i in range(WARMUP + samples):
        started = time.perf_counter()
        request(payload)
        if i >= WARMUP:
            rows.append(round((time.perf_counter() - started) * 1000, 2))
        time.sleep(pause)
    session.close()
    return {"bench": "zenoh", "variant": "keep_alive" if keep else "default", "port": port, "size": size,
            "pause_s": pause, "n": len(rows), "p50_ms": round(statistics.median(rows), 2),
            "p95_ms": round(sorted(rows)[int(0.95 * (len(rows) - 1))], 2), "samples_ms": rows}


def main(argv: "list[str] | None" = None) -> "dict | None":
    """The command line: `server PORT [--keep]` runs until stopped; `client ...` prints and returns its result."""
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    roles = parser.add_subparsers(dest="role", required=True)
    server_args = roles.add_parser("server")
    server_args.add_argument("port", type=int)
    server_args.add_argument("--keep", action="store_true", help="open with the keep-alive file")
    client_args = roles.add_parser("client")
    client_args.add_argument("host")
    client_args.add_argument("port", type=int)
    client_args.add_argument("--keep", action="store_true", help="open with the keep-alive file")
    client_args.add_argument("--size", type=int, default=100_000, help="bytes of each request")
    client_args.add_argument("--pause", type=float, default=0.53, help="seconds of silence after each reply")
    client_args.add_argument("--samples", type=int, default=40)
    args = parser.parse_args(argv)
    if args.role == "server":
        server(args.port, args.keep)
        return None
    result = client(args.host, args.port, args.keep, args.size, args.pause, args.samples)
    print("RESULT " + json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    main()
