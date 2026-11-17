# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
A robot's loop against examples/server.py, with the client-side components on its connection.

    python examples/server.py &
    python examples/robot.py

connect() applies them: BBR and TCP keepalive on the socket where this machine allows, a ping while the connection is
idle, msgpack both ways, each large array compressed while that saves time. examples/components/ has each component
alone, to add to your own client.
"""
from __future__ import annotations

import time

import numpy as np

from sequence_ai.network import connect


def camera(step: int) -> np.ndarray:
    """A camera-like frame: flat regions and edges, which compress as a real one does."""
    yy, xx = np.mgrid[0:480, 0:640]
    return np.stack([(xx // 8) * 9 + step, (yy // 6) * 7, ((xx + yy) // 16) * 11], -1).astype(np.uint8)


def main() -> None:
    with connect("http://127.0.0.1:8000", compress=True) as link:      # the server decompresses the arrays
        for step in range(5):
            response = link.act({"image": camera(step), "state": np.zeros(7, dtype=np.float32)})
            print(f"request {step}: got {response['actions'].shape} actions; "
                  f"the server reports this link as {response['link']}")
            time.sleep(0.5)                                            # the robot executes them: the connection is idle
        snap = link.stats()
    timing = snap["latency_us"]["act_us"]
    print(f"pings while idle: {snap['counters']['pings']}; requests: {timing['n']}, "
          f"p50 {timing['p50'] / 1000:.1f} ms, max {timing['max'] / 1000:.1f} ms")


if __name__ == "__main__":
    main()
