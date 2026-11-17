# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
Quickstart, server side: a policy server in one file. Put your model where `policy` is.

    pip install "sequence-ai[server,arrays]"
    python quickstart_server.py                  # on this machine: 127.0.0.1:8000
    python quickstart_server.py 0.0.0.0 8000     # on your GPU, for a robot elsewhere

It responds to POST /act with a chunk of actions and to any other request with an empty response, which is all a ping
needs. docs/quickstart.md walks through it with examples/quickstart_robot.py.
"""
import sys
import time

import numpy as np
import uvicorn

from sequence_ai.network import compressed, msgpack_io
from sequence_ai.network.one_write import WholeResponse, one_write_http


def policy(observation: dict) -> np.ndarray:
    """Your model here: an observation in, a chunk of actions out. This one returns 8 actions of 7 numbers."""
    return np.full((8, 7), float(observation["image"].mean()), dtype=np.float32)


async def app(scope, receive, send) -> None:
    body = b""
    while True:
        message = await receive()
        body += message.get("body", b"")
        if not message.get("more_body"):
            break
    response, took_ms = b"", 0.0
    if scope["path"] == "/act":
        # msgpack in, arrays as arrays; whatever the robot compressed is decompressed, byte for byte
        observation, _ = compressed.expand(msgpack_io.unpack(body), limit=64 << 20)
        started = time.perf_counter()
        actions = policy(observation)
        took_ms = (time.perf_counter() - started) * 1000
        response = msgpack_io.pack({"actions": actions})
    await send({"type": "http.response.start", "status": 200,
                "headers": [(b"content-type", msgpack_io.MEDIA_TYPE.encode()),
                            (b"content-length", str(len(response)).encode()),
                            (b"server-timing", f"act;dur={took_ms:.1f}".encode())]})
    await send({"type": "http.response.body", "body": response})


if __name__ == "__main__":
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    # what sends each response in one write: WholeResponse around the app, one_write_http() as the protocol
    uvicorn.run(WholeResponse(app), host=host, port=port, lifespan="off", http=one_write_http())
