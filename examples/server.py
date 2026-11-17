# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
Your own policy server with the server-side components applied: its responses are sent in one write, what the robot
compressed is decompressed, and each response reports what the server's kernel sees of its robot's link.

    python examples/server.py            # then, in another terminal: python examples/robot.py

The server is a plain ASGI app under uvicorn; the components wrap a FastAPI or Starlette app the same way. The "policy"
here returns the mean of the frame it was sent: put your model's call in its place.
"""
from __future__ import annotations

import numpy as np
import uvicorn

from sequence_ai.network import compressed, msgpack_io, tcp_link
from sequence_ai.network.one_write import WholeResponse, one_write_http


async def body_of(receive) -> bytes:
    parts = []
    while True:
        message = await receive()
        parts.append(message.get("body", b""))
        if not message.get("more_body"):
            return b"".join(parts)


async def app(scope, receive, send) -> None:
    if scope["path"] == "/ping":
        status, response = 200, b""
    else:
        # msgpack in, arrays as arrays; then every item the robot compressed is decompressed, byte for byte
        observation, _grew = compressed.expand(msgpack_io.unpack(await body_of(receive)), limit=64 << 20)
        actions = np.full((8, 7), float(observation["image"].mean()), dtype=np.float32)      # your policy, here
        # this request's own connection, read from the kernel: None where the kernel provides none (not Linux)
        status, response = 200, msgpack_io.pack({"actions": actions, "link": tcp_link.of(scope["client"])})
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", msgpack_io.MEDIA_TYPE.encode()),
                            (b"content-length", str(len(response)).encode())]})
    await send({"type": "http.response.body", "body": response})


if __name__ == "__main__":
    # what sends each response in one write: WholeResponse around the app, one_write_http() as the protocol
    uvicorn.run(WholeResponse(app), host="127.0.0.1", port=8000, lifespan="off", http=one_write_http())
