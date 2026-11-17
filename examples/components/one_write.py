# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
one_write, alone: the two lines that make a uvicorn server send each response in one write.

    python examples/components/one_write.py         # needs uvicorn; serves briefly and responds to one request
"""
import threading
import time
import urllib.request

import uvicorn

from sequence_ai.network.one_write import WholeResponse, one_write_http


async def app(scope, receive, send):
    await send({"type": "http.response.start", "status": 200, "headers": [(b"content-length", b"2")]})
    await send({"type": "http.response.body", "body": b"ok"})


# WholeResponse around the app, one_write_http() as the protocol: nothing else changes
server = uvicorn.Server(uvicorn.Config(WholeResponse(app), host="127.0.0.1", port=0, log_level="error", lifespan="off",
                                       http=one_write_http()))
thread = threading.Thread(target=server.run, daemon=True)
thread.start()
while not server.started:
    time.sleep(0.01)
port = server.servers[0].sockets[0].getsockname()[1]
print("response:", urllib.request.urlopen(f"http://127.0.0.1:{port}/").read())
server.should_exit = True
thread.join(5)
