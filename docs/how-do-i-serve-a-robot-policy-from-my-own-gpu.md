# How do I serve a robot policy from my own GPU?

Put your model behind a small HTTP server on the GPU and call it from the robot with `connect()`. Everything here is
open source and runs on a GPU you already have: no account and no service of ours is involved.

On the GPU:

```bash
pip install "sequence-ai[server,arrays]"
python quickstart_server.py 0.0.0.0 8000
```

On the robot:

```python
from sequence_ai.network import connect

link = connect("http://YOUR_GPU:8000")                    # BBR, TCP keepalive, a ping when idle, msgpack: all on
actions = link.act({"image": frame, "state": state})["actions"]
```

[`quickstart_server.py`](../examples/quickstart_server.py) is one short file with a function named `policy` that you
replace with your model. The [Quickstart](quickstart.md) runs both sides on one machine first, with the output of each
step. Run between two addresses of one machine, the robot printed:

```text
request 0: 8 actions of 7 numbers, the first [128. 128. 128.]
request 1: 8 actions of 7 numbers, the first [128. 128. 128.]
request 2: 8 actions of 7 numbers, the first [128. 128. 128.]
3 requests, the slowest took 4.8 ms
```

## You already have a server

Keep it. `connect()` needs two things from a server: a POST that takes a msgpack body and responds with msgpack or
JSON, and any cheap GET for its ping. In a FastAPI app that is two handlers, and two more names where uvicorn starts:

```python
import asyncio

import numpy as np
import uvicorn
from fastapi import FastAPI, Request, Response

from sequence_ai.network import compressed, msgpack_io
from sequence_ai.network.one_write import WholeResponse, one_write_http

app = FastAPI()


def policy(observation: dict) -> np.ndarray:
    return np.full((8, 7), float(observation["image"].mean()), dtype=np.float32)      # your model here


@app.post("/act")
async def act(request: Request) -> Response:
    observation, _ = compressed.expand(msgpack_io.unpack(await request.body()), limit=64 << 20)
    actions = await asyncio.to_thread(policy, observation)          # the model runs off the event loop
    return Response(msgpack_io.pack({"actions": actions}), media_type=msgpack_io.MEDIA_TYPE)


@app.get("/ping")
async def ping() -> Response:
    return Response()                                               # all the robot's ping needs: an empty response


if __name__ == "__main__":
    uvicorn.run(WholeResponse(app), host="127.0.0.1", port=8000, http=one_write_http())
```

This file was run with FastAPI 0.163 against the quickstart's robot: three requests answered, and with
`connect(url, compress=True)` three more, with three pings between them.

Your paths can have other names: `connect(url, act_path="/infer", ping_path="/healthz")`.

## You already have a client

Keep that too, and take single components. Each is one file and imports no other component:

```python
import httpx
from sequence_ai.network import window

transport = httpx.HTTPTransport(socket_options=window.socket_options())    # BBR and TCP keepalive, where it can
client = httpx.Client(base_url="http://YOUR_GPU:8000", transport=transport)

warm = window.Warm(ping=lambda: client.get("/ping"))     # a ping while the connection is idle
warm.used()                                              # call it each time you send on the connection
```

Run against the quickstart's server, those lines sent four pings in 0.45 seconds of idle time. One thing `connect()`
does that your own client must do itself: never let a ping and a request run at the same time. An HTTP/1.1 client
that starts a request while a ping is in flight opens a second connection for it, with a new handshake and a cold
window. `connect()` uses one connection and a lock; [`link.py`](../sequence_ai/network/link.py) is short enough to read.

[`examples/components/`](../examples/components/) has a runnable file for each component, and
[Components](components.md) describes what each one does, what it needs and what was measured.

## Which components help on your link

A component saves time only on a link that has the cost it removes. Measure before you add:

| On your link | Add | Measured |
|---|---|---|
| A short round-trip time (round-trip time (RTT), about 21 ms) and pauses between requests | `connect()`, or `window` on your own client | 40 of 40 requests took one extra round trip without BBR, none with it |
| A longer RTT (about 101 or 241 ms), no loss | nothing for the pauses | 0 or 1 of 40 requests took an extra round trip with the default settings |
| A tunnel or proxy in front of the server | `one_write` | GET p50 189.7 ms with headers and body written apart, 106.7 ms in one write; not measured on a direct connection |
| Large raw arrays on a slow uplink | `compress=True` and `compressed.expand` | 172,800-byte frames came out about 1.9 times smaller for 0.37 ms |
| A new connection for each call | keep one open, as `connect()` does | 305.9 ms per call with a new connection, 121.0 ms reusing one |

The conditions behind each row are in the [README](../README.md#results).
[How do I measure the link between my robot and my GPU?](how-do-i-measure-the-link-between-my-robot-and-my-gpu.md)
shows how to get the same numbers for your own link.

## Before a robot depends on it

- **Credentials.** The quickstart's server checks none. Open its port only to your robot's address, or check a header
  in your handler: `connect(url, headers={"authorization": "..."})` sends one with every request.
- **Timeout.** `connect()` waits 10 seconds for a response. A model that loads or compiles on its first call needs more:
  `connect(url, timeout=60)`.
- **Several robots on one GPU.** One request at a time is one forward pass each. The inference layer has `batcher`,
  which batches the waiting requests into one call to the model, and `replicas`, which runs one replica of the model per
  GPU: [How many robots can one GPU serve?](how-many-robots-can-one-gpu-serve.md)
- **No GPU near the robot.** [How do I get a GPU near my robot?](how-do-i-get-a-gpu-near-my-robot.md)

## See also

- [Quickstart](quickstart.md)
- [FAQ](faq.md), for the errors a first run can print
- [Comparison with openpi's and LeRobot's remote inference](comparison.md)
