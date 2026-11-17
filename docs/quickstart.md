# Quickstart: from `pip install` to the first action

Three commands and two short files give you a policy server, a robot that requests actions from it, and the first
action chunk on your screen. No GPU and no model are needed. The output shown is from macOS on Apple silicon with
Python 3.12; Python 3.9 or later on Linux or macOS behaves the same way.

**How long it takes.** The install took 10 seconds the first time and 5 seconds with the packages already downloaded.
The three requests took under a second.

## 1. Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install sequence-ai
```

The last line it prints (your versions may be newer):

```text
Successfully installed annotated-types-0.8.0 anyio-4.16.0 certifi-2026.10.23 click-8.5.1 h11-0.16.0 httpcore-1.0.9 httpx-0.28.1 idna-3.20 numpy-2.5.5 pillow-13.0.0 pydantic-2.14.3 pydantic-core-2.50.4 sequence-ai-0.62.0 sequence-base-0.8.12 typing-extensions-4.16.0 typing-inspection-0.4.5 uvicorn-0.54.0 zstandard-0.25.0
```

The package brings httpx, which `connect()` uses for its connection, Pillow, uvicorn, which runs the quick start's
server, numpy and zstandard for arrays, and what the `seq` command needs (pydantic and `sequence-base`).

## 2. Start a policy server

Save this as `quickstart_server.py`. It is [`examples/quickstart_server.py`](../examples/quickstart_server.py).

```python
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
```

```bash
python quickstart_server.py
```

It prints:

```text
INFO:     Started server process [73374]
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

## 3. Request actions

In a second terminal, with the same environment active, save this as `quickstart_robot.py`. It is
[`examples/quickstart_robot.py`](../examples/quickstart_robot.py).

```python
import sys

import numpy as np

import sequence_ai

url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
frame = np.full((224, 224, 3), 128, dtype=np.uint8)        # your camera's frame here
state = np.zeros(7, dtype=np.float32)                      # your robot's joints here

with sequence_ai.connect(url) as link:                     # BBR, TCP keepalive, a ping when idle, msgpack: all on
    for step in range(3):
        actions = link.act({"image": frame, "state": state})["actions"]
        print(f"request {step}: {actions.shape[0]} actions of {actions.shape[1]} numbers, the first {actions[0, :3]}")
    timing = link.stats()["latency_us"]["act_us"]
    print(f"{timing['n']} requests, the slowest took {timing['max'] / 1000:.1f} ms")
```

```bash
python quickstart_robot.py
```

It prints:

```text
request 0: 8 actions of 7 numbers, the first [128. 128. 128.]
request 1: 8 actions of 7 numbers, the first [128. 128. 128.]
request 2: 8 actions of 7 numbers, the first [128. 128. 128.]
3 requests, the slowest took 2.1 ms
```

That is the first action chunk: 8 actions of 7 numbers. The stand-in `policy` returns the mean of the frame it was
sent, and the frame is all 128. In the server's terminal, each request is one line:

```text
INFO:     127.0.0.1:51427 - "POST /act HTTP/1.1" 200 OK
INFO:     127.0.0.1:51427 - "POST /act HTTP/1.1" 200 OK
INFO:     127.0.0.1:51427 - "POST /act HTTP/1.1" 200 OK
```

The port after `127.0.0.1:` is the same on all three: the robot kept one connection open and reused it.

## What just ran

On the robot, `connect()` did five things that your code would otherwise have to do:

| What | Why | Component |
|---|---|---|
| Kept one connection open for every request | A new connection for each call costs a handshake each time | [`link`](../sequence_ai/network/link.py) |
| Requested BBR and enabled TCP keepalive, where the machine allows | Linux restarts a connection's congestion window after a pause; BBR keeps it | [`window`](../sequence_ai/network/window.py) |
| Sent a small GET whenever the connection was idle for 100 ms | So the connection is never idle between requests | [`window`](../sequence_ai/network/window.py) |
| Sent the frame as msgpack, its bytes as they are | JSON carries bytes as base64, a third larger on the wire | [`msgpack_io`](../sequence_ai/network/msgpack_io.py) |
| Timed every request | `link.stats()` gives p50, p95 and p99 once there are enough samples | [`latency_stats`](../sequence_ai/network/latency_stats.py) |

On the server, `WholeResponse` and `one_write_http()` put each response in one write
([`one_write`](../sequence_ai/network/one_write.py)), `compressed.expand` decompresses anything a client compressed, and
the `server-timing` header tells the robot how long the model itself took.

On this machine, a Mac, BBR is not available, so the connection got TCP keepalive only. On loopback none of
these components saves time: there is no network between the two programs. What each one saves on a real link is in
the README's [results](../README.md#results), and the links where it saves nothing in its
[limitations](../README.md#limitations).

## Put your own model and robot in

1. **Your model.** Replace the body of `policy` in `quickstart_server.py`. It takes the observation the robot sent and
   returns the actions.
2. **Your robot.** Replace `frame` and `state` in `quickstart_robot.py` with your camera's frame and your robot's
   joints, and send the keys your model reads.
3. **Two machines.** On the GPU, listen on every address; on the robot, give the GPU's address:

   ```bash
   python quickstart_server.py 0.0.0.0 8000              # on the GPU
   python quickstart_robot.py http://YOUR_GPU:8000       # on the robot
   ```

   Run that way between two addresses of one machine, the three requests printed the same lines and the slowest took
   4.8 ms. This server checks no credentials: open its port only to your robot.
4. **Measure the link.**
   [How do I measure the link between my robot and my GPU?](how-do-i-measure-the-link-between-my-robot-and-my-gpu.md)
   tells you which components help on it.

## Where to go next

- [How do I serve a robot policy from my own GPU?](how-do-i-serve-a-robot-policy-from-my-own-gpu.md)
- [How do I run π0 (openpi) on a remote GPU?](how-do-i-run-pi0-openpi-on-a-remote-gpu.md)
- [How do I reduce the latency of LeRobot remote inference?](how-do-i-reduce-the-latency-of-lerobot-remote-inference.md)
- [How do I get a GPU near my robot?](how-do-i-get-a-gpu-near-my-robot.md)
- A step failed: [FAQ](faq.md)
