# How do I serve a robot policy from my own GPU?

Give your model to `sequence_ai.serve()` as a function, and call it from the robot with `connect()`. Everything here is
open source and runs on a GPU you already have: no account and no service of ours is involved.

On the GPU:

```python
import sequence_ai

def policy_user(observation_user):                    # your model: one observation in, what the robot needs out
    return {"actions": model_user(observation_user["image"], observation_user["state"])}

sequence_ai.serve(policy_user, port=8000, example=observation_user)
```

On the robot:

```python
import sequence_ai

link_user = sequence_ai.connect("http://YOUR_GPU:8000")
actions_user = link_user.act({"image": frame_user, "state": state_user})["actions"]
```

Both sides' defaults are on: the robot sends pictures as JPEG and compresses large arrays where that saves time, keeps
its one connection warm with pings, and waits longer for the first answer; the server runs the policy on a thread of
its own, answers pings at once, warms the policy up on `example`, and sends torch and JAX outputs back as numpy arrays.
[What is on by default, and how do I turn each part off?](what-is-on-by-default.md) lists every part with its switch,
and `python -m sequence_ai.check http://YOUR_GPU:8000` says which are on between your robot and your server.

[`quickstart_server.py`](../examples/quickstart_server.py) is this server in one short file, with a stand-in
`policy_user` to replace. The [Quickstart](quickstart.md) runs both sides on one machine first, with the output of each
step. Run between two addresses of one machine, the robot printed:

```text
request 0: 8 actions of 7 numbers, the first [128. 128. 128.]
request 1: 8 actions of 7 numbers, the first [128. 128. 128.]
request 2: 8 actions of 7 numbers, the first [128. 128. 128.]
3 requests, the slowest took 3.4 ms
```

## Your model, as a function

`serve()` takes any model as a function. It is called with the observation the robot sent, a dict with numpy arrays as
arrays, and returns what the robot gets back: a dict, an array or a list, with numpy arrays, torch tensors or JAX
arrays in it.

Return an action chunk: the steps the robot plays while it asks for the next one. A policy that returns one step at a
time, a network round trip away, leaves a 30 Hz arm waiting for each step: OpenVLA, which returns one step, held a
simulated arm still for all 1,200 steps of a 40 s run.

| Your model | What to pass |
|---|---|
| A function of one observation | `serve(policy)` |
| A function of a list of observations, one forward pass for all of them | `serve(policy_on_a_list, batch=[1, 2, 4, 8])` |
| openpi (π0, π0.5) | `serve(policy.infer)`, or batched: `serve(openpi.batched(policy), batch=[1, 2, 4, 8])` |
| A PyTorch module: LeRobot's, GR00T's, your own | a function that turns the observation into the module's inputs and calls it |

A PyTorch module, as [`serve_a_pytorch_model.py`](../examples/serve_a_pytorch_model.py) serves one:

```python
import torch

@torch.inference_mode()
def policy_user(observation_user):
    # 1x3xHxW
    image_user = torch.as_tensor(observation_user["image"], device="cuda").permute(2, 0, 1)[None].float() / 255
    state_user = torch.as_tensor(observation_user["state"], device="cuda")[None]
    return {"actions": model_user(image_user, state_user)[0]}            # a tensor: serve() sends it as a numpy array

sequence_ai.serve(policy_user, example=observation_user)
```

Run on a CPU, its robot printed:

```text
a ndarray of shape (8, 7) and dtype float32 from the model on the cpu: the first [ 0.0413 -0.3051  0.1673]
```

An openpi policy, batched:

```python
from openpi.policies import policy_config
from openpi.training import config as openpi_config

import sequence_ai
from sequence_ai.engine import openpi

policy_user = policy_config.create_trained_policy(openpi_config.get_config("pi05_droid"), checkpoint_dir_user)
sequence_ai.serve(openpi.batched(policy_user), batch=[1, 2, 4, 8], example=observation_user)
```

`openpi.batched(policy)` is openpi's own `infer()` for a list of observations: each passes through the policy's
transforms, and the model samples all of them in one call. [`openpi.py`](../sequence_ai/engine/openpi.py) has the
details.

**Batching.** With `batch=[1, 2, 4, 8]`, the observations that wait while the policy runs reach it together, in the
largest listed size they fill, and each robot gets its own output back. Five robots acting at the same moment against
[`components/serve.py`](../examples/components/serve.py) printed:

```text
5 robots, each with its own actions: the first of each [0.0, 1.0, 2.0, 3.0, 4.0]
the warm-up ran batches of [1, 1, 2, 2, 4, 4, 8, 8]; the robots' acts reached the model as batches of [4, 1]
```

The batches depend on when the requests arrive: on another run the same five came as `[2, 2, 1]`. List the sizes your
model is compiled or graphed for: a batch of an unlisted size is never sent to it. Batching serves several robots in
fewer forward passes; one robot's request is no faster for it.

## You already have a server

Keep it. `connect()` needs two things from a server: a POST that takes a msgpack body and responds with msgpack or
JSON, and any cheap GET for its ping. To get the rest of the defaults, a server also says what it restores in an
`X-Seq-Reads` header and restores it, and says it is a sequence-ai server in `X-Seq-Server`. In a FastAPI app:

```python
import asyncio

import numpy as np
import uvicorn
from fastapi import FastAPI, Request, Response

from sequence_ai.network import compressed, frames, msgpack_io
from sequence_ai.network.one_write import WholeResponse, one_write_http

app_user = FastAPI()
# so the robot sends JPEG, compressed
SAYS_user = {"X-Seq-Reads": "jpeg, zstd, zlib", "X-Seq-Server": "sequence-ai/0.63.0"}


def policy_user(observation_user: dict) -> np.ndarray:
    return np.full((8, 7), float(observation_user["image"].mean()), dtype=np.float32)      # your model here


@app_user.post("/act")
async def act_user(request_user: Request) -> Response:
    observation_user, _ = compressed.expand(msgpack_io.unpack(await request_user.body()), limit=64 << 20)
    observation_user = frames.unpack_pictures(observation_user)               # each picture sent as JPEG, as its array
    actions_user = await asyncio.to_thread(policy_user, observation_user)          # the model runs off the event loop
    return Response(msgpack_io.pack({"actions": actions_user}), media_type=msgpack_io.MEDIA_TYPE, headers=SAYS_user)


@app_user.get("/ping")
async def ping_user() -> Response:
    return Response(headers=SAYS_user)                                   # all the robot's ping needs: an empty response


if __name__ == "__main__":
    uvicorn.run(WholeResponse(app_user), host="0.0.0.0", port=8000, http=one_write_http())
```

This file was run with FastAPI 0.163 against the quickstart's robot: three requests answered, the second and third with
their picture sent as JPEG (the first left before the robot had heard what the server reads; `link.wait_until_ready()`
hears it first). `asyncio.to_thread` runs the model on a pool of threads, so two robots' requests can run it at once;
`serve()` runs it on one thread, one call at a time.

Your paths can have other names: `connect(url, act_path="/infer", ping_path="/healthz")`.

## You already have a client

Keep that too, and take single components. Each is one file and imports no other component:

```python
import httpx
from sequence_ai.network import window

transport_user = httpx.HTTPTransport(socket_options=window.socket_options())    # BBR and TCP keepalive, where it can
client_user = httpx.Client(base_url="http://YOUR_GPU:8000", transport=transport_user)

warm_user = window.Warm(ping=lambda: client_user.get("/ping"))     # a ping while the connection is idle
warm_user.used()                                              # call it each time you send on the connection
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
| A tunnel or proxy in front of the server | `one_write` | GET p50 189.7 ms with headers and body written apart, 106.7 ms in one write |
| Large raw arrays on a slow uplink | compression, on by default with `serve()` | 172,800-byte frames came out about 1.9 times smaller for 0.37 ms |
| A new connection for each call | keep one open, as `connect()` does | 305.9 ms per call with a new connection, 121.0 ms reusing one |

The conditions behind each row are in the [README](../README.md#results).
[How do I measure the link between my robot and my GPU?](how-do-i-measure-the-link-between-my-robot-and-my-gpu.md)
shows how to get the same numbers for your own link.

## Before a robot depends on it

- **A token.** `serve()` with no token runs the policy for anyone who can reach its port, and says so when it starts.
  Start it with `serve(policy, token="...")` (or set `SEQ_SERVE_TOKEN` on the GPU) and connect with
  `sequence_ai.connect(url, token="...")`; open the port only to your robot's address as well.
- **The first answer.** A model compiled on its first inference takes tens of seconds. `serve(example=...)` runs it
  before the server says it is ready, `link.wait_until_ready()` waits for that on the robot, and `connect()` gives the
  first act 120 s in any case.
- **Several robots on one GPU.** `serve(..., batch=[1, 2, 4, 8])`, above. One replica of a model per GPU:
  [How many robots can one GPU serve?](how-many-robots-can-one-gpu-serve.md)
- **No GPU near the robot.** [How do I get a GPU near my robot?](how-do-i-get-a-gpu-near-my-robot.md)

## See also

- [What is on by default, and how do I turn each part off?](what-is-on-by-default.md)
- [Quickstart](quickstart.md)
- [FAQ](faq.md), for the errors a first run can print
- [Comparison with openpi's and LeRobot's remote inference](comparison.md)
