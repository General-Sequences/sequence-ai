# Parts for a server you already run

Each part below is one file that goes into a policy server you already run, or next to a client you already use,
without changing the rest of it. A policy written as a class and served with `seq serve` has all but `replicas` on already, its robot sending the
pings: the [Quickstart](quickstart.md) starts there. These are for the code you keep.

## On the GPU

```python
from sequence_ai.engine import optimize
from sequence_ai.engine.batcher import Batcher

optimize(model_user, examples_user)                              # faster in place; kept only where outputs match
batcher_user = Batcher(infer_a_batch_user, sizes=[1, 2, 4, 8])   # several robots' requests in one forward pass
action_user = await batcher_user(observation_user)               # in each request's handler
```

| Part | What it does | Read |
|---|---|---|
| `optimize` | Makes a policy faster on its GPU: each speed-up kept only where its outputs on your observations stay the same, bit for bit unless you give a tolerance | [How do I make a policy faster without changing its outputs, and what does each switch change?](how-do-i-make-a-policy-faster-without-changing-its-outputs.md) |
| `batcher` | Several robots' requests in one forward pass, in the batch sizes the model lists | [Quickstart: batching and replicas](quickstart-inference.md), [How many robots can one GPU serve?](how-many-robots-can-one-gpu-serve.md) |
| `slack` | How long each request may wait, from the slack its robot's action chunk leaves | [`slack`](slack.md) |
| `replicas` | One process per replica, on the GPU you give it, each running one call at a time; a process that dies is replaced | [Should I run two replicas of a policy on one GPU?](should-i-run-two-replicas-of-a-policy-on-one-gpu.md) |
| `one_write`, `ping` | Each response in one write, and a ping to each robot every 100 ms, in a uvicorn or a WebSocket server | [What does each network component do?](components.md) |

[How do I use the inference components with openpi, LeRobot or GR00T?](how-do-i-use-the-inference-components-with-openpi-lerobot-or-gr00t.md)
says which of them applies to which framework.

## In a server of your own

Keep it, and give the robots a port of their own. `connect()` speaks the robot's stream: one WebSocket (`GET /stream`)
carrying each observation as it is made and each act after them, which `seq serve` answers. A route of your own that
takes a whole observation per request is not what `connect()` sends. Inside an ASGI app of your own, `App(model_user)`
from `sequence_ai.engine.serve` is the app `seq serve` runs: a function, an observation in and its output out, under an
ASGI server that runs WebSockets.

## In a client of your own

Keep that too, and take single components. Each is one file and imports no other component:

```python
import httpx
from sequence_ai.network import window

transport_user = httpx.HTTPTransport(socket_options=window.socket_options())    # BBR and TCP keepalive, where it can
client_user = httpx.Client(base_url="http://YOUR_GPU:8000", transport=transport_user)

warm_user = window.Warm(ping=lambda: client_user.get("/ping"))     # a ping while the connection is idle
warm_user.used()                                              # call it each time you send on the connection
```

Run against this package's server, those lines sent four pings in 0.45 seconds of idle time. One thing `connect()`
does that your own client must do itself: never let a ping and a request run at the same time. An HTTP/1.1 client
that starts a request while a ping is in flight opens a second connection for it, with a new handshake and a cold
window.

[`examples/components/`](../examples/components/) has a runnable file for each component, and
[Components](components.md) describes what each one does, what it needs and what was measured.
