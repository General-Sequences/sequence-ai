<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/General-Sequences/sequence-ai/main/docs/banner-dark.svg">
    <img alt="sequence-ai" src="https://raw.githubusercontent.com/General-Sequences/sequence-ai/main/docs/banner-light.svg" width="720">
  </picture>
</p>

<p align="center">
  <a href="https://github.com/General-Sequences/sequence-ai/actions/workflows/tests.yml"><img alt="tests" src="https://github.com/General-Sequences/sequence-ai/actions/workflows/tests.yml/badge.svg"></a>
  <img alt="pip install sequence-ai" src="https://img.shields.io/badge/pip%20install-sequence--ai-15775C?style=flat-square">
  <img alt="Python 3.9 or later" src="https://img.shields.io/badge/python-3.9%2B-15775C?style=flat-square">
  <a href="https://github.com/General-Sequences/sequence-ai/blob/main/LICENSE"><img alt="license: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-15775C?style=flat-square"></a>
</p>

`sequence-ai` is an open-source SDK for serving robot policies from remote GPUs over wide-area networks (WANs). It
supports all open-source robot policies. It runs on your own GPU, or on one rented from
[General Sequences](https://generalsequences.com) near your robot: on the robot, only what `connect()` is given changes.

Give it a policy server and a robot. On the GPU, it replays the module a model calls many times per inference from CUDA
graphs, with exactly the same outputs, and batches several robots' requests within the slack their action chunks
leave. On the connection, it keeps the TCP congestion window open across the robot's pauses, so a 24 KB upload after
a pause takes one round trip where it took two, at RTTs from 12 to 101 ms. Together these are GS-WAN (General Sequences
Wide-Area Network). Each component is one file that can be added to existing code.

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#three-ways-to-run-it">Three ways to run it</a> ·
  <a href="#how-it-fits-together">Architecture</a> ·
  <a href="#the-problem-by-layer">The problem</a> ·
  <a href="#inference-layer">Inference layer</a> ·
  <a href="#network-layer">Network layer</a> ·
  <a href="#results">Results</a> ·
  <a href="#limitations">Limitations</a> ·
  <a href="#documentation">Docs</a>
</p>

## Install

```bash
pip install sequence-ai                          # the robot's client, the seq command, every component
pip install "sequence-ai[server,arrays]"         # and uvicorn, numpy, zstandard: the quick start's server
```

Python 3.9 or later for the client and the components, on Linux and macOS; the `seq` command needs Python 3.10 or
later. `cuda_graphs` uses the torch your model already has (2.4 or later). Each component is one file, and most import
only the standard library, so copying a file into your project works as well.

## Quick start

On the robot, one call applies the client-side network components to one connection:

```python
import sequence_ai

with sequence_ai.connect("http://your-gpu:8000") as policy_user:     # BBR, TCP keepalive, a ping while idle, msgpack
    policy_user.observe({"image": frame_user, "state": state_user})
    actions_user = policy_user.act()["actions"]
```

On the GPU, the inference components wrap the model you already have:

```python
from sequence_ai.engine import cuda_graphs
from sequence_ai.engine.batcher import Batcher

cuda_graphs.install(model_user, ["action_head.denoiser"])        # replayed from CUDA graphs; kept only if outputs match
batcher_user = Batcher(infer_a_batch_user, sizes=[1, 2, 4, 8])   # several robots' requests in one forward pass
action_user = await batcher_user(observation_user)               # in each request's handler
```

Each name ending in `_user` is yours: your model, your frames, your handler. Both run on a laptop with no GPU and no
model:

```bash
python examples/quickstart_server.py & sleep 1 && python examples/quickstart_robot.py    # a server and a robot
python examples/quickstart_batching.py                                                   # five robots, batched
```

The first line each prints:

```text
request 0: 8 actions of 7 numbers, the first [128. 128. 128.]
5 robots served in 113 ms, as batches of {4: 1, 1: 1}
```

[Quickstart: a policy server and a robot](https://github.com/General-Sequences/sequence-ai/blob/main/docs/quickstart.md) and
[Quickstart: batching and replicas](https://github.com/General-Sequences/sequence-ai/blob/main/docs/quickstart-inference.md) walk through each with the output of every step.

## Three ways to run it

| | Whose GPU | What you run on it | On the robot |
|---|---|---|---|
| **Your own server** | Yours, or any GPU the robot can reach | Your policy server, with the components you add | `connect("http://your-gpu:8000")` |
| **A GPU near your robot** | One rented from General Sequences, by city | The same server | `connect("http://<its address>:8000")` |
| **A hosted deployment** | General Sequences' | Nothing: the platform builds your policy, serves it and scores it | `connect("my_policy_user")` |

The first needs no account and costs nothing. The other two are paid services of General Sequences, billed while a GPU
runs, and their commands are in this package:

```bash
seq gpu add --place "San Francisco, CA" --gpu L4 --port 8000    # a GPU near your robot, yours until seq gpu remove
seq deploy my_policy_user                                       # or a deployment: the policy in this folder
```

GPUs are in short supply, and few are near any given robot, which is why the second exists:
[How do I get a GPU near my robot?](https://github.com/General-Sequences/sequence-ai/blob/main/docs/how-do-i-get-a-gpu-near-my-robot.md) has its commands.
[The platform guide](https://github.com/General-Sequences/sequence-ai/blob/main/platform.md) has the third: writing a policy class, deploying it, benchmarks and evals,
teleoperation, and every `seq` command.

## How it fits together

```
            robot                                                GPU
 ┌─────────────────────────┐                        ┌───────────────────────────┐
 │  your control loop      │                        │  your policy              │
 ├─────────────────────────┤      observation       ├───────────────────────────┤
 │  sequence_ai.network    │  ───────────────────▶  │  sequence_ai.engine       │
 │    connect · openpi     │                        │    cuda_graphs · batcher  │
 │    window · compress    │          WAN           │    replicas · slack       │
 │    frames · wifi_power  │                        ├───────────────────────────┤
 │                         │  ◀───────────────────  │  sequence_ai.network      │
 │                         │      action chunk      │    one_write · ping       │
 └─────────────────────────┘                        │    tcp_link · compressed  │
                                                    └───────────────────────────┘

  both ends: msgpack_io · latency_stats             your own link, measured: bench/
```

**`sequence_ai.network`** is the connection between a robot and the GPU its policy runs on. On the robot, `connect()`
applies the client-side components to one connection, and `openpi.keep()` does the same on the connection that openpi's
own client holds. On the server, `one_write` and `ping` wrap a uvicorn or a WebSocket server. **`sequence_ai.engine`**
is the inference layer on the GPU: `cuda_graphs` shortens one inference, `batcher` and `slack` share a GPU among robots,
and `replicas` runs one process per GPU. **`bench/`** measures a link before anything is added to it.

## The problem, by layer

A robot policy built on a vision-language-action (VLA) model runs each inference on a GPU, and a robot without a
suitable GPU on board sends its observations over a WAN to a remote one. Each layer of that path has its own problem.

**The model is most of the latency.** Across thirteen model and network configurations, the model is 68 to 96% of a
request's latency. The time of one inference is therefore the main lever, and `cuda_graphs` shortens it without changing
the model's output.

**One robot leaves its GPU mostly idle.** One robot keeps a GPU busy 21 to 55% of the time, so several robots can share
one, provided that no robot's action chunk runs out while its request waits. `batcher` batches the requests that are
waiting, `slack` computes how long each may wait and how many robots a replica can serve, and `replicas` runs one
replica per GPU.

**A robot pauses, and Linux restarts its connection.** A robot executes a chunk between requests, and Linux restarts a
TCP connection's congestion window after an idle period. The upload after a pause then takes one extra round trip, at
RTTs from 12 to 101 ms. `window`, `ping` and `connect` keep the window open.

**A request carries more round trips and bytes than it needs.** A response split across writes can wait a round trip
behind a tunnel, and a camera frame is 50 to 100 KB. `one_write` sends each response in one write, `compress` compresses
an observation only when that shortens the upload, and `frames` resizes a frame before it is sent.

**GPUs are in short supply.** Few are near any given robot, and every millisecond of round-trip time comes out of the
slack an action chunk leaves. Getting a GPU near the robot is the routing layer: a hosted service of General
Sequences, reached with this package's `seq gpu` commands. [Three ways to run it](#three-ways-to-run-it) has them.

## Inference layer

| Component | What it does | To add it | Requires |
|---|---|---|---|
| [`cuda_graphs`](https://github.com/General-Sequences/sequence-ai/blob/main/sequence_ai/engine/cuda_graphs.py) | Replays a torch module that your model calls many times per run from CUDA graphs captured from its own eager kernels. A graph is kept only when its output is exactly the same. | one call | torch 2.4 or later, a CUDA GPU |
| [`batcher`](https://github.com/General-Sequences/sequence-ai/blob/main/sequence_ai/engine/batcher.py) | Batches the calls waiting for a model into one call, in the batch sizes the model lists. | one object, one `await` | nothing |
| [`replicas`](https://github.com/General-Sequences/sequence-ai/blob/main/sequence_ai/engine/replicas.py) | Runs replicas of a model, each in its own process on the GPU assigned to it, and replaces one that dies. | one object, one `await` | nothing |
| [`slack`](https://github.com/General-Sequences/sequence-ai/blob/main/sequence_ai/engine/slack.py) | Schedules by the slack an action chunk leaves: how many robots a replica can serve, each by its own round-trip time (RTT), how long a request may wait for a batch, and which requests are dispatched first. | a few calls in your server's loop | nothing |

```python
from sequence_ai.engine import cuda_graphs, slack
from sequence_ai.engine.batcher import Batcher
from sequence_ai.engine.replicas import ReplicaPool

cuda_graphs.install(model_user, ["action_head.denoiser"])         # model_user.action_head.denoiser, graphed in place
batcher_user = Batcher(model_on_a_list_user, sizes=[1, 2, 4, 8], wait_ms=2.0)
pool_user = ReplicaPool(load_user, replicas=2, devices=[0, 1])     # one replica per GPU

table_user = slack.measure(forward_of_user, sizes=[1, 2, 4])       # forward_of_user(n): one forward pass of n requests
play_user = slack.playback(hz=20, valid_len=15, replan_at=8)       # 15 valid steps, the next requested at the 8th
seating_user = slack.Seating(table_user, play_user["buffer_s"] * 1000, sizes=[1, 2, 4],
                             cycle_ms=(play_user["play_s"] - play_user["buffer_s"]) * 1000)
seats_user, over_user = seating_user.seat([("near", 30.0), ("far", 200.0)])     # (robot, its RTT in ms)
```

- **`cuda_graphs`** counts the calls each module gets over a few runs (`Census`), captures the module that is called
  many times per run with the same shapes, and checks each new shape against eager mode byte for byte. A shape whose
  capture fails or differs runs in eager mode from then on. The module stays the same object of the same class, with its
  weights, `state_dict` and hooks unchanged.
  [How do I make a PyTorch robot policy faster without changing its outputs?](https://github.com/General-Sequences/sequence-ai/blob/main/docs/how-do-i-make-a-pytorch-robot-policy-faster-without-changing-its-outputs.md)
- **`batcher`** dispatches a batch as soon as it is full, or `wait_ms` after its first call while the model is free.
  While the model is busy, waiting calls accumulate and are dispatched together. Only the listed sizes are used, and a
  batch in which the model raised is re-run call by call.
  [How many robots can one GPU serve?](https://github.com/General-Sequences/sequence-ai/blob/main/docs/how-many-robots-can-one-gpu-serve.md)
- **`replicas`** runs each replica in its own process on its own GPU, so a forward pass never blocks the server's
  event loop. `ReplicaError` is the model's own error, and the replica takes the next call. `ReplicaDied` is a process
  that has exited, and it is replaced.
  [Should I run two replicas of a policy on one GPU?](https://github.com/General-Sequences/sequence-ai/blob/main/docs/should-i-run-two-replicas-of-a-policy-on-one-gpu.md)
- **`slack`** computes a replica's capacity from the playback time an action chunk leaves when its robot requests
  the next one, each robot by its own RTT, and tells a server how long the open batch may wait.
  [`slack`: scheduling by the slack an action chunk leaves](https://github.com/General-Sequences/sequence-ai/blob/main/docs/slack.md)

[How do I use the inference components with openpi, LeRobot or GR00T?](https://github.com/General-Sequences/sequence-ai/blob/main/docs/how-do-i-use-the-inference-components-with-openpi-lerobot-or-gr00t.md)

## Network layer

"Client" is the robot's program and "server" is the policy server on the GPU.

| Component | Side | What it does | Requires |
|---|---|---|---|
| [`window`](https://github.com/General-Sequences/sequence-ai/blob/main/sequence_ai/network/window.py) | client | Keeps the connection's congestion window across the robot's pauses: BBR on its socket, TCP keepalive, a ping while the connection is idle. | nothing |
| [`compress`](https://github.com/General-Sequences/sequence-ai/blob/main/sequence_ai/network/compress.py) | client | Compresses an observation's large arrays only where that saves more upload time than it costs. | nothing; zstd where it is installed |
| [`frames`](https://github.com/General-Sequences/sequence-ai/blob/main/sequence_ai/network/frames.py) | client | Resizes a camera frame to the policy's input size before upload. | Pillow |
| [`wifi_power`](https://github.com/General-Sequences/sequence-ai/blob/main/sequence_ai/network/wifi_power.py) | client | Detects Wi-Fi power saving on the robot's link and reports how to disable it. | Linux |
| [`one_write`](https://github.com/General-Sequences/sequence-ai/blob/main/sequence_ai/network/one_write.py) | server | Sends each HTTP response of a uvicorn server in one write instead of three. | uvicorn |
| [`tcp_link`](https://github.com/General-Sequences/sequence-ai/blob/main/sequence_ai/network/tcp_link.py) | server | Reads each client's TCP state from the server's kernel: RTT, retransmitted segments, congestion window. | Linux |
| [`compressed`](https://github.com/General-Sequences/sequence-ai/blob/main/sequence_ai/network/compressed.py) | server | Decompresses what the client compressed, byte for byte. | numpy; zstandard for zstd |
| [`ping`](https://github.com/General-Sequences/sequence-ai/blob/main/sequence_ai/network/ping.py) | server | A server-side ping, so that the robot's connection is never idle: the options to pass to your server. | a WebSocket server |
| [`msgpack_io`](https://github.com/General-Sequences/sequence-ai/blob/main/sequence_ai/network/msgpack_io.py) | both | Encodes observations and responses as msgpack, with numpy arrays as raw bytes. Also reads openpi's layout. | nothing |
| [`latency_stats`](https://github.com/General-Sequences/sequence-ai/blob/main/sequence_ai/network/latency_stats.py) | both | Counters, gauges and latencies with their p50, p95 and p99, read as one snapshot. | nothing |

Combined, and for clients you may already use:

| | Side | What it does | Requires |
|---|---|---|---|
| [`connect`](https://github.com/General-Sequences/sequence-ai/blob/main/sequence_ai/network/link.py) | client | The client-side components on one connection to your own server. | httpx |
| [`openpi`](https://github.com/General-Sequences/sequence-ai/blob/main/sequence_ai/network/openpi.py) | client | The client-side components on the connection held by openpi's own client. | openpi-client |
| [`lerobot`](https://github.com/General-Sequences/sequence-ai/blob/main/sequence_ai/network/lerobot.py) | both | The Zenoh setting that keeps LeRobot's remote-inference connection from going idle. | lerobot 0.7 |
| [`bench/`](https://github.com/General-Sequences/sequence-ai/tree/main/bench/) | both | Measures your own link: what a pause, a ping and a camera stream cost on it. | nothing |

```python
from sequence_ai.network import lerobot, openpi

kept_user = openpi.keep(policy_user)                              # openpi: one line on its WebsocketClientPolicy
path_user = lerobot.keep_alive_file("zenoh-keep-alive.json5")     # LeRobot 0.7: one Zenoh setting, given to both sides
```

[What does each component do?](https://github.com/General-Sequences/sequence-ai/blob/main/docs/components.md) gives the lines to add, the reason and the measurement for each
one, and [`examples/components/`](https://github.com/General-Sequences/sequence-ai/tree/main/examples/components/) has a runnable file for each.
[How do I run π0 (openpi) on a remote GPU?](https://github.com/General-Sequences/sequence-ai/blob/main/docs/how-do-i-run-pi0-openpi-on-a-remote-gpu.md) |
[How do I reduce the latency of LeRobot remote inference?](https://github.com/General-Sequences/sequence-ai/blob/main/docs/how-do-i-reduce-the-latency-of-lerobot-remote-inference.md)
| [How do I serve a robot policy from my own GPU?](https://github.com/General-Sequences/sequence-ai/blob/main/docs/how-do-i-serve-a-robot-policy-from-my-own-gpu.md)

## Results

Across thirteen model and network configurations, the time a request spends outside the model ranges from 13 to
137 ms, and its share of the latency from 4 to 32%. A remote GPU is therefore close to an on-board GPU of the same
type, and inference time is the main lever.

Inference layer:

| Component | Measured on | Without | With |
|---|---|---|---|
| `cuda_graphs` | The diffusion transformer of GR00T N1.7 on one L4; outputs compared on inputs it was not captured on | 224 ms per run | 146 ms per run (1.54×), exactly the same outputs |
| `cuda_graphs` and `torch.compile` | The same model and GPU | `torch.compile(backend="cudagraphs")`: 225 ms (1.00×). inductor: 1.45×, outputs 0.0051 apart on average | 1.54×, exactly the same outputs |
| `cuda_graphs` | A small flow-style model of many small kernels on one T4 ([`examples/check_on_a_gpu.py`](https://github.com/General-Sequences/sequence-ai/blob/main/examples/check_on_a_gpu.py)) | 16.1 ms per run | 2.4 ms per run, byte for byte |
| Batching | π0.5 on one L4, the model's own batched call | 5.99 requests per second at batch 1 | 8.21, 9.38, 10.52 and 10.66 at batches of 2, 4, 8 and 16 (1.78×) |
| `slack`: the load a replica takes | Four robots, 300 s and about 2,100 requests per run, a stand-in model paced like a 15 Hz policy, in the serving stack the component was taken from | All four on one replica, a load of 0.95: 85.7% of responses within the 467 ms buffer | Two per replica on two replicas, a load of 0.88: 94.1% |

Network layer:

| Component | Measured on | Without | With |
|---|---|---|---|
| `window`, `ping`: each of four ways | A 24 KB upload 0.53 s after each response; two cloud machines, RTTs of 12, 21, 41, 65, 81 and 101 ms; 40 uploads per cell | 2.0 round trips, 40 of 40 uploads | 1.0 round trips, 0 of 40: the server's ping, the robot's ping, BBR on the robot's socket, or `tcp_slow_start_after_idle=0` |
| `window`: BBR | A robot sending a request every 0.53 s with 24 KB of frames; RTT about 21 ms; 40 requests per setting | 40 of 40 requests took one extra round trip (21.5 ms) | 0 of 40 (0.5 to 3.0 ms) |
| `window`: BBR, with 1% loss | The 24 KB upload at 22 ms with 1% loss on the link | 40 of 40 uploads took one extra round trip | 0 of 40 |
| `openpi`, and `ping` over a WebSocket | A 100 KB upload 0.53 s after each response; an openpi-style WebSocket client; RTTs of 11, 21, 41 and 81 ms; 40 uploads per cell | 37.6, 44.4, 82.4 and 163.8 ms (3.3, 2.1, 2.0 and 2.0 round trips) | 12.7, 22.7, 41.8 and 82.6 ms with `openpi.keep()` (1.1, 1.1, 1.0 and 1.0 round trips); 1.2, 1.1, 1.0 and 1.0 round trips with the server pinging |
| `lerobot` | A 100 KB request 0.53 s after each response between two Zenoh peers; the same machines and RTTs | 38.1, 44.8, 82.8 and 164.3 ms (3.4, 2.1, 2.0 and 2.0 round trips) | 13.9, 23.2, 42.3 and 82.2 ms with the keep-alive file on both ends (1.2, 1.1, 1.0 and 1.0 round trips) |
| `ping`, the server's ping | 100 KB, 500 KB and 1 MB after 500 ms of silence; RTTs of 11.7 and 69.2 ms; 20 per cell | 3.0 round trips each | 1.1, 1.1 and 1.25 |
| `window.Warm`, the robot's ping | 162 KB uploads 3 s apart; RTT 226 ms; 30 requests per setting | 303.6 ms for the upload to arrive | under 1 ms |
| `one_write` | A GET behind one kind of tunnel; TCP connect p50 89.5 ms | 189.7 ms (headers, then body) | 106.7 ms (one write) |
| `compress` | Camera frames of 172,800 bytes, decoded from a public dataset's video, on a laptop | 172,800 bytes | about 1.9 times smaller, for 0.37 ms to compress and 0.21 ms to decompress |

[Measurements](https://github.com/General-Sequences/sequence-ai/blob/main/docs/measurements.md) lists every command, and [`bench/`](https://github.com/General-Sequences/sequence-ai/tree/main/bench/) measures your own link.

## Limitations

- **`cuda_graphs` is for PyTorch models bound by kernel launches.** π0.5 under JAX already fuses and graphs its
  kernels and was 98% GPU-busy: there is no launch overhead to remove. A module called once per run, or with shapes
  that change on every call, gains nothing. Only GR00T N1.7's diffusion transformer and a small test model have been
  captured.
- **Batching does not amortize a model's prefix.** On π0.5 the prefix (images and prompt) took 138 ms per request at
  batch 1 and 119 ms at batch 16, so the gain levels off near 1.78×. Called as a server calls it, each observation
  through the policy's input and output transforms, a batch of 8 gave 1.52× (5.1 to 7.8 requests per second). On
  that model and an L4, batching added no capacity: one replica served 2 robots of a 15 Hz policy in time, and so
  did openpi's own server, one request at a time.
  [How many robots can one GPU serve?](https://github.com/General-Sequences/sequence-ai/blob/main/docs/how-many-robots-can-one-gpu-serve.md#checked-on-a-real-model)
- **Two replicas on one GPU give no throughput gain.** With π0.5 on an L4, two replicas whose calls overlapped each
  took 2.1 times as long as one alone. With the CUDA Multi-Process Service (MPS) they were 6 to 9% faster than that.
  Use one replica per GPU.
- **`batcher` and `replicas` have unit tests and no measurement of their own.** The batching result measures the
  model's own batched call. The `slack` results come from the serving stack the file was taken from, and from runs
  of the file itself around the batched calls of π0.5, GR00T N1.6 and SmolVLA on an L4, in a test server with
  simulated robots. No inference component has been run inside openpi's own policy server or LeRobot's.
- **`slack` computes a capacity on a simulated clock, not on your model under load.** It runs the robots as its loop
  would serve them, from up to 128 starts, and a number of robots is within capacity when 95% of their responses are
  in time. From three models' batch times on an L4 it gives 2 robots of a 15 Hz policy for π0.5 and GR00T N1.6,
  which 2 robots confirmed, and 1 for SmolVLA, where a formula for the load had given 8 and 4 robots had 95% of
  their responses late. Rerun on SmolVLA, 3 robots had late responses and 2 had none: the 1 comes from the limit on
  the load, which may be low for that model.
  [Why the capacity is computed on a clock](https://github.com/General-Sequences/sequence-ai/blob/main/docs/how-many-robots-can-one-gpu-serve.md#why-the-capacity-is-computed-on-a-clock)
  has the measurement, and [`examples/fleet_on_a_clock.py`](https://github.com/General-Sequences/sequence-ai/blob/main/examples/fleet_on_a_clock.py) prints what the clock runs
  for your batch times.
- **A policy that leaves no slack is not batched by `slack`.** Where the robot sends a request only when a chunk is
  fully executed, a replica serves one robot.
- **The idle restart costs nothing on a long link.** At an RTT of 242 ms the 24 KB upload took 1.0 round trips with
  no component, and at about 101 and 241 ms a robot on a fixed 0.53 s period took 0 or 1 extra round trips in 40.
  Whether a robot is affected depends on its RTT, its pause and its upload size: [`bench/`](https://github.com/General-Sequences/sequence-ai/tree/main/bench/) measures it.
- **A ping does not restore a window that losses cut.** With 1% loss at RTTs of 22 and 101 ms, 31 and 35 of 40
  uploads still took an extra round trip with the robot's ping. BBR removed it at 22 ms.
- **On one home connection the ping helped only small uploads.** From a Linux client, a 100 KB upload after a pause went
  from 5.5 round trips to 1.7 with the server's ping. 500 KB and 1 MB took 7.8 and 12.7 round trips with it and 10.3 and
  13.4 without: the line's upload rate and packet loss were the cost, and BBR took 5.1 and 8.3.
  [The table](https://github.com/General-Sequences/sequence-ai/blob/main/docs/measurements.md#from-a-linux-client-on-a-home-connection) has every cell.
- **`openpi.keep()` and the LeRobot setting were measured mostly with stand-ins.** The WebSocket and Zenoh measurements
  used the `websockets` library and two Zenoh peers. `openpi.keep()` was also run once with openpi's own server and
  client, on one link, two episodes per setting; the LeRobot setting was not run with LeRobot's own server and robot.
- **`one_write` was measured behind one kind of tunnel**, not on a direct connection.

[Measurements](https://github.com/General-Sequences/sequence-ai/blob/main/docs/measurements.md#limitations) has the rest.

## Repository layout

```
sequence_ai/
  engine/        inference layer: cuda_graphs · batcher · replicas · slack
  network/       network layer
                   client: connect · window · compress · frames · wifi_power · openpi
                   server: one_write · ping · tcp_link · compressed
                   both:   msgpack_io · latency_stats · lerobot
  ...            the client for General Sequences' GPUs and deployments: connect, gpu, teleop, record
seq/             the seq command, and the decorators a hosted policy or benchmark is written with
bench/           measures a link: a TCP client and server, WebSocket, Zenoh
examples/        the quick starts, and one runnable file per component
docs/            one page per question, the measurements, the FAQ
tests/           the components' tests: Python 3.9 to 3.12, on a machine with no GPU
tools/           writes llms-full.txt
platform.md      the guide to the hosted platform; eval.md and observability.md are its references
```

About 4,500 lines in `sequence_ai/engine` and `sequence_ai/network`, and 202 test functions.

## Documentation

- [Quickstart: a policy server and a robot](https://github.com/General-Sequences/sequence-ai/blob/main/docs/quickstart.md)
- [Quickstart: batching and replicas](https://github.com/General-Sequences/sequence-ai/blob/main/docs/quickstart-inference.md)
- Inference layer
  - [How do I make a PyTorch robot policy faster without changing its outputs?](https://github.com/General-Sequences/sequence-ai/blob/main/docs/how-do-i-make-a-pytorch-robot-policy-faster-without-changing-its-outputs.md)
  - [How many robots can one GPU serve?](https://github.com/General-Sequences/sequence-ai/blob/main/docs/how-many-robots-can-one-gpu-serve.md)
  - [Should I run two replicas of a policy on one GPU?](https://github.com/General-Sequences/sequence-ai/blob/main/docs/should-i-run-two-replicas-of-a-policy-on-one-gpu.md)
  - [How do I use the inference components with openpi, LeRobot or GR00T?](https://github.com/General-Sequences/sequence-ai/blob/main/docs/how-do-i-use-the-inference-components-with-openpi-lerobot-or-gr00t.md)
  - [`slack`: scheduling by the slack an action chunk leaves](https://github.com/General-Sequences/sequence-ai/blob/main/docs/slack.md)
- Network layer
  - [How do I serve a robot policy from my own GPU?](https://github.com/General-Sequences/sequence-ai/blob/main/docs/how-do-i-serve-a-robot-policy-from-my-own-gpu.md)
  - [How do I run π0 (openpi) on a remote GPU?](https://github.com/General-Sequences/sequence-ai/blob/main/docs/how-do-i-run-pi0-openpi-on-a-remote-gpu.md)
  - [How do I reduce the latency of LeRobot remote inference?](https://github.com/General-Sequences/sequence-ai/blob/main/docs/how-do-i-reduce-the-latency-of-lerobot-remote-inference.md)
  - [Why does my robot's upload get slow after a pause?](https://github.com/General-Sequences/sequence-ai/blob/main/docs/why-does-my-robots-upload-get-slow-after-a-pause.md)
  - [How do I measure the link between my robot and my GPU?](https://github.com/General-Sequences/sequence-ai/blob/main/docs/how-do-i-measure-the-link-between-my-robot-and-my-gpu.md)
  - [How do I keep a WebSocket policy server's connection warm?](https://github.com/General-Sequences/sequence-ai/blob/main/docs/how-do-i-keep-a-websocket-policy-servers-connection-warm.md)
  - [What do the network components save, measured with the tools in this repository?](https://github.com/General-Sequences/sequence-ai/blob/main/docs/measurements.md)
  - [How do the components compare with openpi's remote inference and LeRobot's?](https://github.com/General-Sequences/sequence-ai/blob/main/docs/comparison.md)
  - [What does each network component do?](https://github.com/General-Sequences/sequence-ai/blob/main/docs/components.md)
- General Sequences' GPUs
  - [How do I get a GPU near my robot?](https://github.com/General-Sequences/sequence-ai/blob/main/docs/how-do-i-get-a-gpu-near-my-robot.md)
  - [The platform guide](https://github.com/General-Sequences/sequence-ai/blob/main/platform.md), with its references for [benchmarks and evals](https://github.com/General-Sequences/sequence-ai/blob/main/eval.md) and
    [debugging a deployment](https://github.com/General-Sequences/sequence-ai/blob/main/observability.md)
  - [generalsequences.com/docs](https://generalsequences.com/docs/)
- [FAQ: errors you may see, and what to do next](https://github.com/General-Sequences/sequence-ai/blob/main/docs/faq.md)

[`llms.txt`](https://github.com/General-Sequences/sequence-ai/blob/main/llms.txt) is a map of the repository for a coding assistant, and [`llms-full.txt`](https://github.com/General-Sequences/sequence-ai/blob/main/llms-full.txt) is
this README and every page of `docs/` in one file.

## A note on the numbers

Every number in this README is given with the model, the GPU or the link it was measured on, and
[Measurements](https://github.com/General-Sequences/sequence-ai/blob/main/docs/measurements.md) lists the command behind each one. What a network component saves depends on the
link, so [`bench/`](https://github.com/General-Sequences/sequence-ai/tree/main/bench/) measures yours before you add anything. Each module's docstring states what the component
does in one sentence, why, and what was measured.

## Develop

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"
.venv/bin/python -m pytest -q
.venv/bin/ruff check sequence_ai/engine sequence_ai/network tests examples bench tools
```

The tests are the components', and run on Python 3.9 to 3.12 on a machine with no GPU. The client for the hosted
platform and the `seq` command are developed in General Sequences' own repository and published here, as they ship, at
each release. [CONTRIBUTING.md](https://github.com/General-Sequences/sequence-ai/blob/main/CONTRIBUTING.md) has what that means for a pull request.

## Contributing

Issues and pull requests are welcome: see [CONTRIBUTING.md](https://github.com/General-Sequences/sequence-ai/blob/main/CONTRIBUTING.md) and the
[Code of Conduct](https://github.com/General-Sequences/sequence-ai/blob/main/CODE_OF_CONDUCT.md). To report a security problem, see [SECURITY.md](https://github.com/General-Sequences/sequence-ai/blob/main/SECURITY.md).

## Cite

If you use `sequence-ai` in research, please cite it:

```bibtex
@software{generalsequences2026sequenceai,
  title  = {sequence-ai},
  author = {{General Sequences}},
  year   = {2026},
  url    = {https://github.com/General-Sequences/sequence-ai}
}
```

[CITATION.cff](https://github.com/General-Sequences/sequence-ai/blob/main/CITATION.cff) has the same, with the version.

## License

[Apache-2.0](https://github.com/General-Sequences/sequence-ai/blob/main/LICENSE). Copyright 2026 General Sequences Company. One component is adapted from another project, and the
page `seq eval replay` serves carries others' files: [THIRD_PARTY_NOTICES.md](https://github.com/General-Sequences/sequence-ai/blob/main/THIRD_PARTY_NOTICES.md).
