# How do the components compare with openpi's remote inference and LeRobot's?

It does not replace either one. openpi and LeRobot each ship a complete way to run a policy on a remote GPU: a server
that loads the model, and a client on the robot that talks to it. The network components here load no model. The network layer is a set
of components for the connection underneath, and you use it in one of three ways:

| You run | You keep | You add from `sequence_ai.network` |
|---|---|---|
| openpi | openpi's server and client, unchanged | on the robot, `openpi.keep(policy)`, and `image_tools` from `sequence_ai.network.openpi` for pictures as JPEG; on the GPU, `openpi.patch()` in its server script |
| LeRobot's remote inference | LeRobot's server and client, unchanged | one Zenoh setting, in a file: `lerobot.keep_alive_file(path)` |
| your own server | your model and your handler | `window` and `frames` on the robot; `one_write`, `ping`, `msgpack_io`, `tcp_link` on the server |

This page compares what each one does and does not do, from source inspection: openpi at commit `215abfb`
(2026-08-24), LeRobot 0.7.0, and this package's components. Results from tests are marked as such. No head-to-head measurement
against openpi or LeRobot has been made.

## Side by side

| | openpi | LeRobot 0.7.0 | These components |
|---|---|---|---|
| What it is | A policy server and a client for openpi's models | A policy server and a remote executor for LeRobot's policies | Components for the connection; no model loading |
| Starts with | `uv run scripts/serve_policy.py --env=...` and `WebsocketClientPolicy(host, port)` | `lerobot-policy-server` and `lerobot-rollout --inference.type=remote` | `serve(policy)` on the GPU, `sequence_ai.connect(url)` on the robot |
| Transport | WebSocket; the client opens one connection and reuses it | Zenoh, over TCP by default (`tcp/127.0.0.1:7447`) | The robot's stream: one WebSocket, kept open; options for WebSocket servers of your own |
| Message format | msgpack, numpy arrays as raw bytes | msgpack, arrays as raw bytes | msgpack, numpy arrays as raw bytes; reads openpi's array layout and msgpack-numpy's |
| Camera frames | Resized on the robot with `image_tools.resize_with_pad` | Sent raw, or as JPEG (`encoding`, `jpeg_quality`) | `frames.resize_image` (its `"pad"` is openpi's `resize_with_pad`), `frames.encode` for JPEG and `frames.decode` on the server, `openpi.image_tools` and `openpi.patch()` for openpi's own client and server; `sequence_ai.Video` as low-latency video, a picture not marked as a keyframe of its own (`pictures="exact"`: as it is) |
| Action chunks | `ActionChunkBroker` executes a chunk open-loop and requests the next when it is exhausted | Requests ahead of time, based on the turnaround it measured; aligns the new chunk with the one being executed; real-time chunking modes | None: `act()` returns what the policy returned |
| Robots on one server | Each connection's handler calls `policy.infer` itself; no batching across connections | One robot session; a second receives `busy` | Whatever your server does; `sequence_ai.engine` has `batcher`, `slack` and `replicas` |
| Timing it reports | `server_timing` in each response: the model's time in milliseconds | The turnaround of recent requests, kept by the client | p50, p95 and p99 of each act (`policy.stats()`); each client's RTT and retransmissions, from the server's kernel (`tcp_link`) |

## The connection between requests

This is the one area where the network layer adds to both, so it is described in full.

| | openpi | LeRobot 0.7.0 | These components |
|---|---|---|---|
| What is sent on an idle connection | The `websockets` library's own ping, every 20 seconds by default: neither openpi's `serve(...)` call nor its client's `connect(...)` passes another interval | Zenoh's keep-alive frames: on an idle link between two peers, one every 1.5 s or less often (seen with Zenoh 1.10.2) | A ping every 100 ms while idle: from the robot (`connect()`, `window.Warm`, `openpi.keep`) or from a WebSocket server (`ping`); for Zenoh, a setting that makes it a frame every 100 ms |
| Congestion control on the robot's socket | Not set, so the machine's default | Not set by LeRobot; Zenoh opens the sockets | BBR where Linux allows it (`window.socket_options()`, `window.keep(sock)`); not for Zenoh, which opens its own sockets |
| TCP keepalive | Not set by openpi's client, nor by the `websockets` library under it | Not checked | Set, where the machine accepts the options |
| Tells you what the machine does | No | No | `window.check()`: whether the kernel restarts a window after a pause; `wifi_power.check()`: Wi-Fi power saving |

Why the interval matters: on Linux, a connection that has sent nothing for longer than its retransmission timeout (at
least its round-trip time (RTT) plus 200 ms) has its congestion window restarted. A ping every 20 seconds, or a frame
every 1.5 seconds, is longer than that, so neither keeps a window through a robot's sub-second pauses. That is a
statement about the mechanism. Measure your own link with [`bench/`](../bench/).

## Test coverage

| | Tested with |
|---|---|
| openpi | `openpi.keep()` on openpi-client 0.1.2's real `WebsocketClientPolicy`, against a stand-in server speaking openpi's protocol on one machine ([`examples/quickstart_openpi.py`](../examples/quickstart_openpi.py), `tests/network/test_openpi.py`). On a real link: `openpi.keep()`, and a server that pings, on a WebSocket connection of the kind openpi holds, with a stand-in for its client ([Measurements](measurements.md)). For the technical report: openpi's policy server (commit `215abfb`, `pi05_droid`, on an L4) and its own client in a closed loop over a home connection, with no network component of this package on the connection; and the same server and client with `openpi.keep()`, from a Linux virtual machine on a home connection ([the table](measurements.md#openpikeep-with-openpis-own-server-and-client)); and the same server pinging and decoding the pictures as `openpi.patch()` makes it, with the client sending them as JPEG as `openpi.image_tools` does, from a Mac on a home connection ([the table](measurements.md#pictures-as-jpeg-with-openpis-own-server-and-client)). `openpi.patch()`, the command and `image_tools` against a server module built as openpi's (`tests/network/test_openpi_server.py`) |
| LeRobot | The keep-alive file on Zenoh 1.10.2: two idle peers on one machine then sent a frame about every 100 ms each way (`tests/network/test_lerobot.py`). On a real link: two Zenoh peers, with and without the file ([Measurements](measurements.md)) |
| The components alone | Every measurement in the [README's table](../README.md#results), with the tool in [`bench/`](../bench/) or the servers named there |

## Which should I use?

- **You serve an openpi model (π0, π0-FAST, π0.5).** Use openpi's server and client. Patch the server with
  `openpi.patch()` in its script and import `image_tools` from `sequence_ai.network.openpi` on the robot to send the pictures as JPEG, and
  add `openpi.keep(policy)` if the robot is on Linux and pauses between chunks:
  [How do I run π0 (openpi) on a remote GPU?](how-do-i-run-pi0-openpi-on-a-remote-gpu.md)
- **You serve a LeRobot policy.** Use LeRobot's remote inference. Add the Zenoh file on both sides:
  [How do I reduce the latency of LeRobot remote inference?](how-do-i-reduce-the-latency-of-lerobot-remote-inference.md)
- **You have a model and no server yet, or your own server.** The network components are the transport:
  [How do I serve a robot policy from my own GPU?](how-do-i-serve-a-robot-policy-from-my-own-gpu.md) You then write, or
  take from another project, what openpi and LeRobot give you ready made: chunk playback on the robot, and the model's
  own input and output transforms.
- **Whichever you use,** [measure the link](how-do-i-measure-the-link-between-my-robot-and-my-gpu.md) first. The tool
  needs neither project.

## Where each fact was read

openpi, commit `215abfb217dbac7d5f1273282331b9b1866c0479` (Apache-2.0):

- `src/openpi/serving/websocket_policy_server.py`: the `serve(...)` call and its arguments; the handler that unpacks
  an observation, calls `self._policy.infer(obs)` and sends the response with `server_timing`.
- `packages/openpi-client/src/openpi_client/websocket_client_policy.py`: the client's `connect(...)` call and its
  arguments; `infer`.
- `packages/openpi-client/src/openpi_client/action_chunk_broker.py`: `ActionChunkBroker`.
- `packages/openpi-client/src/openpi_client/msgpack_numpy.py`: the array layout.
- `packages/openpi-client/src/openpi_client/image_tools.py`: `resize_with_pad`.
- `docs/remote_inference.md`: the two commands and the client-side resize.

LeRobot 0.7.0 (Apache-2.0):

- `lerobot/inference/factory.py`, `RemoteInferenceConfig`: `endpoint`, `refill_seconds`, `chunk_merge`, `encoding`,
  `jpeg_quality`, `zenoh_config_path`.
- `lerobot/inference/execution.py`: the turnaround kept from recent requests and the refill threshold made from it.
- `lerobot/remote_inference/server.py` and `lerobot/scripts/lerobot_policy_server.py`: one robot session for a server.
- `lerobot/transport/zenoh.py`, `ZenohConfig`: `config_file`, and the keys `build()` sets over it.
- `lerobot/transport/protocol/codec.py`: msgpack, arrays as raw bytes, JPEG.
- `lerobot/scripts/lerobot_rollout.py`: `--inference.type=remote`.

The `websockets` library, version 17.4: `ping_interval: float | None = 20` in the signatures of its `serve` and
`connect`.

If a fact here is out of date or wrong, please open an issue at
<https://github.com/General-Sequences/sequence-ai/issues>.
