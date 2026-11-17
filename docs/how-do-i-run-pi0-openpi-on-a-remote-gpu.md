# How do I run π0 (openpi) on a remote GPU?

Run openpi's own policy server on the GPU and openpi's own client on the robot, as openpi's guide describes.
`sequence-ai` then adds one line on the robot, which keeps the connection that client already holds from going idle between
action chunks.

On the GPU, in openpi's repository (the command is from openpi's `docs/remote_inference.md`; it serves on port 8000):

```bash
uv run scripts/serve_policy.py --env=DROID
```

On the robot:

```python
from openpi_client import websocket_client_policy
from sequence_ai.network import openpi

policy = websocket_client_policy.WebsocketClientPolicy(host="YOUR_GPU", port=8000)
kept = openpi.keep(policy)                              # the one line sequence-ai adds
action_chunk = policy.infer(observation)["actions"]     # openpi's own call, unchanged
```

```bash
pip install sequence-ai          # no extra needed: this component imports only the standard library
```

π0 and π0.5 are the models; openpi is the repository that serves them. Nothing of openpi's is replaced here: the
client, the server, the msgpack layout and the responses stay openpi's own.

## What the one line does

`openpi.keep(policy)` takes the WebSocket connection the client opened and puts three things on it:

| What | When it applies |
|---|---|
| BBR as the socket's congestion control | Linux, with the kernel's BBR module loaded (`sudo modprobe tcp_bbr`) |
| TCP keepalive | every machine that takes the options |
| A WebSocket ping each 100 ms the connection is idle, for 60 s after the last `infer()` | always |

A robot that executes an action chunk before requesting the next one leaves its connection silent in between. On
Linux, a connection silent for longer than its retransmission timeout has its congestion window restarted, so the
next observation can take more round trips to upload.
[Why does my robot's upload get slow after a pause?](why-does-my-robots-upload-get-slow-after-a-pause.md) has the
mechanism and the links where it did and did not cost anything.

`kept.bbr` reports whether the socket accepted BBR, `kept.congestion` which congestion control it now uses, and
`kept.warm.pings` how many pings were answered. `kept.close()` stops the pings; the connection stays openpi's.

## Try it without a GPU or a checkpoint

[`examples/quickstart_openpi.py`](../examples/quickstart_openpi.py) starts a stand-in server on your machine that
speaks openpi's protocol, then runs openpi's real client against it with the one line added. It needs openpi's client
(`packages/openpi-client` in openpi's repository, `pip install -e .`).

```bash
python examples/quickstart_openpi.py
```

Run in a new environment with sequence-ai 0.62.0 and openpi-client 0.1.2, on a Mac, it printed:

```text
BBR on its socket: False; TCP keepalive options accepted: 4
infer 0: a chunk of 10 actions
infer 1: a chunk of 10 actions
infer 2: a chunk of 10 actions
pings answered while it was idle: 12
```

Twelve pings is four in each of the three half-second pauses. BBR is `False` because a Mac has none.

Against a real openpi server, [`examples/openpi_robot.py`](../examples/openpi_robot.py) takes its host and port.

## Tested

- **Verified:** `openpi.keep()` against openpi-client 0.1.2 on a local connection, as above and in
  `tests/network/test_openpi.py`.
- **Measured on a real link:** `keep()` on a WebSocket connection of the kind openpi's client
  holds (the `websockets` library on both ends, compression off, a stand-in for the client class), between two cloud
  machines. A 100 KB upload 0.53 s after each response took 37.6, 44.4, 82.4 and 163.8 ms at round-trip times (RTTs) of
  11, 21, 41 and 81 ms, and 12.7, 22.7, 41.8 and 82.6 ms with `keep()`: two to three round trips became one, on 40 of 40
  uploads. [What do the network components save?](measurements.md) has the table.
- **Measured with openpi's own server and client:** one closed-loop run with π0.5 (`pi05_droid`) on an L4, the client in
  a Linux virtual machine on a home connection, 301 KB per request. The time a request spent outside inference had a
  median of 181 and 152 ms in two episodes without `keep()`; with it, 92 ms in an episode without packet loss and 157 ms
  in one that lost packets; with the BBR module loaded as well, 86 and 91 ms.
  [The table](measurements.md#openpikeep-with-openpis-own-server-and-client) has the idle times and the congestion
  window.

## Limitations

- With openpi's own policy server, `keep()` has been measured on one link only: two 45 s episodes per setting, from a
  Linux virtual machine on a home connection. In the episode that lost packets it was no better than without it.
- The README's [limitations](../README.md#limitations) list the links where the idle restart costs nothing.
- The same ping from the server side has not been tested with openpi's server. That server is built on the
  `websockets` library, whose `serve(...)` takes `ping_interval`; `ping.websockets_options()` gives
  `{'ping_interval': 0.1, 'ping_timeout': 20.0}`. Passing it requires a change in openpi's
  `websocket_policy_server.py`. `keep()` requires no change to openpi.

## What openpi already does

openpi's client and server already do several things a remote policy needs, so this package has no component for them:

- **One connection, kept open.** The client connects once and reuses the WebSocket for every `infer()`.
- **Arrays as raw bytes.** Observations and responses are sent as msgpack with numpy arrays as raw bytes.
- **Images resized on the robot.** openpi's guide resizes each image on the client with
  `image_tools.resize_with_pad(img, 224, 224)` before sending it.
- **Action chunks executed open-loop.** `ActionChunkBroker` requests a new chunk only when the current one is
  exhausted.

If you write your own server or client for an openpi model, `msgpack_io` reads openpi's array layout, and
`frames.resize_image(..., mode="pad", filter="bilinear")` is openpi's `resize_with_pad`, byte for byte.
[The comparison page](comparison.md) sets the two side by side.

## If something fails

- `TypeError: this client keeps no WebSocket connection at _ws`: pass the `WebsocketClientPolicy` itself to
  `keep()`, not an `ActionChunkBroker` wrapped around it. Keep the broker for `infer()`.
- `ModuleNotFoundError: No module named 'typing_extensions'` on importing `openpi_client`: openpi-client 0.1.0 imports
  it without listing it as a dependency, so a new environment may lack it. `pip install typing_extensions`.
- The client prints nothing and never returns: while the connection is refused it retries every 5 seconds, so the
  server is not up yet or the port is wrong. This is read from openpi's source, not made to happen here.

More in the [FAQ](faq.md#with-openpi-and-lerobot).

## Where the GPU comes from

Any GPU the robot can reach on port 8000. [How do I get a GPU near my robot?](how-do-i-get-a-gpu-near-my-robot.md)
covers the case where you have none nearby. The robot's code changes in one place: the `host`.
