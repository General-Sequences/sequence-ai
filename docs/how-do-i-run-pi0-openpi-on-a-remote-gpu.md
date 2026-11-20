# How do I run π0 (openpi) on a remote GPU?

Run openpi's own policy server on the GPU and openpi's own client on the robot, as openpi's guide describes.
SequenceLink then adds two lines to the server's script, and two on the robot: the server pings each robot
and takes its pictures as JPEG, and the robot sends them that way and keeps its connection from going idle between
action chunks.

On the GPU, in openpi's repository. openpi's `docs/remote_inference.md` starts the server with
`uv run scripts/serve_policy.py --env=DROID`, on port 8000. Add two lines near the top of `scripts/serve_policy.py`,
before it makes the server:

```python
from sequence_ai.network import openpi

openpi.patch()        # openpi's server from here on: a ping each 100 ms, a JPEG picture handed over as its array
```

and start it as before, with sequence-ai in its environment:

```bash
uv run --with sequence-ai scripts/serve_policy.py --env=DROID
```

On the robot:

```python
from sequence_ai.network import openpi
from sequence_ai.network.openpi import image_tools      # was: from openpi_client import image_tools

# openpi_client's own WebsocketClientPolicy, made once the server answers: it waits while the server loads
policy_user = openpi.connect("YOUR_GPU", 8000)
# keeps the connection warm
kept_user = openpi.keep(policy_user)
observation_user["observation/exterior_image_1_left"] = image_tools.resize_with_pad(frame_user, 224, 224)  # now a JPEG
# openpi's own call, unchanged
action_chunk_user = policy_user.infer(observation_user)["actions"]
```

```bash
pip install sequence-ai          # on the robot: no extra needed. The GPU's command above brings its own.
```

π0 and π0.5 are the models; openpi is the repository that serves them. Nothing of openpi's is replaced here: the
client, the server, the msgpack layout and the responses stay openpi's own. A server patched this way serves a robot
that sends arrays as before, and `keep()` needs nothing of the server; a robot that sends its pictures as JPEG needs the
server patched.

## What the lines do

| Line | What it does | When it applies |
|---|---|---|
| `openpi.patch()`, on the GPU | openpi's server pings each connection every 100 ms while no inference runs, and hands its policy each picture sent as JPEG as the array it reads | always; a picture sent as an array reaches the policy as before |
| `image_tools` from `sequence_ai.network.openpi`, on the robot | `resize_with_pad` fits each picture exactly as openpi's does, then returns it as base64 JPEG at quality 95; `convert_to_uint8` returns that as it is | always; the server must be started as above |
| `openpi.keep(policy)`, on the robot | BBR as the socket's congestion control | Linux, with the kernel's BBR module loaded (`sudo modprobe tcp_bbr`) |
| | TCP keepalive | every machine that takes the options |
| | A WebSocket ping each 100 ms the connection is idle, for 60 s after the last `infer()` | always |

A robot that executes an action chunk before requesting the next one leaves its connection silent in between. On
Linux, a connection silent for longer than its retransmission timeout has its congestion window restarted, so the
next observation can take more round trips to upload.
[Why does my robot's upload get slow after a pause?](why-does-my-robots-upload-get-slow-after-a-pause.md) has the
mechanism. The pings keep the connection from going silent while the server can answer them. With openpi's own server
on Linux, also load BBR (`sudo modprobe tcp_bbr`, which `keep()` then puts on the connection) so the window holds
across a pause ([measurements](measurements.md)).
JPEG makes the upload itself smaller: two 224 x 224 pictures are 301,056 bytes as arrays.

`kept.bbr` reports whether the socket accepted BBR, `kept.congestion` which congestion control it now uses, and
`kept.warm.pings` how many pings were answered. `kept.close()` stops the pings; the connection stays openpi's.

`openpi.patch()` runs before the script makes openpi's `WebsocketPolicyServer`, and needs openpi's own
`websocket_policy_server.py`; any other server is refused with `TypeError`.

## Try it without a GPU or a checkpoint

[`examples/quickstart_openpi.py`](../examples/quickstart_openpi.py) starts a stand-in server on your machine that
speaks openpi's protocol, then runs openpi's real client against it with the one line added. It needs openpi's client
(`packages/openpi-client` in openpi's repository, `pip install -e .`).

```bash
python examples/quickstart_openpi.py
```

Run in a new environment with sequence-ai 0.64.1 and openpi-client 0.1.0, on a Mac, it printed:

```text
BBR on its socket: False; TCP keepalive options accepted: 4
infer 0: a chunk of 10 actions
infer 1: a chunk of 10 actions
infer 2: a chunk of 10 actions
pings answered while it was idle: 12
```

Twelve pings is four in each of the three half-second pauses. BBR is `False` because a Mac has none.

Against a real openpi server, [`examples/openpi_robot.py`](../examples/openpi_robot.py) takes its host and port.

## Or serve the policy as a class

To serve several robots in batches, or to give the robot `connect()` in place of openpi's client, write the policy as
a class in openpi's environment on the GPU, and serve it there with `seq serve`. openpi's `download.maybe_download`
fetches the `pi05_droid` checkpoint once, from the address openpi's README gives for it, into `~/.cache/openpi`
(`OPENPI_DATA_HOME` moves it); `droid_policy.make_droid_example()` makes a random observation with DROID's keys and
shapes, for the warm-up:

```python
from openpi.policies import droid_policy, policy_config
from openpi.shared import download
from openpi.training import config as openpi_config

import seq


@seq.policy(hz=15, action_chunk_valid_len=15, action_chunk_replan_at=8)
class Pi05Droid_user:
    def __init__(self):
        checkpoint_dir_user = download.maybe_download(pi05_droid_checkpoint_user)   # the address in openpi's README
        self.policy_user = policy_config.create_trained_policy(openpi_config.get_config("pi05_droid"),
                                                               checkpoint_dir_user)

    @seq.sample
    def sample_user(self):     # two 224x224 pictures, 7 joint positions, a gripper, a prompt: an act after each
        return [("observe", droid_policy.make_droid_example()), ("act",)] * 3

    @seq.infer
    def infer_user(self, observation_user):
        return self.policy_user.infer(observation_user)
```

```bash
seq serve                    # in the policy's folder
```

On the robot, the observation keeps openpi's keys, and `act()` returns what openpi's `infer()` returns:

```python
import sequence_ai

policy_user = sequence_ai.connect("http://YOUR_GPU:8000")
# observation["observation/exterior_image_1_left"]: the frame
policy_user.observe(observation_user)
action_chunk_user = policy_user.act()["actions"]
```

Batched, the acts that wait reach `@seq.infer` together, and `batched(policy)`, in
[`examples/serve_openpi_batched.py`](../examples/serve_openpi_batched.py), passes each observation through the policy's
transforms, as `infer()` does, and samples all of them in one call, each with its own noise:
[How do I use the inference components with openpi, LeRobot or GR00T?](how-do-i-use-the-inference-components-with-openpi-lerobot-or-gr00t.md#openpi-and-lerobot-more-than-one-robot-on-a-gpu)
has the class. The policy's JAX model compiles each listed batch size once, in the warm-up on its sample. A PyTorch
checkpoint runs its observations one by one. Every default of [What is on by
default](what-is-on-by-default.md) applies: the robot sends each 224×224 picture as a video keyframe of its own, and the
server turns it back into the array the policy reads; `connect(url, pictures="exact")` sends exact pixels.

## Tested

- **Verified:** `openpi.keep()` against openpi-client 0.1.2 on a local connection, as above and in
  `tests/network/test_openpi.py`.
- **Verified:** `batched()` of `examples/serve_openpi_batched.py` against stand-ins of openpi's model and transforms:
  one call to the model per batch, each observation's own transforms and output (`tests/engine/test_openpi.py`).
- **Verified:** the server's pings and its decoding, and `image_tools`, against a server module built as openpi's
  `websocket_policy_server.py` is, on a local connection; and the command, against a script run as
  `scripts/serve_policy.py` is (`tests/network/test_openpi_server.py`).
- **Measured on a real link:** `keep()` on a WebSocket connection of the kind openpi's client
  holds (the `websockets` library on both ends, compression off, a stand-in for the client class), between two cloud
  machines. A 100 KB upload 0.53 s after each response took 37.6, 44.4, 82.4 and 163.8 ms at round-trip times (RTTs) of
  11, 21, 41 and 81 ms, and 12.7, 22.7, 41.8 and 82.6 ms with `keep()`: two to three round trips became one, on 40 of 40
  uploads. [What do the network components save?](measurements.md) has the table.
- **Measured with openpi's own server and client:** one closed-loop run with π0.5 (`pi05_droid`) on an L4, the client in
  a Linux virtual machine on a home connection, 301 KB per request. The time a request spent outside inference had a
  median of 181 and 152 ms in two episodes without `keep()`; with it, 92 ms; with the BBR module loaded as well, 86 and
  91 ms.
  [The table](measurements.md#openpikeep-with-openpis-own-server-and-client) has the idle times and the congestion
  window.
- **Measured with openpi's own server and client, the pictures as JPEG:** one closed-loop run with π0.5 on an L4, the
  robot a simulated arm on a Mac on a home connection, the server pinging. Each request's upload went from 301.5 KB to
  18.8 KB, and the time it spent outside the server, beyond one RTT, from about 97 to about 21 ms at the median.
  [The table](measurements.md#pictures-as-jpeg-with-openpis-own-server-and-client).

## What openpi already does

openpi's client and server already do several things a remote policy needs, so this package has no component for them:

- **One connection, kept open.** The client connects once and reuses the WebSocket for every `infer()`.
- **Arrays as raw bytes.** Observations and responses are sent as msgpack with numpy arrays as raw bytes.
- **Images resized on the robot.** openpi's guide resizes each image on the client with
  `image_tools.resize_with_pad(img, 224, 224)` before sending it, as an array.
- **Action chunks executed open-loop.** `ActionChunkBroker` requests a new chunk only when the current one is
  exhausted.

If you write your own server or client for an openpi model, `msgpack_io` reads openpi's array layout,
`frames.resize_image(..., mode="pad", filter="bilinear")` is openpi's `resize_with_pad`, byte for byte, and
`frames.decode` turns a picture sent as JPEG back into its array. [The comparison page](comparison.md) sets the two
side by side.

## If something fails

- `TypeError: this client keeps no WebSocket connection at _ws`: pass the `WebsocketClientPolicy` itself to
  `keep()`, not an `ActionChunkBroker` wrapped around it. Keep the broker for `infer()`.
- `ConnectionClosedError: sent 1011 (internal error) keepalive ping timeout` on the robot's first `infer()`: openpi's
  server answers no ping while its first inference compiles the model. `openpi.keep(policy)` keeps the connection
  through it.
- `TimeoutError: timed out while waiting for handshake response` as the robot connects: the server is loading its
  policy, or compiling another robot's first inference. `openpi.connect(host, port)` waits for it.
- `RuntimeError: Error in inference server:` on the robot, after its pictures went as JPEG: the server was started
  without `sequence_ai.network.openpi`, so its policy was handed the pictures as text. Start it with the command above,
  or import `image_tools` from `openpi_client` again.
- `ValueError: observation/...: not a JPEG or PNG image` in the server's traceback: a picture under that key came as
  text or bytes that are not one. Send it as `image_tools` makes it, or as an array.
- `TypeError: this server is not built as openpi's websocket_policy_server.py is`: the openpi on the GPU starts its
  server another way than `patch()` was written for. Start it with openpi's own command, and send arrays.
- `ModuleNotFoundError: No module named 'typing_extensions'` on importing `openpi_client`: openpi-client 0.1.0 imports
  it without listing it as a dependency, so a new environment may lack it. `pip install typing_extensions`.
- The client prints nothing and never returns: while the connection is refused it retries every 5 seconds, so the
  server is not up yet or the port is wrong. This is read from openpi's source, not made to happen here.

More in the [FAQ](faq.md#with-openpi-and-lerobot).

## Where the GPU comes from

Any GPU the robot can reach on port 8000. [How do I get a GPU near my robot?](how-do-i-get-a-gpu-near-my-robot.md)
covers the case where you have none nearby. The robot's code changes in one place: the `host`.
