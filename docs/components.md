# What does each network component do?

Each component is one file that you can use independently. This page gives, for each: the lines to add, why it
exists, what was measured and on which link, and the output of its example. The examples were run on a Mac with
Python 3.12, so the components that read the Linux kernel return `None` or `False` there.

| Client side | Server side | Both sides | Combined |
|---|---|---|---|
| [`window`](#window-the-congestion-window-kept-through-the-pauses) | [`ping`](#ping-a-server-side-ping) | [`msgpack_io`](#msgpack_io) | [`connect`](#connect-the-client-side-components-combined) |
| [`compress`](#compress-only-where-it-saves-time) | [`one_write`](#one_write-a-response-in-one-write) | [`latency_stats`](#latency_stats) | [`openpi`](#openpi-and-lerobot) |
| [`frames`](#frames-a-frame-resized-before-upload-and-decoded-on-the-server) | [`tcp_link`](#tcp_link-each-clients-link-read-from-the-kernel) | | [`lerobot`](#openpi-and-lerobot) |
| [`wifi_power`](#wifi_power-wi-fi-power-saving-detected) | [`compressed`](#compressed-decompress-what-the-client-compressed) | | [`bench/`](how-do-i-measure-the-link-between-my-robot-and-my-gpu.md) |

What a component saves depends on your link, and every number below is given with the link it was measured on. The
inference components (`sequence_ai.engine`) are described in the [README](../README.md#inference-layer).

## Client side

### `window`: the congestion window, kept through the pauses

```python
from sequence_ai.network import window

transport_user = httpx.HTTPTransport(socket_options=window.socket_options())    # BBR and TCP keepalive, where it can
window.keep(sock_user)                                                          # or BBR on your own socket

warm_user = window.Warm(ping=lambda: client.get("/ping"))    # a ping while the connection is idle
warm_user.used()                                              # each time you send on it

window.check()                                           # warns once: what this machine does after a pause
```

**Why.** On Linux a connection that sends nothing for longer than its retransmission timeout has its congestion window
restarted, so the next upload takes one extra round trip. A robot that sends a request every half second incurs it on
every request.

**Measured.**

- **BBR.** A robot sending a request every 0.53 s with 24 KB of frames, on a round-trip time (RTT) of about 21 ms: 40 of
  40 requests took one more round trip without it (21.5 ms), and none with it (0.5 to 3.0 ms).
- **Where the pause cost nothing.** On RTTs of about 101 and 241 ms the same robot took no extra round trip
  with the default settings (0 or 1 of 40 requests), until the link lost packets (38 to 40 of 40 at 1% loss) or the size
  of the request varied.
- **A ping while idle.** On an RTT of 226 ms, 162 KB uploads 3 s apart took 303.6 ms to arrive without the
  pings, and under 1 ms with them.

A ping cannot keep a window that the pings do not fill; BBR, or `tcp_slow_start_after_idle=0`, keeps that one.

**An act in halves.** Where the window restarts anyway - on Linux without BBR, after a policy whose answer takes longer
than the retransmission timeout - `connect()` sends an act whose body is more than the restarted window's first flight
and no more than two (13,032 to 26,064 bytes on Ethernet) in two halves at once, each on a connection of its own, so
each fits a first flight of its own. `serve()` joins them and runs the act once. Measured at a 50 ms round trip with a
200 ms policy, 13 to 26 KB acts took a p50 of 108 to 111 ms whole and 56.5 to 57.0 ms in halves; `split=False` sends
them whole.
`window.check()` issues a warning once per process, with both ways to keep the window, and changes nothing on the
machine.
[Why does my robot's upload get slow after a pause?](why-does-my-robots-upload-get-slow-after-a-pause.md) has the
full table.

`python examples/components/window.py` printed:

```text
socket options this machine accepts: [(65535, 8, 1), (6, 16, 20), (6, 257, 5), (6, 258, 3)]
restarts a window after a pause: False
BBR on your own socket: False; its congestion control: None
pings in 0.3 s of idle time: 5
```

### `compress`: only where it saves time

```python
from sequence_ai.network import compress, msgpack_io

squeeze_user = compress.Squeeze()                                       # one per connection
# large arrays compressed in place, if that saves time
body_user = msgpack_io.pack(squeeze_user.apply(observation_user))
squeeze_user.uplink.note(len(body_user), seconds_the_upload_took_user)  # so it learns this link's upload rate
```

An array is compressed with zstd level 1 only while the upload time that saves, at the uplink rate measured on this
connection, exceeds the time to compress and decompress it. Data that does not compress, such as a JPEG's bytes, is sent
uncompressed and tried again later. Without the `zstandard` package (and before Python 3.14, which has zstd built in) it
uses zlib level 1.

**Measured.** Camera frames of 172,800 bytes (320 by 180, from a public robot dataset): 0.37 ms to compress, 0.21 ms
to decompress, about 1.9 times smaller, on a laptop. Those frames were decoded from video, which compresses better
than a camera's raw frames.

`python examples/components/compress.py` printed:

```text
a frame of 921,600 bytes was uploaded as 37,617
decompressed byte for byte: 884,137 bytes larger than what arrived
on that link the next frame is sent uncompressed: True
```

The example's frame is synthetic, flat regions and edges, so it compresses far more than a camera's does. The last
line shows the rule: once the link is fast enough that compressing costs more than it saves, the frame is sent
uncompressed.

### `frames`: a frame resized before upload, and decoded on the server

```python
from sequence_ai.network import frames

frame_user = frames.resize_image(camera_user.read(), 224, 224, mode="pad", filter="bilinear")
data_user = frames.encode(frame_user)            # a base64 JPEG
image_user = frames.decode(data_user)            # on the server: the HxWx3 uint8 array its policy reads
```

`"pad"` is openpi's `resize_with_pad`, byte for byte. The mode and the filter have no default and must be passed: a
policy sees what it was trained on. `frames.encode(frame)` gives a base64 JPEG for an input that carries its frames as
text, and `frames.image_size(data)` reads a frame's width and height from its header without Pillow. On the server,
`frames.decode(data)` turns that JPEG, or a PNG, back into the array, and returns an array it is given unchanged. It
opens no other format, since a picture comes over the network.

`python examples/components/frames.py` printed:

```text
pad: (224, 224, 3) with black rows at the top: True
stretch: (224, 224, 3)
encoded: 17,056 characters for 921,600 bytes; size from its header: (224, 224)
decoded: (224, 224, 3) uint8, 2.0 levels from the frame on average
```

### `wifi_power`: Wi-Fi power saving, detected

```python
from sequence_ai.network import wifi_power

wifi_power.check()    # the interface whose power saving is on, after warning how to disable it; else None
```

With power saving enabled, a Wi-Fi adapter sleeps between the access point's beacons, and a policy's response is
delayed until the adapter wakes. `check()` reads the machine's settings and changes nothing. Linux only.

`python examples/components/wifi_power.py` printed, on a Mac:

```text
Wi-Fi power saving is on: no (or this is not Linux, or not Wi-Fi)
```

### `connect`: the client-side components combined

```python
from sequence_ai.network import connect

with connect("http://your-gpu:8000", compress=True) as link_user:     # compress: only if your server decompresses
    response_user = link_user.act({"image": frame_user, "state": state_user})
    print(link_user.stats()["latency_us"]["act_us"])                   # the requests' p50, p95 and p99
```

One httpx connection with `window`'s socket options on it, a small GET while it is idle, msgpack both ways, and each
request timed. A ping and a request never run at once and never take a second connection: a client that starts a request
while a ping is in flight would open a new connection for it, with a new handshake and a cold window.

Your server needs a POST that takes a msgpack body and responds with msgpack or JSON, and any cheap GET for the
ping (`ping_path=None` disables the pings). [`examples/server.py`](../examples/server.py) is one, and so is the
[Quickstart](quickstart.md)'s.

| Option | Default | What it is |
|---|---|---|
| `act_path` | `"/act"` | the POST path the observation is sent to |
| `ping_path` | `"/ping"` | the GET sent while idle; `None` for no pings |
| `ping_every_s` | `0.1` | the idle time after which a ping is sent |
| `ping_for_s` | `60.0` | how long pings continue after the last request |
| `compress` | `False` | compress large arrays where that saves time; the server must call `compressed.expand` |
| `timeout` | `10.0` | seconds to wait for a response |
| `headers` | `None` | headers sent with every request |
| `check` | `True` | warn once if this machine restarts its window after a pause, and if Wi-Fi power saving is on |
| `split` | `True` | an act whose window a pause restarted, one to two first flights long, sent in two halves on two connections while the server joins them |

**One connection, kept open.** This is one line in any client: create one `httpx.Client` and reuse it. Measured: a
new connection for each call took a median 305.9 ms; one reused connection, 121.0 ms. The handshake was 184.9 ms per
call. `connect()` keeps one.

## Server side

### `ping`: a server-side ping

```python
import websockets
from sequence_ai.network import ping

# websockets' own server, with a ping every 100 ms
async with ping.serve(handler_user, host_user, port_user) as server_user:
    await server_user.serve_forever()

# the same, as its options
async with websockets.serve(handler_user, host_user, port_user, **ping.websockets_options()):
    ...

uvicorn.run(app_user, **ping.uvicorn_options())                                         # the same, under uvicorn
```

A WebSocket library answers a ping with a pong by itself. A server that pings therefore makes the robot send a few
bytes each interval, and the robot's kernel never counts the connection idle. Nothing changes on the robot.

**Measured.** Between machines of one cloud provider, on RTTs of 11.7 and 69.2 ms, an upload after 500 ms of silence
took 3.0 round trips; with the server pinging every 100 ms it took 1.1 (100 KB and 500 KB) and 1.25 (1 MB).

A server that answers one HTTP request with one response cannot send anything unsolicited. There the ping comes from the
robot: `window.Warm`, which `connect()` applies to its connection. [How do I keep a WebSocket policy server's connection
warm?](how-do-i-keep-a-websocket-policy-servers-connection-warm.md) has the rest.

`python examples/components/ping.py` printed:

```text
response: observation
the server measured the connection's RTT from its pings: 0.17 ms
```

### `one_write`: a response in one write

```python
import uvicorn
from sequence_ai.network.one_write import WholeResponse, one_write_http

uvicorn.run(WholeResponse(app_user), http=one_write_http())
```

`app_user` is any ASGI app: FastAPI's and Starlette's too.

uvicorn writes a response in pieces. A tunnel or proxy in front of the server can delay every piece after the first
until the client has acknowledged the first, so each request takes one more round trip than necessary.

**Measured.** Behind such a tunnel, with a TCP connect of p50 89.5 ms: headers then body, GET p50 189.7 ms; the same
response in one write, 106.7 ms.

`python examples/components/one_write.py` printed `response: b'ok'`.

### `tcp_link`: each client's link, read from the kernel

```python
from sequence_ai.network import tcp_link

tcp_link.read(sock_user)             # any connected socket
tcp_link.of(scope_user["client"])    # in an ASGI app served with one_write_http(): this request's own connection
```

Each returns `rtt_ms`, `rttvar_ms`, `retrans`, `total_retrans`, `cwnd` and, where the kernel provides it,
`delivery_kbps`; or `None` where the kernel provides none, as on macOS. A robot's own OS may not expose these to
the program that runs on it, and the server reads every robot the same way.

`python examples/components/tcp_link.py` printed, on a Mac:

```text
this connection, as the server's kernel reports it: None
```

### `compressed`: decompress what the client compressed

```python
from sequence_ai.network import compressed, msgpack_io

observation_user, grew_user = compressed.expand(msgpack_io.unpack(body_user), limit=64 << 20)
```

An item whose declared size is smaller than its decompressed size is rejected rather than decompressed past `limit`.

## Both sides

### `msgpack_io`

```python
from sequence_ai.network import msgpack_io

body_user = msgpack_io.pack({"image": frame_user, "state": joints_user})
observation_user = msgpack_io.unpack(body_user)
```

Bytes are sent as bytes and numpy arrays as their raw data, whereas JSON carries bytes as base64, a third larger on the
wire. It is plain Python, so neither side installs a msgpack package. It writes arrays as msgpack-numpy does and reads
openpi's layout as well. An object array is rejected: decoding one would run code chosen by the sender.

`python examples/components/msgpack_io.py` printed:

```text
150,670 bytes for an image of 150,528; content type application/x-seq-msgpack
arrays returned as arrays: (224, 224, 3) float32
```

### `latency_stats`

```python
from sequence_ai.network.latency_stats import LatencyStats, now_us

stats_user = LatencyStats(latencies=["request_us"])
sent_user = now_us(); ...; stats_user.since("request_us", sent_user)
stats_user.snapshot()["latency_us"]["request_us"]    # n, mean, max, p50, p95, p99
```

A percentile is `None` until its window contains enough samples for it: the p99 of 20 samples is only the largest of the
20. So p50 needs 2 samples, p95 needs 20 and p99 needs 100.

`python examples/components/latency_stats.py` printed:

```text
200 requests: p50 248 ms, p95 281 ms, p99 290 ms
```

## openpi and LeRobot

**openpi.** Keep openpi's client and server. On the GPU, start openpi's own server through this package: it then pings
each robot every 100 ms and hands its policy each picture sent as JPEG as the array it reads. On the robot, import
`image_tools` from here, so its pictures go as JPEG, and add the client-side components to the connection the client
holds.

```bash
uv run --with sequence-ai python -m sequence_ai.network.openpi scripts/serve_policy.py --env=DROID   # the GPU
```

```python
from sequence_ai.network import openpi
from sequence_ai.network.openpi import image_tools     # in place of openpi_client's: pictures go as JPEG

policy_user = openpi.connect(host_user, port_user)    # openpi_client's own client, once its server answers
kept_user = openpi.keep(policy_user)  # BBR and TCP keepalive on its socket; a WebSocket ping while it is idle
observation_user["observation/image"] = image_tools.resize_with_pad(frame_user, 224, 224)
action_user = policy_user.infer(observation_user)     # as before
```

In a server script of your own, `openpi.patch()` before it makes openpi's server does what the command does.

`keep()` is verified against openpi-client 0.1.2 on a local connection, and the server side and `image_tools` against
a server module built as openpi's `websocket_policy_server.py` is. Measured on a WebSocket connection of the kind
openpi's client holds, between two cloud machines: a 100 KB upload 0.53 s after each response took two to three round
trips without `keep()` and one with it ([Measurements](measurements.md)). With openpi's own server and client, `keep()`
was measured in one closed-loop run ([the table](measurements.md#openpikeep-with-openpis-own-server-and-client)), and
the pictures as JPEG with the server's pings in another
([the table](measurements.md#pictures-as-jpeg-with-openpis-own-server-and-client)).
[How do I run π0 (openpi) on a remote GPU?](how-do-i-run-pi0-openpi-on-a-remote-gpu.md)

**LeRobot.** LeRobot 0.7.0's remote inference talks over Zenoh, which sends a frame on an idle link only every few
seconds. One setting makes it every 100 ms:

```python
from sequence_ai.network import lerobot

path_user = lerobot.keep_alive_file("zenoh-keep-alive.json5")
```

Give the file to both sides: the policy server reads it from `zenoh.config_file`, and the robot's remote inference
from `zenoh_config_path`.

Checked: Zenoh 1.10.2 takes the setting, and two idle peers on one machine then sent a frame every 102 ms each way.
Measured between two Zenoh peers on two cloud machines: a 100 KB request 0.53 s after each response took two to three
round trips without the file and one with it ([Measurements](measurements.md)).
[How do I reduce the latency of LeRobot remote inference?](how-do-i-reduce-the-latency-of-lerobot-remote-inference.md)
