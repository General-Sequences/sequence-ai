# FAQ: errors you may see, and what to do next

Each entry starts with what to do. The error text under it is the code's output when the error was reproduced
(macOS on Apple silicon, Python 3.12, sequence-ai 0.62.0, httpx 0.28, uvicorn 0.54), unless the entry states
otherwise.

Network layer (`sequence_ai.network`):

- [Installing](#installing)
- [Connecting](#connecting)
- [Sending and reading](#sending-and-reading)
- [With openpi and LeRobot](#with-openpi-and-lerobot)
- [On the server](#on-the-server)
- [What a component reports about your machine](#what-a-component-reports-about-your-machine)
- [A network component made nothing faster](#a-network-component-made-nothing-faster)

Inference layer (`sequence_ai.engine`):

- [`cuda_graphs`](#cuda_graphs)
- [`batcher`](#batcher)
- [`replicas`](#replicas)
- [`slack`](#slack)

[General](#general)

## Installing

### `ImportError: connect() uses httpx for its connection, and httpx is not installed`

httpx is a dependency of `sequence-ai`, so `pip install sequence-ai` installs it. The error appears where `link.py` was
copied into a project that has no httpx: install it there, as the error message states.

```text
ImportError: connect() uses httpx for its connection, and httpx is not installed: pip install httpx
```

### `ModuleNotFoundError: No module named 'numpy'` or `'uvicorn'`

The quickstart's two files import numpy, and its server runs under uvicorn. Both are dependencies of `sequence-ai`, so
`pip install sequence-ai` installs them. The error appears where the files were copied into a project that has neither:
`pip install numpy uvicorn`.

```text
ModuleNotFoundError: No module named 'numpy'
ModuleNotFoundError: No module named 'uvicorn'
```

numpy is optional for the components themselves. Without it, `connect()` and `msgpack_io` carry plain values, and an
array the server sent comes back as nested lists:

```text
{'state': [0.0, 1.0], 'prompt': 'pick'}
```

### `ImportError: resizing a frame needs Pillow, which is not installed here`

Pillow is a dependency of `sequence-ai`. The error appears where `frames.py` was copied into a project that has no
Pillow: `pip install pillow`. `frames.resize_image`, `frames.encode` and `frames.decode` use Pillow, and
`openpi.image_tools` and `openpi.patch()` through them; no other component does.

```text
ImportError: resizing a frame needs Pillow, which is not installed here: pip install pillow
ImportError: encoding an array or PIL image needs Pillow, and it is missing or broken here: pip install pillow. Pass an already-encoded base64 str or bytes to avoid it.
```

The first line is what `frames.resize_image` raises, the second what `frames.encode` raises.

### What do I install?

| You use | Install |
|---|---|
| `connect()` on the robot, the quick start's server, `frames`, `window`, `ping`, `tcp_link`, `one_write` under uvicorn, `wifi_power`, `msgpack_io`, `latency_stats`, `lerobot`, every inference component, numpy arrays in observations or responses, `compress=True` | `sequence-ai`, nothing more |
| `bench/`, the measuring tool | a checkout of the repository; it is standard library only |

Python 3.9 or later.

## Connecting

### `httpx.ConnectError: [Errno 61] Connection refused`

Nothing is listening at that address and port. Check, in this order:

1. The server is running, and its log shows the port you gave `connect()`.
2. The server listens on an address the robot can reach. `127.0.0.1` is reachable from the same machine only.
   `serve()` listens on every address unless it is given `host`; a server of your own, on a GPU elsewhere, starts on
   `0.0.0.0`.
3. The URL starts with `http://` and has the port: `connect("http://YOUR_GPU:8000")`.

```text
httpx.ConnectError: [Errno 61] Connection refused: no response from http://127.0.0.1:8000. Is the policy server running there, listening on an address this machine can reach (0.0.0.0, not 127.0.0.1, for another machine), on that port?
```

The text up to the first colon after `ConnectError:` comes from the operating system, so Linux prints a different
number. The rest comes from `connect()`, with the address it was given.

### `httpx.ConnectTimeout: timed out`

The address did not respond at all. That is the symptom of a closed firewall: the GPU's network drops the robot's
packets instead of refusing them. Open the server's port to the robot's address in the GPU's firewall, then try again.

```text
httpx.ConnectTimeout: timed out
```

It is raised after `timeout` seconds, 10 unless you pass another: `connect(url, timeout=2)`.

### `httpx.ConnectError: [Errno 8] nodename nor servname provided, or not known`

The host name in the URL does not resolve. Check the spelling, or use the GPU's IP address.

```text
httpx.ConnectError: [Errno 8] nodename nor servname provided, or not known
```

### `httpx.ReadTimeout: no response from ... The server may still be running this act`

The server took longer than the robot waits: up to 120 s for the first act (`first_timeout`), 10 s for the others
(`timeout`). A model's first inference is often its slowest, because it compiles then. Start the server with
`serve(policy, example=observation)`, which runs the policy before it says it is ready, and call
`link.wait_until_ready()` on the robot before it moves. If every call is slow, the time is in the model and not in the
link: `python -m sequence_ai.check` says how long the server says its model took.

With `timeout=1.0, first_timeout=1.0` against a model that took 3 s (sequence-ai 0.63.0):

```text
httpx.ReadTimeout: no response from http://127.0.0.1:8013 in 1 s. The server may still be running this act - a model's first inference compiles it, often for tens of seconds - and the next act waits for it first, as long as a first act waits. Wait for the server once, before the robot moves, with wait_until_ready(example=observation), or give connect() a longer first_timeout= (now 1 s) or timeout= (now 1 s)
```

## Sending and reading

### `httpx.HTTPStatusError: Client error '404 Not Found'`

The server has no POST at that path. `connect()` posts to `/act` unless told otherwise:
`connect(url, act_path="/infer")`, or `link.act(observation, path="/infer")` for one call.

```text
httpx.HTTPStatusError: Client error '404 Not Found' for url 'http://127.0.0.1:8012/infer'
```

### `httpx.HTTPStatusError: Client error '401 Unauthorized'`

The server was started with a token, and the robot sent none or another. Connect with the server's token:
`sequence_ai.connect(url, token="...")`, the one given to `serve(policy, token=...)` or set as `SEQ_SERVE_TOKEN` on the
GPU. `wait_until_ready()` says the same of `/ping`. Reproduced with sequence-ai 0.63.0:

```text
httpx.HTTPStatusError: Client error '401 Unauthorized' for url 'http://127.0.0.1:8011/act'
```

### `httpx.HTTPStatusError: Server error '500 Internal Server Error'` after adding `compress=True`

A server of your own does not decompress the arrays. (`serve()` does, and says so, and `connect()` compresses
nothing for a server that does not say it decompresses; `compress=True` compresses whatever the server says.) Either
decompress where the server reads the body, or leave `compress=True` off:

```python
from sequence_ai.network import compressed, msgpack_io

observation_user, _ = compressed.expand(msgpack_io.unpack(body_user), limit=64 << 20)
```

The robot sees this:

```text
httpx.HTTPStatusError: Server error '500 Internal Server Error' for url 'http://127.0.0.1:8012/act'
```

and the server's log shows the model receiving a map where it expected an array:

```text
AttributeError: 'dict' object has no attribute 'mean'
```

### `MsgpackError: ... cannot be sent as msgpack`

The observation contains a type msgpack cannot encode. Send JSON-like values, bytes and numpy arrays; convert a set
to a list, and an object to a dict of its fields.

```text
MsgpackError: set cannot be sent as msgpack (JSON values, bytes and numpy arrays can)
```

### `MsgpackError: a numpy array of dtype object is not carried`

Give the array a numeric, bool or string dtype. An array of Python objects is refused on purpose: reading one back
would run code the sender chose.

```text
MsgpackError: a numpy array of dtype object is not carried: only plain numeric, bool and string dtypes are
```

### `MsgpackError: msgpack type 0xc1 is not read here`

The body is not msgpack that this reader takes: it is another format, it was cut short, or it uses a msgpack
extension type. Check the response's `content-type`. `connect()` reads `application/json` responses as JSON and
everything else as msgpack.

```text
MsgpackError: msgpack type 0xc1 is not read here (extension types and the unused code)
```

## With openpi and LeRobot

### `TypeError: this client keeps no WebSocket connection at _ws`

`openpi.keep()` takes openpi's own `WebsocketClientPolicy`, which holds its connection at `_ws`. Pass that object,
and not a wrapper around it such as an `ActionChunkBroker`.

```text
TypeError: this client keeps no WebSocket connection at `_ws` (openpi_client's WebsocketClientPolicy does): nothing to keep warm
```

### `ModuleNotFoundError: No module named 'openpi_client'`

Install openpi's client from openpi's repository (`packages/openpi-client`, `pip install -e .`).
`sequence_ai.network.openpi` adds to that client and does not replace it.

```text
ModuleNotFoundError: No module named 'openpi_client'
```

### `ModuleNotFoundError: No module named 'typing_extensions'` on importing `openpi_client`

`pip install typing_extensions`. openpi-client 0.1.0 imports it without listing it as a dependency, so a new
environment that contains only openpi's client may lack it; inside openpi's own environment it is already there.

```text
ModuleNotFoundError: No module named 'typing_extensions'
```

### My openpi client prints nothing and never returns

It is waiting for the server. While the connection is refused, openpi's client retries it every 5 seconds, and logs
`Waiting for server at ...` and `Still waiting for server...` at INFO level, which is hidden unless logging is set
up. So the server is not up yet, or the port is wrong. This is read from openpi's source
(`websocket_client_policy.py`); it was not made to happen here.

### `BBR on its socket: False`

Expected anywhere but Linux, and on Linux until the BBR module is loaded: `sudo modprobe tcp_bbr`. The keepalive
options and the ping while idle are on either way. On this Mac:

```text
BBR on its socket: False; TCP keepalive options accepted: 4
```

### Does the LeRobot setting need to be applied on both sides?

Yes, each side sends its own keep-alive frames. In LeRobot 0.7.0 the policy server reads it from `zenoh.config_file`
and the robot's remote inference from `zenoh_config_path`. Both names are from LeRobot's source; the file has not been
tested with LeRobot's own client and server. See
[How do I reduce the latency of LeRobot remote inference?](how-do-i-reduce-the-latency-of-lerobot-remote-inference.md)

## On the server

### `OSError: [Errno 48] port 8000 is taken: a server already answers at 127.0.0.1:8000 on this machine`

Another program has the port, often an earlier `serve()` still running: `lsof -nP -iTCP:8000 -sTCP:LISTEN` names it.
Stop it, or give the server and the robot another port:

```bash
python quickstart_server.py 127.0.0.1 8001
python quickstart_robot.py http://127.0.0.1:8001
```

`serve()` (sequence-ai 0.63.0) prints:

```text
OSError: [Errno 48] port 8000 is taken: a server already answers at 127.0.0.1:8000 on this machine (an earlier serve() still running?); stop it, or pass another port
```

A uvicorn server of your own prints:

```text
ERROR:    [Errno 48] error while attempting to bind on address ('127.0.0.1', 8000): [errno 48] address already in use
```

### `ERROR: the policy failed on the example (KeyError: 'camera'): stopping`

`serve()` ran the policy on `example` before saying it is ready, and the policy raised: the server stops, and the
traceback it raises is the policy's own. Give `example` the keys and shapes your robot sends, or fix the policy. With a
policy that reads a key the example does not have (sequence-ai 0.63.0):

```text
ERROR:    the policy failed on the example (KeyError: 'camera'): stopping
INFO:     Shutting down
...
KeyError: 'camera'
```

### My server's log fills with `GET /ping`

That is the robot keeping its connection warm: one small request for each 100 ms the connection is idle, for a
minute after the last request. `serve()` writes no line per request; a uvicorn server of your own wrote three in 0.35
seconds:

```text
INFO:     127.0.0.1:52970 - "GET /ping HTTP/1.1" 200 OK
INFO:     127.0.0.1:52970 - "GET /ping HTTP/1.1" 200 OK
INFO:     127.0.0.1:52970 - "GET /ping HTTP/1.1" 200 OK
```

To keep the pings and lose the lines, start uvicorn with `access_log=False`. To stop the pings, pass
`connect(url, ping_path=None)`; `ping_every_s` and `ping_for_s` change how often and for how long.

### Can the server ping instead, so the robot's code does not change?

Only over a WebSocket. A server that answers one HTTP request with one response cannot send anything unsolicited. See
[How do I keep a WebSocket policy server's connection warm?](how-do-i-keep-a-websocket-policy-servers-connection-warm.md)

## What a component reports about your machine

### `UserWarning: this machine restarts a connection's send window after each pause`

Do one of the two things it names, once, on the robot. Both need root. `connect(url, check=False)` disables the
warning without changing anything. The text below is the package's own, printed on a Mac with the Linux state
simulated; it is issued on Linux only.

```text
UserWarning: this machine restarts a connection's send window after each pause (net.ipv4.tcp_slow_start_after_idle is 1), so an upload of 0.3 MB sent after one takes several round trips. `sudo sysctl -w net.ipv4.tcp_slow_start_after_idle=0` keeps the window through pauses (a line in /etc/sysctl.d/ keeps it after a reboot), as does loading BBR (`sudo modprobe tcp_bbr`), which window.socket_options() then sets on your connections
```

[Why does my robot's upload get slow after a pause?](why-does-my-robots-upload-get-slow-after-a-pause.md) explains it.

### `Wi-Fi power saving is on for wlan0`

Run the command it gives on the robot. The text is the package's own, printed here for an interface named `wlan0`; it
is issued on Linux only, when the robot's default route is a Wi-Fi interface with power saving enabled.

```text
Wi-Fi power saving is on for wlan0, so its adapter can delay this robot's packets while it sleeps; `sudo iw dev wlan0 set power_save off` disables it until the next reboot, and `wifi.powersave = 2` under [connection] in a file in /etc/NetworkManager/conf.d/ keeps it disabled
```

### `window.keep(sock)` returns `False`, `tcp_link.read(sock)` and `wifi_power.check()` return `None`

Expected off Linux: those three read or set things only the Linux kernel has. Nothing fails, and the rest works. On
this Mac:

```text
window.keep(sock): False | window.congestion_of(sock): None
window.restarts_after_idle(): False | window.check(): False
tcp_link.read(sock): None
wifi_power.check(): None
```

On Linux, `window.keep(sock)` is `False` until the kernel's BBR module is loaded (`sudo modprobe tcp_bbr`).

## A network component made nothing faster

What a network component saves depends on the link:

- On round-trip times (RTTs) of about 101 and 241 ms, a robot sending a request every 0.53 s took no extra round trip
  after its pauses with the default settings, so there was nothing for BBR or a ping to save.
- On loopback, or between two machines on one switch, a round trip costs almost nothing, so saving one saves almost
  nothing.

The README has the [results](../README.md#results).
[Measure your own link](how-do-i-measure-the-link-between-my-robot-and-my-gpu.md) before and after.

## `cuda_graphs`

Lines that begin `cuda graphs:` are printed, not raised: the model keeps running, in eager mode where it was not
graphed.

### `cuda graphs: no CUDA GPU available: nothing captured`

This machine has no CUDA GPU that torch can use, so the model is unchanged. That is the expected line on a laptop.
On a machine that has a GPU, check `torch.cuda.is_available()`: a CPU-only build of torch prints the same line.

### `cuda graphs: action_head.denoiser is not in this model: not captured`

The path names no attribute of your model. Paths are dotted attribute names from the model you passed, as
`cuda_graphs.modules_of(model)` lists them. Use the census to get the path instead of writing it by hand:

```python
from sequence_ai.engine import cuda_graphs

census_user = cuda_graphs.Census(cuda_graphs.modules_of(model_user))
for _ in range(3):
    model_user(observation_user)
    census_user.end_act()
census_user.close()
print(census_user.candidates())        # [{'path': 'denoiser', 'calls': 10, 'ms': 1.05}]
```

The line is printed on a machine with no GPU too, so a mistyped path shows up on a laptop.

### `cuda graphs: ... is not a torch module: not captured`

The path names something that is not a `torch.nn.Module`: a function, a tensor, a list. Name the module that contains
it.

### `cuda graphs: ... runs in eager mode for one of its shapes: ...`

One shape of that module's inputs could not be graphed, and runs in eager mode from now on. The other shapes are
unaffected. The text after the colon gives the reason:

- `its capture failed (...)`: the module does something a CUDA graph cannot record, with torch's own error in
  parentheses. Naming a smaller module inside it often works.
- `its output is of a type a graph cannot return`: the module returns something other than tensors and plain
  values.
- `its graph's output differed from eager mode's`: the replay did not equal eager mode byte for byte, so it is not used.
  The check is working as intended, and that shape stays in eager mode.
- `cuda graphs need torch 2.4 or later`: older torch cannot swap the GPU's random generator while a capture runs,
  and a failed capture would corrupt your model's random noise. Nothing is captured. Upgrade torch.

### I installed it and nothing got faster

In order of likelihood:

1. **The module is called once per run.** The gain comes from the kernel launches across many calls. Run the census:
   `calls` is the number of calls per run.
2. **The model was not launch-bound.** A model whose GPU is already busy nearly all the time has no launch overhead
   to remove: π0.5 under JAX was 98% GPU-busy.
3. **Every call has a new shape.** Each new shape is captured once; a run of new shapes is a run of captures.
4. **It is running in eager mode.** Look for a `runs in eager mode` line above.

[How do I make a PyTorch robot policy faster without changing its outputs?](how-do-i-make-a-pytorch-robot-policy-faster-without-changing-its-outputs.md)

## `batcher`

### `ValueError: sizes [2, 4] must include 1: a call that waits alone is a batch of 1, and a batch in which the model raised is re-run call by call`

Include 1 in the sizes: `Batcher(run, sizes=[1, 2, 4])`. A call that arrives alone must be dispatched as a batch of
one.

### `ValueError: the model returned 3 outputs for a batch of 4`

Your function must return one output for each input, in order. It is given a list and returns a list of the same
length.

### My batches are all of size 1

Calls are batched only while they wait. With an idle model and `wait_ms=0`, each call is dispatched as soon as it
arrives. Raise `wait_ms` slightly (the examples use 2 ms), or expect batches only when the model is busy.

## `replicas`

### `TypeError: load must be a function a new process can import: defined at a module's top level, not a lambda and not a function inside another`

Each replica starts in its own process and imports `load_user` by name. Move it to the top of a module:

```python
def load_user():                       # at the top of a module, not inside main() and not a lambda
    model_user = build_model_user()
    return model_user.infer            # what one call runs
```

### `ReplicaStartFailed: the model's load raised: ...`

Your `load_user` raised in the replica's process; the text after the colon is its error, and the exception's `traceback`
is the replica's. Run `load()` once in a plain script to see it directly.

### `ReplicaStartFailed: the model did not load within 900 s`

The replica did not finish loading in time. For a model that takes longer: `ReplicaPool(load, start_timeout=1800)`.

### `ReplicaDied: the replica died mid-call (exit code -9)`

The process has exited: killed, or out of memory (an exit code of -9 is SIGKILL, which is also what the Linux
out-of-memory killer sends). The pool replaces it, and the call that was running gets this error. `ReplicaError`, by
contrast, is your model's own exception: the replica is alive and takes the next call.

### `ReplicaDied: the call exceeded its ... s deadline: the replica was terminated`

A call took longer than the deadline you gave it. The replica is terminated and replaced, since a model stuck in a call
cannot be interrupted any other way.

### `PoolHalted: ...`

The pool stopped replacing replicas after three consecutive deaths (`deaths_in_a_row`), or it was closed; the text
after the colon states which. A model that dies on every call is not reloaded indefinitely. Fix the cause, and create
a new pool.

### `ValueError: 1 replicas and 2 devices: one device per replica, at least one replica`

`devices` has one entry for each replica: `ReplicaPool(load, replicas=2, devices=[0, 1])`.

### Two replicas on one GPU are no faster

That is what was measured:
[Should I run two replicas of a policy on one GPU?](should-i-run-two-replicas-of-a-policy-on-one-gpu.md)

## `slack`

### `SizeFailed`

`slack.measure` could not run your forward pass at one of the batch sizes you listed, for example a batch that does
not fit in GPU memory. The exception carries the size, the timings measured so far, and your forward pass's own
error. List only the sizes that run.

### A replica serves only one robot

The policy leaves no slack: its robot requests the next chunk only when the current one is fully executed, so there
is no time in which to wait for a batch. See "Rules and requirements" in [`docs/slack.md`](slack.md#rules-and-requirements).

## General

### Does any of this change my policy's outputs?

No. `cuda_graphs` keeps a graph only when its output equals eager mode's byte for byte. `batcher` returns to each
call the output your model produced for it. `replicas` and `slack` do not modify the model. No network component
touches the model, and `compress` is lossless: the server gets back the bytes the robot built.
`frames.resize_image` and `frames.encode` do change pixels, which is their purpose, and only where you call them: the
resize mode and filter have no default, so you name the ones your policy was trained with. `openpi.image_tools` resizes
as openpi's own does and then sends JPEG, and a server started through `sequence_ai.network.openpi` hands the policy the
JPEG's pixels: at quality 95 they moved a π0.5 policy's actions by 5.9% on average over 50 real observations, about a
tenth of what the policy's own sampling noise moves them.

### Do I need a GPU to try it?

No. The tests and every example except `examples/check_on_a_gpu.py` run on a machine with no GPU.

### Does it work on Windows?

It is tested on Linux and macOS. `replicas` starts its processes with `spawn`, which is available on all three, but
nothing here is tested on Windows.

### Something else

Open an issue at <https://github.com/General-Sequences/sequence-ai/issues> with a minimal script that reproduces it,
the full error text and the output of `python -c "import sys, platform; print(sys.version, platform.platform())"`.
