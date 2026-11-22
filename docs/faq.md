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

- [`optimize`](#optimize)
- [`batcher`](#batcher)
- [`replicas`](#replicas)
- [`slack`](#slack)

[General](#general)

## Installing

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

### `ImportError: resizing a frame requires Pillow, which is not installed`

Pillow is a dependency of `sequence-ai`. The error appears where `frames.py` was copied into a project that has no
Pillow: `pip install pillow`. `frames.resize_image`, `frames.encode` and `frames.decode` use Pillow, and
`openpi.image_tools` and `openpi.patch()` through them; no other component does.

```text
ImportError: resizing a frame requires Pillow, which is not installed: pip install pillow
ImportError: encoding an array or PIL image requires Pillow, which is missing or broken: pip install pillow. Pass a base64 str or encoded bytes to avoid it.
```

The first line is what `frames.resize_image` raises, the second what `frames.encode` raises.

### What do I install?

| You use | Install |
|---|---|
| `connect()` on the robot, `frames`, `window`, `ping`, `tcp_link`, `one_write` under uvicorn, `wifi_power`, `msgpack_io`, `latency_stats`, `lerobot`, every inference component, numpy arrays in observations or responses | `sequence-ai`, nothing more |
| `seq serve`, and the quick start's server | `sequence-ai`, on Python 3.10 or later |
| `sequence_ai.Video`, `Depth` and `Tactile`, and pictures sent as video keyframes, `connect()`'s default | `sequence-ai`, on Python 3.10 or later; on 3.9, pictures go as they are |
| `bench/`, the measuring tool | a checkout of the repository; `client.py` and `server.py` are standard library only, `ws_bench.py` needs `websockets` and `zenoh_bench.py` `eclipse-zenoh` |

Python 3.9 or later on the robot; 3.10 or later for `seq serve` and for video.

## Connecting

### `Unavailable: ... had no answer from http://...:8000 (nothing listens there, or this machine's network does not reach it)`

`wait_until_ready()` reached nothing at that address before its bound. Check, in this order:

1. The server is running, and its log shows the port you gave `connect()`.
2. The server listens on an address the robot can reach. `127.0.0.1` is reachable from the same machine only.
   `seq serve` listens on every address unless it is given `--host`.
3. The URL starts with `http://`, has the port, and its host name resolves: `connect("http://YOUR_GPU:8000")`. A
   misspelled name gives the same line.
4. The GPU's firewall lets the robot's address in on that port: a closed one drops the robot's packets, and nothing
   answers.

```text
sequence_ai.errors.Unavailable: 127.0.0.1-8000 had no answer from http://127.0.0.1:8000 (nothing listens there, or this machine's network does not reach it) after 3s (wait_until_ready timeout_s)
```

Without `wait_until_ready()`, the first act meets the same thing as it opens the robot's stream:

```text
sequence_ai.errors.Unavailable: worker unreachable after 5 reconnect attempts: the stream to the worker did not open: the worker could not be reached (1 try in 0.0 s, the last in its TCP connect) (hint: a robot reaches its worker on one WebSocket (wss): check that this machine's network lets it through)
```

## Sending and reading

### `AuthError: the server at ... refused this robot's ping (401)`

The server was started with a token, and the robot sent none or another. Connect with the server's token:
`sequence_ai.connect(url, token="...")`, the one `seq serve` was given with `--token` or as `SEQ_SERVE_TOKEN` on the
GPU. `wait_until_ready()` raises it at once: asking again changes nothing.

```text
sequence_ai.errors.AuthError: the server at http://127.0.0.1:8000 refused this robot's ping (401) (hint: it asks for its token: connect(url, token=...))
```

### `the policy raised ...`

The policy raised on the observation the act was answered from. The server's log has the traceback, and the server
goes on answering the next acts. A built-in exception, such as `KeyError`, comes back as itself; any other as
`PolicyError`.

```text
KeyError: "the policy raised KeyError: 'camera' (the server's log has the traceback)"
```

### `PolicyError: the policy's answer cannot go back as msgpack: ...`

The policy returned something msgpack cannot carry. Return JSON-like values, bytes and numpy arrays; torch tensors and
JAX arrays are turned into numpy arrays for you.

```text
sequence_ai.errors.PolicyError: the policy's answer cannot go back as msgpack: object cannot be sent as msgpack (JSON values, bytes and numpy arrays can)
```

### `InvalidRequest: observe(): this observation cannot be sent - ...`

The observation contains something msgpack cannot carry, and nothing was sent. Send JSON-like values, bytes and numpy
arrays: convert a set to a list, and an object to a dict of its fields. An array of Python objects (dtype object) is
refused on purpose: reading one back would run code the sender chose.

```text
sequence_ai.errors.InvalidRequest: observe(): this observation cannot be sent - set cannot be sent as msgpack (JSON values, bytes and numpy arrays can)
sequence_ai.errors.InvalidRequest: observe(): this observation cannot be sent - a numpy array of dtype object is not carried: only plain numeric, bool and string dtypes are
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

### `error[conflict]: port 8000 is taken: a server already answers at 127.0.0.1:8000 on this machine`

Another program has the port, often an earlier `seq serve` still running: `lsof -nP -iTCP:8000 -sTCP:LISTEN` names it.
Stop it, or give the server and the robot another port:

```bash
seq serve --port 8001                                    # in the policy's folder
python quickstart_robot.py http://127.0.0.1:8001
```

`seq serve` prints, and exits with 5:

```text
error[conflict]: port 8000 is taken: a server already answers at 127.0.0.1:8000 on this machine (an earlier seq serve still running?); stop it, or give seq serve another --port
```

A uvicorn server of your own prints:

```text
ERROR:    [Errno 48] error while attempting to bind on address ('127.0.0.1', 8000): [errno 48] address already in use
```

### `ERROR: the policy failed on the example (KeyError: 'camera'): stopping`

`seq serve` ran the policy on its sample's first observation before saying it is ready, and the policy raised: the
server stops, and the traceback it raises is the policy's own. Give `@seq.sample` the keys and shapes your robot sends,
or fix the policy. With a policy that reads a key its sample does not have (sequence-ai 0.64.1):

```text
ERROR:    the policy failed on the example (KeyError: 'camera'): stopping
INFO:     Shutting down
...
KeyError: 'camera'
```

### Can the server ping instead, so the robot's code does not change?

Only over a WebSocket. A server that answers one HTTP request with one response cannot send anything unsolicited. See
[How do I keep a WebSocket policy server's connection warm?](how-do-i-keep-a-websocket-policy-servers-connection-warm.md)

## What a component reports about your machine

### `UserWarning: this machine restarts a connection's send window after each pause`

Do one of the two things it names, once, on the robot. Both need root. `warnings.filterwarnings("ignore", "this machine restarts")`
silences the warning without changing anything. The text below is the package's own, printed on a Mac with the Linux state
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

## `optimize`

`seq serve` tries the speed-ups during the warm-up and prints one line for each switch it tries. A switch that stays
off says why; the model keeps running as it was.

### `optimize: nothing turned on - this machine has no CUDA GPU`

The switches that keep the outputs exactly the same need a CUDA GPU that torch, or JAX, can use. That is the expected
line on a laptop. On a machine that has a GPU, check `torch.cuda.is_available()`: a CPU-only build of torch prints the
same line.

### `optimize: nothing turned on - no observation to check the outputs on: give examples=, or the policy a @seq.sample method`

The checks run the policy on the observations of its `@seq.sample` episode. Give it one, preferably of 32 or more of
your robot's own observations.

### `optimize: cuda_graphs off - no module is called more than once a run with the same shapes`

A module called once per run has no launches to save across calls, so no module was graphed. `whole_graph` can still
cover the run's whole call; its line says whether it did.

### `optimize: cuda_graphs off - its outputs differed from eager mode's on run ...`

A graph replayed an output that is not eager mode's byte for byte, so it was taken out and the model runs as it did.
The check is working as intended.

### `optimize: whole_graph off - <call>: the policy raised with it on: ...`

The policy's own code refused its call once it was replayed from a graph, with its own error after the colon: code
that reads a value back from the GPU in the middle of a run cannot be captured whole. The modules `cuda_graphs` graphs
are unaffected.

### `optimize: cuda_graphs off - 1% faster (act ... ms), under the 10% it needs`

A switch stays on only when acts are at least 10% faster with it, so it was taken out and the model runs as it did. A
switch that changes the outputs is timed as it will run, with `cuda_graphs` on in both turns, and the times on its
line say `with cuda_graphs on in both`.

### `ValueError: torch_compile changes the policy's outputs: give tolerance=...`

`torch_compile`, `fused_kernels`, `tensorrt`, `precision`, `cfg_parallel` and `pipeline_parallel` change the
outputs, so each runs only with the difference you accept: `seq serve --tolerance 1e-3`, or
`--tolerance max_abs=0.02,cosine=0.999`.

### Every switch stayed off: no faster

In order of likelihood:

1. **The model was not launch-bound.** A model whose GPU is already busy nearly all the time has no launch overhead
   to remove: π0.5 under JAX was 98% GPU-busy.
2. **Every call has a new shape.** Each new shape is captured once; a run of new shapes is a run of captures.
3. **The sample episode was not the robot's.** Make the `@seq.sample` episode's observations as your robot sends them,
   so the shapes checked are the shapes served.

[How do I make a policy faster without changing its outputs, and what does each switch
change?](how-do-i-make-a-policy-faster-without-changing-its-outputs.md)

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

### One replica per GPU

One replica uses the whole card. To serve more robots from it, batch their requests (`@seq.concurrent`) rather than
start a second replica.

## `slack`

### `SizeFailed`

`slack.measure` could not run your forward pass at one of the batch sizes you listed, for example a batch that does
not fit in GPU memory. The exception carries the size, the timings measured so far, and your forward pass's own
error. List only the sizes that run.

### A replica serves only one robot

The policy leaves no slack: its robot requests the next chunk only when the current one is fully executed, so there
is no time in which to wait for a batch. See "Rules and requirements" in [`docs/slack.md`](slack.md#rules-and-requirements).

### `503 ... this server cannot answer this robot within the 350 ms left of each answer`

The server was started with `hz` and seats only the robots it answers in time, each by its own round trip. The message
names what this robot's acts spend outside the server and what a batch takes there. Serve the robot from a GPU nearer
it, or act again once another robot has stopped: a robot gives its place up a few seconds after its last act. `GET
/ping` shows the robots seated, the seats, and each batch size's time.

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
