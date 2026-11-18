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

import numpy as np

import sequence_ai


def policy_user(observation_user: dict) -> dict:
    """Your model here: an observation in, a chunk of actions out. This one returns 8 actions of 7 numbers."""
    return {"actions": np.full((8, 7), float(observation_user["image"].mean()), dtype=np.float32)}


# an observation as your robot sends it: serve() runs the policy on it before it says it is ready
example_user = {"image": np.zeros((224, 224, 3), dtype=np.uint8), "state": np.zeros(7, dtype=np.float32)}

if __name__ == "__main__":
    host_user = sys.argv[1] if len(sys.argv) > 1 else None
    port_user = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    sequence_ai.serve(policy_user, host=host_user, port=port_user, example=example_user)
```

```bash
python quickstart_server.py
```

It prints:

```text
INFO:     serving policy_user on http://0.0.0.0:8000 and http://[::]:8000 (POST /act, GET /ping, GET /example); one observation per call; reads jpeg, zstd, zlib
WARNING:  no token: anyone who can reach port 8000 can run the policy (serve(token=...) or $SEQ_SERVE_TOKEN asks for one)
INFO:     Started server process [35969]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     warm-up: one observation took 0.6 ms, then 0.3 ms
INFO:     ready
```

It listens on every address of this machine, and ran the policy on `example_user` before it said `ready`. The warning is
about the token: [Put your own model and robot in](#put-your-own-model-and-robot-in) has what to do before a robot
depends on it.

## 3. Request actions

In a second terminal, with the same environment active, save this as `quickstart_robot.py`. It is
[`examples/quickstart_robot.py`](../examples/quickstart_robot.py).

```python
import sys

import numpy as np

import sequence_ai

url_user = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
frame_user = np.full((224, 224, 3), 128, dtype=np.uint8)        # your camera's frame here
state_user = np.zeros(7, dtype=np.float32)                      # your robot's joints here

with sequence_ai.connect(url_user) as link_user:                # BBR, TCP keepalive, a ping when idle, msgpack: all on
    for step_user in range(3):
        actions_user = link_user.act({"image": frame_user, "state": state_user})["actions"]
        print(f"request {step_user}: {actions_user.shape[0]} actions of {actions_user.shape[1]} numbers, "
              f"the first {actions_user[0, :3]}")
    timing_user = link_user.stats()["latency_us"]["act_us"]
    print(f"{timing_user['n']} requests, the slowest took {timing_user['max'] / 1000:.1f} ms")
```

```bash
python quickstart_robot.py
```

It prints:

```text
request 0: 8 actions of 7 numbers, the first [128. 128. 128.]
request 1: 8 actions of 7 numbers, the first [128. 128. 128.]
request 2: 8 actions of 7 numbers, the first [128. 128. 128.]
3 requests, the slowest took 2.6 ms
```

That is the first action chunk: 8 actions of 7 numbers. The stand-in `policy_user` returns the mean of the frame it was
sent, and the frame is all 128. On a Mac on Wi-Fi it also prints one warning first: Wi-Fi power saving can hold a
robot's packets back, and macOS gives nothing to check it with.

## 4. Check what is on

```bash
python -m sequence_ai.check http://127.0.0.1:8000
```

It prints:

```text
sequence-ai check: http://127.0.0.1:8000

  ok  server              sequence-ai 0.63.0, pid 35969, up 1 s, policy policy_user; it answers /ping with 200 and restores jpeg, zlib, zstd
  ok  round trip          1.5 ms for a ping on a kept connection (median of 10); 1.0 ms for a TCP connect
  ok  send window         macOS keeps a connection's send window through pauses
  --  BBR                 macOS has no BBR, and needs none for the window
  --  Wi-Fi power saving  not checked: macOS gives nothing to read it from; this machine's requests go over Wi-Fi (en0), and a wired connection takes the question away
  ok  pings when idle     connect() pings after each 100.0 ms of idle time, and the server answers them
  ok  pictures as JPEG    the server restores them, so connect() sends each HxWx3 uint8 array as JPEG: 1 in this observation, 150,658 bytes as arrays, 1,518 with them as JPEG
  ok  compression         the server restores zlib, zstd, so connect() compresses large arrays with zstd where that saves time
  --  pings in inference  each act took 1.0 ms, too short to tell
  ok  server time         the server says it took 0.3 ms of a 1.0 ms act: 0.7 ms outside it, on the network and in this machine
  ok  first act           1.4 ms, the next ones 1.0 ms (observation from GET /example); connect() waits up to 120 s for the first
```

Each `ok` is a part that is on; `--` is one this machine does without, or one this run could not tell (an act of
1 ms leaves no time to ping inside it); `no` would say what is off and why.
[What is on by default, and how do I turn each part off?](what-is-on-by-default.md) has each part.

## What just ran

On the robot, `connect()` did what your code would otherwise have to do:

| What | Why | Component |
|---|---|---|
| Kept one connection open for every request | A new connection for each call costs a handshake each time | [`link`](../sequence_ai/network/link.py) |
| Requested BBR and enabled TCP keepalive, where the machine allows | Linux restarts a connection's congestion window after a pause; BBR keeps it | [`window`](../sequence_ai/network/window.py) |
| Sent a small GET whenever the connection was idle for 100 ms | So the connection is never idle between requests | [`window`](../sequence_ai/network/window.py) |
| Sent the frame as JPEG once the server said it reads JPEG | Far fewer bytes: the check's frame, 224×224 of one color, took 1,518 bytes instead of 150,658 | [`frames`](../sequence_ai/network/frames.py) |
| Sent the rest as msgpack, its bytes as they are | JSON carries bytes as base64, a third larger on the wire | [`msgpack_io`](../sequence_ai/network/msgpack_io.py) |
| Timed every request | `link.stats()` gives p50, p95 and p99 once there are enough samples | [`latency_stats`](../sequence_ai/network/latency_stats.py) |

On the server, `serve()` ran the policy on a thread of its own, answered each ping at once, turned the JPEG back into
the array the policy reads, put each response in one write ([`one_write`](../sequence_ai/network/one_write.py)) and
told the robot how long the policy took (`Server-Timing`). It said what it restores (`X-Seq-Reads`) and what it is
(`X-Seq-Server`) on every response: that is how the robot knew to send JPEG.

On this machine, a Mac, BBR is not available, so the connection got TCP keepalive only. On loopback none of
these components saves time: there is no network between the two programs. What each one saves on a real link is in
the README's [results](../README.md#results).

## Put your own model and robot in

1. **Your model.** Replace the body of `policy_user` in `quickstart_server.py`. It takes the observation the robot sent
   and returns what the robot needs. A PyTorch model or an openpi policy goes in the same way: [How do I serve a robot
   policy from my own GPU?](how-do-i-serve-a-robot-policy-from-my-own-gpu.md#your-model-as-a-function)
2. **Your robot.** Replace `frame_user` and `state_user` in `quickstart_robot.py` with your camera's frame and your
   robot's joints, and send the keys your model reads. Put one real observation in `example_user` in the server, too.
3. **Two machines.** The server already listens on every address; on the robot, give the GPU's address:

   ```bash
   python quickstart_server.py                           # on the GPU
   python quickstart_robot.py http://YOUR_GPU:8000       # on the robot
   ```

   Run that way between two addresses of one machine, the three requests printed the same lines and the slowest took
   3.4 ms.
4. **A token.** Before a robot depends on it, start the server with one, `SEQ_SERVE_TOKEN=... python
   quickstart_server.py`, and connect with `sequence_ai.connect(url, token=...)`. Open the port only to your
   robot as well.
5. **Measure the link.**
   [How do I measure the link between my robot and my GPU?](how-do-i-measure-the-link-between-my-robot-and-my-gpu.md)
   tells you which components help on it.

## Where to go next

- [How do I serve a robot policy from my own GPU?](how-do-i-serve-a-robot-policy-from-my-own-gpu.md)
- [How do I run π0 (openpi) on a remote GPU?](how-do-i-run-pi0-openpi-on-a-remote-gpu.md)
- [How do I reduce the latency of LeRobot remote inference?](how-do-i-reduce-the-latency-of-lerobot-remote-inference.md)
- [How do I get a GPU near my robot?](how-do-i-get-a-gpu-near-my-robot.md)
- A step failed: [FAQ](faq.md)
