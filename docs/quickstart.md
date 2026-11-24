# Quickstart: a policy and a robot

Two short files and two commands give you a policy served from your GPU and a robot that plays its answers, asking for
the next one before it runs out. No GPU and no model are needed. The output shown is from macOS on Apple silicon with
Python 3.12; Linux behaves the same way.

## 1. Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install sequence-ai
```

The package brings httpx, which `connect()` uses for its connection, uvicorn, which runs the server, numpy and
zstandard for arrays, Pillow and (on Python 3.10 or later) PyAV for pictures, and what the `seq` command needs (pydantic and `sequence-base`).

## 2. Write a policy

Save this as `policy.py`, in a folder of its own. It is
[`examples/quickstart_policy/policy.py`](../examples/quickstart_policy/policy.py).

```python
import numpy as np
import seq


@seq.policy(hz=15, action_chunk_valid_len=8, action_chunk_replan_at=5)
class QuickstartPolicy_user:
    @seq.sample
    def sample_user(self):
        observation_user = {"image": np.zeros((224, 224, 3), dtype=np.uint8), "state": np.zeros(7, dtype=np.float32)}
        return [("observe", observation_user), ("act",)] * 3          # an observation, then an act, three times

    @seq.infer
    def infer_user(self, observation_user):
        frame_user = observation_user["image"]                       # your model here: the next 10 steps, 7 joints each
        return {"actions": np.full((10, 7), float(frame_user.mean()), dtype=np.float32)}
```

- **`@seq.policy`** says how its answers play: 15 steps a second (`hz`), the first 8 steps of each answer
  (`action_chunk_valid_len`), and the next answer asked for at step 5 (`action_chunk_replan_at`), so that it arrives
  before the 8 are played.
- **`@seq.sample`** is an episode as your robot plays it: observations and acts, in order. The server warms the model up
  on its first observation before it says it is ready; without a sample, the first robot's act is the model's first.
- **`@seq.infer`** answers the latest observation with a chunk of actions. Its first dimension is time: row `i` is the
  step to play `i` steps after the observation it answers. Without the decorator, a method named `infer` is used.

## 3. Serve it

In the policy's folder:

```bash
seq serve
```

It prints:

```text
serving quickstart-policy-user on port 8000, each robot held and seated by its time at 15 Hz: a robot connects with sequence_ai.connect("http://<this machine>:8000")
INFO:     serving QuickstartPolicy_user.infer_user on http://0.0.0.0:8000 and http://[::]:8000 (GET /stream, POST /observe, POST /act, GET /ping, GET /example, GET /record); one observation per call
INFO:     holds and seats each robot by its own time: 15 Hz, 8 valid steps, the next asked for at step 5
WARNING:  no token: anyone who can reach port 8000 can run the policy (seq serve --token TOKEN or $SEQ_SERVE_TOKEN asks for one)
INFO:     Started server process [68489]
INFO:     Waiting for application startup.
INFO:     optimize: nothing turned on - the policy holds no torch module
INFO:     Application startup complete.
INFO:     warm-up: one observation took 0.3 ms, then 0.3 ms at the median of 3
INFO:     ready
```

It listens on every address of this machine, and ran the policy on the sample's first observation before it said
`ready`. The warning is about the token: [Put your own model and robot in](#put-your-own-model-and-robot-in) has what
to do before a robot depends on it.

## 4. Run the robot's loop

In a second terminal, with the same environment active, save this as `quickstart_robot.py`. It is
[`examples/quickstart_robot.py`](../examples/quickstart_robot.py).

```python
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import sequence_ai

HZ_USER = 15            # the policy's hz: the steps a second
VALID_USER = 8          # its action_chunk_valid_len: the steps of an answer the robot plays
REPLAN_AT_USER = 5      # its action_chunk_replan_at: the step at which the robot asks for the next answer

url_user = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"


def observation_now_user() -> dict:
    """Your robot here: its camera's frame and its joints."""
    return {"image": np.full((224, 224, 3), 128, dtype=np.uint8), "state": np.zeros(7, dtype=np.float32)}


def move_user(step_user: np.ndarray) -> None:
    """Your robot here: one step's 7 joint targets."""


def hold_user() -> None:
    """Your robot here: hold still where it is."""


def ask_user(policy_user) -> tuple:
    """The next answer's valid steps, and when it was asked for."""
    asked_user = time.monotonic()
    return policy_user.act()["actions"][:VALID_USER], asked_user


with sequence_ai.connect(url_user) as policy_user, ThreadPoolExecutor(1) as background_user:
    policy_user.wait_until_ready()                                         # once, before the robot moves
    policy_user.observe(observation_now_user())
    chunk_user, asked_user = ask_user(policy_user)
    print(f"the first answer: {len(chunk_user)} steps of {chunk_user.shape[1]} joints, the first {chunk_user[0, :3]}")
    played_user = 0
    while played_user < 3 * HZ_USER:                                       # 3 s of steps
        started_user, next_user = time.monotonic(), None
        for i_user, step_user in enumerate(chunk_user):
            if i_user == REPLAN_AT_USER:
                next_user = background_user.submit(ask_user, policy_user)  # 1. ask early, in the background
            move_user(step_user)
            played_user += 1
            policy_user.observe(observation_now_user())                    # 2. the newest observation, each step
            time.sleep(max(0.0, started_user + (i_user + 1) / HZ_USER - time.monotonic()))   # 3. on to the next step
        try:
            chunk_user, asked_user = (next_user or background_user.submit(ask_user, policy_user)).result()
            chunk_user = chunk_user[int((time.monotonic() - asked_user) * HZ_USER):]       # 4. skip the steps passed
        except Exception:                          # 5. a dropped link, a restarted server, the policy raising:
            hold_user()                            #    hold still,
            policy_user.wait_until_ready()         #    wait until the server answers again,
            policy_user.observe(observation_now_user())
            chunk_user, asked_user = ask_user(policy_user)                                  #    and go on from here
    acts_user = policy_user.stats()["latency_us"]["act_us"]
    print(f"{played_user} steps played at {HZ_USER} a second; {acts_user['n']} answers, the slowest in "
          f"{acts_user['max'] / 1000:.1f} ms")
```

```bash
python quickstart_robot.py
```

It prints:

```text
the first answer: 8 steps of 7 joints, the first [128. 128. 128.]
49 steps played at 15 a second; 8 answers, the slowest in 53.6 ms
```

The loop is yours, and its five steps keep the robot moving at 15 steps a second, and still when it cannot:

1. **It asks early, in the background.** At step 5 of the 8 it asks for the next answer on a thread of its own, so the
   answer comes while it plays the last three.
2. **It hands over the newest observation each step**, so the next answer starts from where the robot is.
3. **It sleeps to the next step** rather than for a fixed time, so a slow `move_user` does not slow the steps down.
4. **It skips the steps that passed while the answer was on its way.** An answer's row `i` is for `i` steps after the
   observation it answered, so the rows the robot has already lived through are dropped.
5. **It holds still while the server cannot answer.** A dropped link, a restarted server or the policy raising makes
   the answer raise: the loop holds the arm, waits until the server answers again, and goes on from a fresh answer.
   The connection comes back by itself; there is nothing to connect again. With the server stopped for 2 seconds in
   the middle of a 10-second run, the loop held once, at step 35, and played on when the server was back.

The stand-in policy returns the mean of the frame it was sent, and the frame is all 128. On Linux, with Wi-Fi power
saving on, it also prints one warning first: power saving can hold a robot's packets back.

## 5. Check what is on

```bash
python -m sequence_ai.check http://127.0.0.1:8000
```

It prints:

```text
sequence-ai check: http://127.0.0.1:8000

  ok  server              sequence-ai 0.66.0, pid 68489, up 9 s, policy QuickstartPolicy_user.infer_user, no speed-up on; it answers /ping with 200 and takes the robot's stream (x-seq-stream/3)
  ok  round trip          0.3 ms for a ping on a kept connection (median of 10); 0.0 ms for a TCP connect
  ok  send window         macOS keeps a connection's send window through pauses
  --  BBR                 macOS has no BBR, and needs none for the window
  --  Wi-Fi power saving  not checked: macOS gives nothing to read it from; this machine's requests go over Wi-Fi (en0), and a wired connection takes the question away
  ok  pings when idle     connect() pings after each 100.0 ms of idle time, and the server answers /ping with 200: the pings go on the robot's stream, from its first observation until 60 s after its last observation or act, and keep the connection, and the Wi-Fi radio, from going idle
  ok  pings in inference  answered while the model ran: 0.4 ms against 0.3 ms idle (3 pings in 3 acts)
  ok  server time         the server says it took 0.3 ms of a 1.7 ms act: 1.4 ms outside it, on the network and in this machine
  ok  first act           108.3 ms, the next ones 1.7 ms (observation from GET /example)
```

Each `ok` is a part that is on; `--` is one this machine does without, or one this run could not tell; `no` would say
what is off and why.
[What is on by default, and how do I turn each part off?](what-is-on-by-default.md) has each part.

## What just ran

On the robot, `connect()` did what your code would otherwise have to do:

| What | Why | Component |
|---|---|---|
| Kept one stream open: each observation sent as it was made, each act after them | A new connection for each call costs a handshake each time | [`connect`](../sequence_ai/__init__.py) |
| Requested BBR and enabled TCP keepalive, where the machine allows | Linux restarts a connection's congestion window after a pause; BBR keeps it | [`window`](../sequence_ai/network/window.py) |
| Pinged the stream whenever it was idle for 100 ms | So the connection is never idle between requests | [`window`](../sequence_ai/network/window.py) |
| Sent the frame as a video keyframe of its own | Far fewer bytes: the quickstart's frame, 224×224 of one color, took 834 bytes for its first frame and 247 after it, instead of 150,576 | [`markers`](../sequence_ai/markers.py) |
| Sent the rest as msgpack, its bytes as they are | JSON carries bytes as base64, a third larger on the wire | [`msgpack_io`](../sequence_ai/network/msgpack_io.py) |
| Timed every act | `policy.stats()` gives p50, p95 and p99 once there are enough samples | [`latency_stats`](../sequence_ai/network/latency_stats.py) |

On the server, `seq serve` built the class, warmed it up on its sample, and ran it on a thread of its own; it answered
each ping at once, turned the keyframe back into the array the policy reads, put each response in one write
([`one_write`](../sequence_ai/network/one_write.py)) and told the robot in each answer how long it took (`perf`). It
names the robot's stream (`X-Seq-Act-Formats`) and what it is (`X-Seq-Server`) on every response.

On this machine, a Mac, BBR is not available, so the connection got TCP keepalive only. On loopback none of
these components saves time: there is no network between the two programs. What each one saves on a real link is in
the README's [results](../README.md#results).

## Put your own model and robot in

1. **Your model.** Load it in the class's `__init__` and call it in `infer_user`, which takes the observation the robot
   sent and returns what the robot needs, its first dimension time. A PyTorch model or an openpi policy goes in the
   same way: [How do I serve a robot policy from my own GPU?](how-do-i-serve-a-robot-policy-from-my-own-gpu.md)
2. **Your robot.** Fill in `observation_now_user`, `move_user` and `hold_user` in `quickstart_robot.py`, send the keys
   your model reads, and put one real observation in the sample.
3. **Its numbers.** Set `hz`, `action_chunk_valid_len` and `action_chunk_replan_at` to your robot's, and the same three
   at the top of the loop.
4. **Two machines.** The server already listens on every address; on the robot, give the GPU's address:

   ```bash
   seq serve                                             # on the GPU, in the policy's folder
   python quickstart_robot.py http://YOUR_GPU:8000       # on the robot
   ```
5. **A token.** Before a robot depends on it, start the server with one, `SEQ_SERVE_TOKEN=... seq serve`, and connect
   with `sequence_ai.connect(url, token=...)`. Open the port only to your robot as well.
6. **Measure the link.**
   [How do I measure the link between my robot and my GPU?](how-do-i-measure-the-link-between-my-robot-and-my-gpu.md)
   tells you which components help on it.

## Where to go next

- [How do I serve a robot policy from my own GPU?](how-do-i-serve-a-robot-policy-from-my-own-gpu.md)
- [How do I run π0 (openpi) on a remote GPU?](how-do-i-run-pi0-openpi-on-a-remote-gpu.md)
- [How do I reduce the latency of LeRobot remote inference?](how-do-i-reduce-the-latency-of-lerobot-remote-inference.md)
- [How do I get a GPU near my robot?](how-do-i-get-a-gpu-near-my-robot.md)
- A step failed: [FAQ](faq.md)
