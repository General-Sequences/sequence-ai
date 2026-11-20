# How do I serve a robot policy from my own GPU?

Write the policy as a class, serve it with `seq serve`, and call it from the robot with `connect()`. Everything here is
open source and runs on a GPU you already have: no account and no service of ours is involved.

On the GPU, `policy.py` in a folder of its own:

```python
import seq


@seq.policy(hz=15, action_chunk_valid_len=8, action_chunk_replan_at=5)
class MyPolicy_user:
    def __init__(self):
        self.model_user = load_model_user()                       # your model, loaded once

    @seq.sample
    def sample_user(self):
        return [("observe", observation_user), ("act",)] * 3     # an episode as your robot plays it

    @seq.infer
    def infer_user(self, observation_user):                       # the latest observation in, an action chunk out
        return {"actions": self.model_user(observation_user["image"], observation_user["state"])}
```

```bash
seq serve                                                         # in the policy's folder: port 8000, every address
```

On the robot:

```python
import sequence_ai

policy_user = sequence_ai.connect("http://YOUR_GPU:8000")
policy_user.wait_until_ready()                                    # once, before the robot moves
policy_user.observe({"image": frame_user, "state": state_user})
actions_user = policy_user.act()["actions"]
```

Both sides' defaults are on: the robot sends pictures as JPEG and compresses large arrays where that saves time, keeps
its one connection warm with pings, and waits longer for the first answer; the server runs the policy on a thread of
its own, answers pings at once, warms the policy up on its sample, and sends torch and JAX outputs back as numpy arrays.
[What is on by default, and how do I turn each part off?](what-is-on-by-default.md) lists every part with its switch,
and `python -m sequence_ai.check http://YOUR_GPU:8000` says which are on between your robot and your server.

The [Quickstart](quickstart.md) runs both sides on one machine first, with the robot's own loop and the output of each
step. Run between two addresses of one machine, its robot printed:

```text
the first answer: 8 steps of 7 joints, the first [128. 128. 128.]
49 steps played at 15 a second; 8 answers, the slowest in 6.3 ms
```

## Your model, in a class

`__init__` loads the model once. `@seq.infer` is called with the observation the robot sent, a dict with numpy arrays as
arrays, and returns what the robot gets back: a dict, an array or a list, with numpy arrays, torch tensors or JAX arrays
in it. `@seq.sample` is an episode as your robot plays it, observations and acts in order: the server warms the model up
on it before it says it is ready, so the first robot does not wait for a model compiled on its first inference.

Return an action chunk, its first dimension time: the steps the robot plays while it asks for the next one. A policy
that returns one step at a time, a network round trip away, leaves a 30 Hz arm waiting for each step: OpenVLA, which
returns one step, held a simulated arm still for all 1,200 steps of a 40 s run. `@seq.policy` says how the chunk plays:
`hz` steps a second, the first `action_chunk_valid_len` steps of each, the next asked for at `action_chunk_replan_at`.

A PyTorch module, LeRobot's, GR00T's or your own, is called in `@seq.infer` on the inputs it takes, as
[`examples/pytorch_policy/policy.py`](../examples/pytorch_policy/policy.py) does:

```python
import torch

import seq


@seq.policy(hz=15, action_chunk_valid_len=8, action_chunk_replan_at=5)
class TorchPolicy_user:
    def __init__(self):
        self.model_user = load_model_user().to(DEVICE_user).eval()

    @seq.sample
    def sample_user(self):
        return [("observe", observation_user), ("act",)] * 3

    @seq.infer
    @torch.inference_mode()
    def infer_user(self, observation_user):
        image_user = torch.as_tensor(observation_user["image"], device=DEVICE_user).permute(2, 0, 1)[None].float() / 255
        state_user = torch.as_tensor(observation_user["state"], device=DEVICE_user)[None]
        return {"actions": self.model_user(image_user, state_user)[0]}   # a tensor: it reaches the robot as numpy
```

An openpi policy, its checkpoint downloaded by openpi's own `download.maybe_download` (once, into `~/.cache/openpi`, from
the address openpi's README gives), at the numbers its DROID example plays:

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
    def sample_user(self):
        return [("observe", droid_policy.make_droid_example()), ("act",)] * 3

    @seq.infer
    def infer_user(self, observation_user):
        return self.policy_user.infer(observation_user)
```

**Batching.** `@seq.concurrent(sizes=[1, 2, 4, 8])` on the `@seq.infer` method hands it the observations of the acts
that wait while the policy runs, together: each array stacked along a new first axis, in the largest listed size they
fill. It returns one batch, a row an act, and each robot gets its own row back. List the sizes your model is compiled
or graphed for: a batch of an unlisted size is never sent to it. Batching serves several robots in fewer forward
passes; one robot's request is no faster for it.
[How many robots can one GPU serve?](how-many-robots-can-one-gpu-serve.md) has what it gained on one model.

A server or a client you already run keeps its code: its parts are one file each, in
[Parts for a server you already run](parts-for-a-server-you-already-run.md).

## Which components help on your link

A component saves time only on a link that has the cost it removes. Measure before you add:

| On your link | Add | Measured |
|---|---|---|
| A short round-trip time (round-trip time (RTT), about 21 ms) and pauses between requests | `connect()`, or `window` on your own client | 40 of 40 requests took one extra round trip without BBR, none with it |
| A longer RTT (about 101 or 241 ms), no loss | nothing for the pauses | 0 or 1 of 40 requests took an extra round trip with the default settings |
| A tunnel or proxy in front of the server | `one_write` | GET p50 189.7 ms with headers and body written apart, 106.7 ms in one write |
| Large raw arrays on a slow uplink | compression, on by default with `seq serve` | 172,800-byte frames came out about 1.9 times smaller for 0.37 ms |
| A new connection for each call | keep one open, as `connect()` does | 305.9 ms per call with a new connection, 121.0 ms reusing one |

The conditions behind each row are in the [README](../README.md#results).
[How do I measure the link between my robot and my GPU?](how-do-i-measure-the-link-between-my-robot-and-my-gpu.md)
shows how to get the same numbers for your own link.

## Before a robot depends on it

- **A token.** `seq serve` with no token runs the policy for anyone who can reach its port, and says so when it starts.
  Start it with one, `SEQ_SERVE_TOKEN=... seq serve`, and connect with `sequence_ai.connect(url, token="...")`; open
  the port only to your robot's address as well.
- **The first answer.** A model compiled on its first inference takes tens of seconds. `@seq.sample` runs it before
  the server says it is ready, `wait_until_ready()` waits for that on the robot, and `connect()` gives the first act
  120 s in any case.
- **Several robots on one GPU.** `@seq.concurrent(sizes=[1, 2, 4, 8])`, above. One replica of a model per GPU:
  [How many robots can one GPU serve?](how-many-robots-can-one-gpu-serve.md)
- **No GPU near the robot.** [How do I get a GPU near my robot?](how-do-i-get-a-gpu-near-my-robot.md)

## Plans and episodes

`@seq.plan`, `@seq.observe` and `@seq.on_episode` run on `seq serve` as on a deployment: each robot's episodes are its
own, `on_episode` gets `phase="end"` when one ends, and a plan never holds up an act.

## See also

- [What is on by default, and how do I turn each part off?](what-is-on-by-default.md)
- [Quickstart](quickstart.md)
- [FAQ](faq.md), for the errors a first run can print
- [Comparison with openpi's and LeRobot's remote inference](comparison.md)
