# How do I use the inference components with openpi, LeRobot or GR00T?

Keep the model and the server you have, and add the component that fits where your model spends its time. Which
component applies depends on the framework the policy runs in, not on its name:

| Your policy | `optimize` | `batcher` and `slack` | `replicas` |
|---|---|---|---|
| A PyTorch policy with a denoising loop: GR00T, LeRobot's diffusion and flow policies, or your own | Applies: CUDA graphs of what repeats, measured on GR00T N1.7 | Apply where your model takes a batch | One replica per GPU |
| openpi's models under JAX (π0, π0.5) | Tried when you ask, `cuda_graphs=True, whole_graph=True`: XLA already compiles each jitted function whole, and π0.5 keeps its GPU 98% busy | Apply. Measured on π0.5's batched call, and `slack` in a test server around it | One replica per GPU |

What follows is where each component goes and what was measured on the models themselves.

## GR00T, and other PyTorch policies: a faster forward pass

```python
import sequence_ai

# infer_user(observation) -> its output: your policy's call, as your code makes it
fast_user = sequence_ai.engine.optimize(infer_user, examples_user)    # 32 or more of your robot's observations
print(fast_user.report["on"])                                          # for example ["cuda_graphs", "whole_graph"]
```

`serve(infer_user, ...)` runs the same during its warm-up. Measured with `seq serve`: GR00T N1.7 on an L4, its diffusion
transformer and the action encoder and decoder graphed, acts from 149.9 ms to 117.5 ms, with exactly the same outputs. The report says which modules were
graphed and why any switch stayed off, and
[`examples/check_on_a_gpu.py`](../examples/check_on_a_gpu.py) compares every output with eager mode's on your GPU. [How do I make a PyTorch robot policy faster without changing its
outputs?](how-do-i-make-a-pytorch-robot-policy-faster-without-changing-its-outputs.md)

## openpi and LeRobot: more than one robot on a GPU

openpi's policy server calls the model once for each request, with no batching across connections, and LeRobot 0.7's
policy server holds one robot session (both verified in their source code:
[the comparison page](comparison.md)). To
serve several robots in one forward pass, the requests must reach the model as a batch. With `seq serve` that is one
decorator, `@seq.concurrent`, and for an openpi policy `batched(policy)` of
[`examples/serve_openpi_batched.py`](../examples/serve_openpi_batched.py) is the batched call:

```python
from openpi.policies import droid_policy

import seq
from serve_openpi_batched import batched     # the example's file, copied beside your policy


@seq.policy(hz=15, action_chunk_valid_len=15, action_chunk_replan_at=8)
class Pi05DroidBatched_user:
    def __init__(self):
        self.batched_user = batched(load_policy_user())            # openpi's infer() for a list of observations

    @seq.sample
    def sample_user(self):
        return [("observe", droid_policy.make_droid_example()), ("act",)] * 3

    @seq.concurrent(sizes=[1, 2, 4], collate=False)
    @seq.infer
    def infer_user(self, observations_user):                        # the acts that run together, as a list
        return self.batched_user(observations_user)                 # one sampling call, an answer for each
```

`load_policy_user()` makes openpi's trained policy from its downloaded checkpoint, as
[How do I run π0 (openpi) on a remote GPU?](how-do-i-run-pi0-openpi-on-a-remote-gpu.md)
shows.

In a server of your own, a handler around the model's batched call does the same:

```python
from sequence_ai.engine.batcher import Batcher

# infer_a_batch_user(observations: list) -> list
batcher_user = Batcher(infer_a_batch_user, sizes=[1, 2, 4], wait_ms=2.0)
output_user = await batcher_user(observation_user)  # in each connection's handler
```

`infer_a_batch_user` stacks the observations, calls the model once, and splits the output. For an openpi policy that is
each observation through the policy's input transform, the results stacked, one call of the model's `sample_actions`,
and each row through the output transform: the example's `batched(policy)` does exactly that. Under JAX each batch size
compiles on its first call: a policy compiled on its first inference can take tens of seconds, so `@seq.sample` runs it
before the server says it is ready.

A server that holds a batch by each robot's slack runs its own loop with `slack` in place of `Batcher`:
[How many robots can one GPU serve?](how-many-robots-can-one-gpu-serve.md#measured) has what one GPU answered in time.

## Any of them: the model off the server's event loop

```python
from sequence_ai.engine.replicas import ReplicaPool

pool_user = ReplicaPool(load_user, replicas=1)                     # load_user() builds the policy in its own process
output_user = await pool_user.run(observation_user)
```

Run one replica per GPU: one copy of the model answers all its robots in one pass.

## The connection to the robot

The inference components run on the server. For the link between the robot and the GPU, serve the policy with
`seq serve` and connect the robot with `sequence_ai.connect()`, and [what is on by default](what-is-on-by-default.md) is
on that connection ([How do I run π0 (openpi) on a remote GPU?](how-do-i-run-pi0-openpi-on-a-remote-gpu.md)).

## When it applies

- `optimize`'s graphs apply to PyTorch models; a JAX model compiles its own kernels, and gets XLA's command buffers.
