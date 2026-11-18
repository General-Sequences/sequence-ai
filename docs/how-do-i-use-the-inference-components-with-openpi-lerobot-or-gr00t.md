# How do I use the inference components with openpi, LeRobot or GR00T?

Keep the model and the server you have, and add the component that fits where your model spends its time. Which
component applies depends on the framework the policy runs in, not on its name:

| Your policy | `cuda_graphs` | `batcher` and `slack` | `replicas` |
|---|---|---|---|
| A PyTorch policy with a denoising loop: GR00T, LeRobot's diffusion and flow policies, or your own | Applies. Measured on GR00T N1.7 only | Apply where your model takes a batch | One replica per GPU |
| openpi's models under JAX (π0, π0.5) | Does not apply: it is for PyTorch, and π0.5 under JAX was already 98% GPU-busy | Apply. Measured on π0.5's batched call, and `slack` in a test server around it | One replica per GPU. Measured: two on one GPU gained nothing |

What follows is where each component goes and what was measured on the models themselves.

## GR00T, and other PyTorch policies: a faster forward pass

```python
from sequence_ai.engine import cuda_graphs

census_user = cuda_graphs.Census(cuda_graphs.modules_of(policy_user))
for _ in range(3):
    policy_answer_user = run_policy_user(policy_user, observation_user)      # one inference, as your code makes it
    census_user.end_act()
census_user.close()
# the module called many times per run comes first
paths_user = [row_user["path"] for row_user in census_user.candidates()]
cuda_graphs.install(policy_user, paths_user[:1])  # from here on, replayed and checked byte for byte
```

Measured: GR00T N1.7 on an L4, its DiT captured, 224 ms to 146 ms per run, with exactly the same outputs on inputs it
was not captured on. The census shows in three runs whether your model has a module worth graphing, and
[`examples/check_on_a_gpu.py`](../examples/check_on_a_gpu.py) shows how to compare every output with eager mode's on
your GPU. [How do I make a PyTorch robot policy faster without changing its
outputs?](how-do-i-make-a-pytorch-robot-policy-faster-without-changing-its-outputs.md)

## openpi and LeRobot: more than one robot on a GPU

openpi's policy server calls the model once for each request, with no batching across connections, and LeRobot 0.7's
policy server holds one robot session (both verified in their source code:
[the comparison page](comparison.md)). To
serve several robots in one forward pass, the requests must reach the model as a batch. With `serve()` that is one
argument, and for an openpi policy `openpi.batched(policy)` is the batched call:

```python
import sequence_ai
from sequence_ai.engine import openpi

sequence_ai.serve(openpi.batched(policy_user), batch=[1, 2, 4], example=observation_user)
```

In a server of your own, a handler around the model's batched call does the same:

```python
from sequence_ai.engine.batcher import Batcher

# infer_a_batch_user(observations: list) -> list
batcher_user = Batcher(infer_a_batch_user, sizes=[1, 2, 4], wait_ms=2.0)
output_user = await batcher_user(observation_user)  # in each connection's handler
```

`infer_a_batch_user` stacks the observations, calls the model once, and splits the output. For an openpi policy that is
each observation through the policy's input transform, the results stacked, one call of the model's `sample_actions`,
and each row through the output transform: `openpi.batched(policy)` does exactly that. Measured on π0.5 on an L4 with
the model's own batched call: 5.99 requests per second at batch 1 and 10.66 at batch 16. That model's time is mostly in
its prefix, which batching does not amortize, so the gain levels off near 1.78 times. Through the transforms, batches of
1, 2, 4 and 8 took 195, 312, 540 and 1,027 ms, 1.52 times at batch 8, and each size compiled for 32 to 43 s on its first
call.

A server that holds a batch by each robot's slack runs its own loop with `slack` in place of `Batcher`. That loop was
run around π0.5's batched call with openpi's own client: the capacity `slack` computed, 2 robots on one L4 for a
15 Hz policy, had no late response, and 3 robots had 21 to 26%. Batching added no capacity on that model and GPU:
openpi's own server, one request at a time, served the same 2 robots. For SmolVLA the formula `slack` then used for
the load gave too high a capacity, and the capacity is computed on a clock since:
[Why the capacity is computed on a clock](how-many-robots-can-one-gpu-serve.md#why-the-capacity-is-computed-on-a-clock).
[How many robots can one GPU serve?](how-many-robots-can-one-gpu-serve.md#checked-on-a-real-model)

## Any of them: the model off the server's event loop

```python
from sequence_ai.engine.replicas import ReplicaPool

pool_user = ReplicaPool(load_user, replicas=1)                     # load_user() builds the policy in its own process
output_user = await pool_user.run(observation_user)
```

Run one replica per GPU: one copy of the model answers all its robots in one pass.

## The connection to the robot

The inference components run on the server. For the link between the robot and the GPU, with openpi's own client or
LeRobot's remote inference kept as they are, the network layer adds one line on openpi's client ([How do I run π0
(openpi) on a remote GPU?](how-do-i-run-pi0-openpi-on-a-remote-gpu.md)) or one Zenoh setting for LeRobot ([How do I
reduce the latency of LeRobot remote inference?](how-do-i-reduce-the-latency-of-lerobot-remote-inference.md)).

## When it applies

- `cuda_graphs` applies to PyTorch models; a JAX model compiles its own kernels.
