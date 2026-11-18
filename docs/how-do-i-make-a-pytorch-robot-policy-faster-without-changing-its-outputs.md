# How do I make a PyTorch robot policy faster without changing its outputs?

Replay the module your policy calls many times per run, its denoiser, from CUDA graphs captured from its own eager
kernels. `cuda_graphs.install` does that in one call, and keeps a graph only when its output equals eager mode's byte
for byte. On GR00T N1.7 on an L4 this reduced a run from 224 ms to 146 ms (1.54 times faster) with exactly the same
actions. It gains nothing on a model that already fuses and graphs its kernels.

```python
from sequence_ai.engine import cuda_graphs

cuda_graphs.install(model_user, ["action_head.denoiser"])    # model.action_head.denoiser, graphed in place
model_user(observation_user)                                       # a new shape: run in eager mode, captured, checked
model_user(observation_user)                                       # replayed
```

Nothing else in your code changes. The module stays where it is in your model, the same object of the same class,
with its weights, its `state_dict` and its hooks unchanged. Only its `forward` is replaced.

## Why this is faster

A vision-language-action policy with a diffusion or flow action head calls its denoiser several times per inference.
In eager PyTorch at batch 1, each of those calls launches many small kernels one by one, and launch overhead
dominates: on GR00T N1.7 on an L4 the GPU was busy 41% of the time. A CUDA graph replays the same kernels as one
launch.

## Why the outputs do not change

A graph replays the same kernels that eager mode ran, in the same order, so its arithmetic is identical. A compiler
that fuses kernels changes the order of the arithmetic, and with it the outputs. The three were measured on the same
model and GPU:

| Approach | Speed | Outputs |
|---|---|---|
| `torch.compile(backend="cudagraphs")` | 1.00× (225 ms) | unchanged |
| `torch.compile`, inductor | 1.45× | changed: 0.0051 apart on average |
| `cuda_graphs` | 1.54× (224 ms to 146 ms) | exactly the same, on inputs it was not captured on |

GR00T N1.7, one L4, its DiT captured. One model and one GPU: **these are not numbers for your model.**

`cuda_graphs` verifies the equality instead of assuming it. Each new shape is run in eager mode once, captured, and
replayed on the same inputs; the graph is kept only if the two outputs are equal byte for byte. A shape whose capture
fails or differs runs in eager mode from then on, which is reported once.

## Which module to name

One that is called more than once per run with the same shapes. To find it, count the calls over a few runs:

```python
from sequence_ai.engine import cuda_graphs

census_user = cuda_graphs.Census(cuda_graphs.modules_of(model_user))
for _ in range(3):
    model_user(observation_user)                                   # one inference, as your code makes it
    census_user.end_act()
census_user.close()
paths_user = [row_user["path"] for row_user in census_user.candidates()]    # the outermost and slowest first
cuda_graphs.install(model_user, paths_user[:1])
```

[`examples/graph_a_module.py`](../examples/graph_a_module.py) does all of it on a small flow-style model, and runs on
a machine with no GPU, where it prints which module it would graph and leaves the model unchanged.

## Check it on your GPU

```python
import torch
from sequence_ai.engine import cuda_graphs

torch.manual_seed(0)                                     # a policy that draws noise: the same noise for each run
eager_user = model_user(observation_user)
cuda_graphs.install(model_user, paths_user[:1])
torch.manual_seed(0)
model_user(observation_user)  # a new shape: captured, and checked against eager mode here
torch.manual_seed(0)
print(cuda_graphs.same(eager_user, model_user(observation_user)))       # True: a replayed run, byte for byte
cuda_graphs.ACTIVE = False                               # the same model in eager mode again, to compare timings
```

[`examples/check_on_a_gpu.py`](../examples/check_on_a_gpu.py) runs the capture on a real GPU and compares every
output with eager mode's. On a T4, its small flow-style model went from 16.1 ms to 2.4 ms per run: byte for byte on
inputs the capture never saw, the model's random noise unchanged, a new batch size captured automatically.

## What it is safe against

- **A graph whose output differs is never used.**
- **A failed capture does not change your model's random noise.** The GPU's random generator is swapped for a clone
  while a capture runs, and restored afterwards.
- **What your model checks about its module still holds:** `isinstance`, its type, its methods and its hooks, which
  fire around a replayed call too.

## When it applies

- **A PyTorch model that launches many small kernels.** A model that already fuses and compiles its kernels, such as
  π0.5 under JAX at 98% GPU-busy, has no launch overhead to remove.
- **A module called many times per run.** A module called once has one launch sequence to save.
- **Shapes that repeat.** Each new shape is captured and checked once, so a module gains from a shape's second call
  on. A prompt of a new length is a new shape, captured the first time it occurs.
- **A CUDA GPU.** Without one, nothing is graphed, which is reported once, and the model runs as before.
- **torch 2.4 or later,** which lets a capture swap the GPU's random generator; with an earlier torch, nothing is
  captured.
