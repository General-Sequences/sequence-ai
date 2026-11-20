# How do I make a PyTorch robot policy faster without changing its outputs?

Replay the module your policy calls many times per run, its denoiser, from CUDA graphs captured from its own eager
kernels. `seq serve` does that by default while it warms the policy up: it finds the modules, captures them, and keeps
the graphs only when the outputs on its `@seq.sample` episode equal eager mode's byte for byte and acts are at least
10% faster. On GR00T N1.7 on an L4, graphing its diffusion transformer reduced a run from 224 ms to 146 ms
(1.54 times faster) with exactly the same actions. It gains nothing on a model that already fuses and graphs its
kernels.

```bash
seq serve my_policy_user     # its warm-up prints a line for each switch: on or off, the modules, why
```

Nothing else in your code changes. Each graphed module stays where it is in your model, the same object of the same
class, with its weights, its `state_dict` and its hooks unchanged. Only its `forward` is replaced.
[How do I make a policy faster without changing its outputs, and what does each switch change?](how-do-i-make-a-policy-faster-without-changing-its-outputs.md)
has every switch and what its line says.

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

GR00T N1.7 on one L4, its DiT captured.

`seq serve` verifies the equality instead of assuming it. It runs the `@seq.sample` episode's observations with fixed
random states, in eager mode and then with the graphs, and keeps them only if every output is equal byte for byte. After
that, each new shape is run in eager mode once, captured, and replayed on the same inputs; its graph is kept only if the
two outputs are equal. A shape whose capture fails or differs runs in eager mode from then on, which is reported once.

## Which modules it graphs

Each module called more than once per run with the same shapes. `seq serve` counts the calls each module gets while
the episode runs, graphs those that repeat, the outermost and slowest first, and then tries `whole_graph`: the
run's whole call into the model as one graph, its denoising loop and the noise it draws included. Its `cuda_graphs`
line names the modules, and its `whole_graph` line the call.

[`examples/graph_a_module.py`](../examples/graph_a_module.py) runs it on a small flow-style model, and runs on a
machine with no GPU, where it prints that nothing was captured and leaves the model unchanged.

## Check it on your GPU

Its warm-up lines are the check: each switch that is on was compared with eager mode on every observation, byte for
byte, and timed against it. A server's `/ping` lists the switches on, and
`python -m sequence_ai.check http://<gpu>:8000` prints them on the robot.

[`examples/check_on_a_gpu.py`](../examples/check_on_a_gpu.py) runs `optimize()` on a real GPU and compares every output
with eager mode's: on inputs it never saw, the model's random noise unchanged, a new batch size captured on its first
call. On an L4, its small flow-style model went from 12.09 ms to 1.22 ms per run, with every check passing.

## What it is safe against

- **A graph whose output differs is never used.**
- **A failed capture does not change your model's random noise.** Each capture is tried on a clone of the GPU's
  random generator, and the generator is put back as it was if a capture fails.
- **What your model checks about its module still holds:** `isinstance`, its type, its methods and its hooks, which
  fire around a replayed call too.

## When it applies

- **A PyTorch model that launches many small kernels.** A model that already fuses and compiles its kernels, such as
  π0.5 under JAX at 98% GPU-busy, has no launch overhead to remove.
- **A module called many times per run.** A module called once has one launch sequence to save.
- **Shapes that repeat.** Each new shape is captured and checked once, so a module gains from a shape's second call
  on. A prompt of a new length is a new shape, captured the first time it occurs.
- **A CUDA GPU.** Without one, nothing is graphed, the report says so, and the model runs as before.
- **torch 2.4 or later,** which lets a capture swap the GPU's random generator; with an earlier torch, nothing is
  captured.
