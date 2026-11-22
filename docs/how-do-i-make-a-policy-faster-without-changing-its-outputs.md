# How do I make a policy faster without changing its outputs, and what does each switch change?

You do nothing: `seq serve` tries each speed-up that keeps a PyTorch policy's outputs exactly the same while it warms
the policy up on its `@seq.sample` episode, checks it on those observations, and keeps it only where acts are at least
10% faster. The speed-ups that change the outputs are yours to turn on, each with the difference you accept.

## The switches

| Switch | With `seq serve` | On by default | Changes the outputs | What it does |
|---|---|---|---|---|
| `cuda_graphs` | `--no-cuda-graphs` leaves it out | yes, on a PyTorch policy | no | Replays each module the model calls more than once per run with the same shapes, a denoiser for example, from a CUDA graph of its own kernels. |
| `whole_graph` | `--no-whole-graph` leaves it out | yes, on a PyTorch policy | no | Replays a run's whole call into the model, its denoising loop and the noise it draws included, as one graph where it can be captured. Where it cannot, `cuda_graphs` alone. |
| `torch_compile` | `--torch-compile` | no | yes | Compiles the modules the act spends its time in with `torch.compile`. |
| `fused_kernels` | `--fused-kernels` | no | yes | Runs each norm, gated activation and rotary embedding in those modules as one kernel. |
| `tensorrt` | `--tensorrt` | no | yes | Builds those modules with TensorRT, TF32 off. Install the torch-tensorrt made for your torch: `pip install "torch-tensorrt==2.9.*"` for torch 2.9. |
| `precision` | `--precision fp16`, `fp32` or `fp8` | no: the model's own dtype | yes | `fp16` or `fp32` runs those modules in that dtype. `fp8` runs their linear layers in 8-bit floating point, on a GPU of compute capability 8.9 or later (L4, H100). |
| `cfg_parallel` | `--cfg-parallel`, with `seq serve` started as two processes: `torchrun --nproc-per-node 2 -m seq serve ...` | no | yes | Splits a model's classifier-free guidance between the two processes, each on a GPU of one type: each call of a module the policy makes with the guidance's two halves in one batch runs one half in each process, and the two come back as one answer. |
| `pipeline_parallel` | `--pipeline-parallel`, with `seq serve` started as two processes or more: `torchrun --nproc-per-node 2 -m seq serve ...` | no | yes | Splits the blocks the model calls one after another, a transformer's layers, into one stage a process, each with as many blocks as its GPU runs in the time the others run theirs, so GPUs of different types can share them. Each call goes through the stages in parts, so they work at once: a diffusion transformer's tokens in patches, each patch's attention reading the other patches' keys and values from the step before, or rows of a batch of several robots. |

## A switch that changes the outputs needs a tolerance

`--tolerance` is the largest difference you accept between an output with the switch on and the model's own. It is a
number, the largest absolute difference of any value, or limits on `max_abs`, `mean_abs` and `cosine` (the least
cosine similarity):

```bash
seq serve my_policy_user --precision fp8 --tolerance max_abs=0.02,cosine=0.999
```

A switch that changes the outputs, turned on without a tolerance, is refused before anything runs.

## How each switch is checked

- **The same inputs and the same noise.** The policy runs on each observation of its `@seq.sample` episode with a
  fixed random state, its noise included, and again to see that it gives the same outputs. A value that changes from
  one run to the next, such as a timing the policy reports, is left out of every comparison.
- **One switch at a time.** Each is turned on, the policy runs on every observation again, and its outputs are compared
  with the model's own: exactly, byte for byte, for `cuda_graphs` and `whole_graph`, and within your `--tolerance` for
  the others.
- **Timed as it will run.** Acts are timed with and without the switch, in turns, and the switch stays on only when
  acts are at least 10% faster. Otherwise it is taken out and the model is as it was. A switch that changes the outputs
  is timed with `cuda_graphs` on in both turns, because it will run under the graphs, which take away the time each
  kernel takes to launch: on GR00T N1.6 on an L4, `fp8` under the graphs took acts from 170 ms to 136 ms. Its outputs
  are still compared with the model's own, without the graphs.
- **The graphs last.** The switches that change the outputs are tried first, and `cuda_graphs` and `whole_graph` are
  then captured from what runs and compared exactly with it.

With fewer than 8 observations in the episode, each is run again with other random states until there are 8 runs;
an episode of 32 or more of your robot's own observations checks more.

## Reading what it kept

`seq serve` prints one line for each switch as it is decided. On GR00T N1.7 (the gr00t-droid template, built as
`seq serve` builds it) on an L4, `cuda_graphs` said, its times the median of the acts it timed in turns:

```text
optimize: cuda_graphs on - infer._policy.model.action_head.model (4 calls a run), infer._policy.model.action_head.action_encoder (4 calls a run), infer._policy.model.action_head.action_decoder (4 calls a run): the same outputs, bit for bit, on 8 runs; act 149.9 → 117.5 ms
```

A switch that is off says why. On the same model, `whole_graph` said of the action head's whole call
`infer._policy.model.action_head.get_action: its graph's output differed from eager mode's`, and stayed off. A switch
that changes the outputs also gives the largest differences it found: `--precision fp16` there gave a `max_abs` of
0.0797659, a `mean_abs` of 0.0110621 and a `cosine` of 0.999861. A server's `/ping` lists the switches on, and
`python -m sequence_ai.check http://<gpu>:8000` prints them on the robot.

## A JAX policy

A policy whose model `jax.jit` compiles gets each switch's counterpart in XLA:

| Switch | What it does on a JAX policy |
|---|---|
| `cuda_graphs` | Compiles each jitted function with XLA's command buffers, its CUDA graphs, around even the shortest stretches of its work. XLA puts its longer ones in them already. |
| `whole_graph` | Puts its loops, a denoising loop for example, in the command buffers too. |
| `precision` | Casts the parameters its jitted functions are given to `fp16` or `fp32`. |
| `torch_compile`, `fused_kernels` | Nothing to add: XLA compiles each jitted function whole and fuses its kernels already. |
| `tensorrt` | No counterpart: TensorRT builds torch modules. |
| `cfg_parallel`, `pipeline_parallel` | No counterpart: they split the calls of torch modules. |

XLA already puts most of a jitted function's work in command buffers, so on a JAX policy these two are tried only when
you turn them on: `seq serve --cuda-graphs --whole-graph`. The checks are the same. The random key the policy holds is
set from each run's seed, so two runs of one observation draw the same noise.

## When a switch stays off

- On a machine with no CUDA GPU, nothing is captured or built, and its line says so.
- A policy whose outputs change from one run to the next with the same random state cannot be shown to keep them
  exactly, so `cuda_graphs` and `whole_graph` stay off and their lines say which output changed.
- A module that is called once per run, or whose shapes change within a run, has no graph to replay many times;
  `whole_graph` can still cover the run.
- `cfg_parallel` needs `seq serve` started as two processes, and a module the policy calls with the guidance's two
  halves in one even batch: a policy that runs its guidance as two calls has nothing to split.
- `pipeline_parallel` needs `seq serve` started as two processes or more, and a list of blocks (a `ModuleList` or
  `Sequential`) the model calls one after another, each on the one before's output. Its patches need a transformer's
  tokens as the blocks' input and a self-attention without a mask that runs through
  `torch.nn.functional.scaled_dot_product_attention`; parts of a batch make one robot's act no faster. Each process
  holds the whole model, so it makes a policy faster and does not fit one larger than a GPU.
- `fp8` needs a GPU of compute capability 8.9 or later. A linear layer whose output moves more than 5% in fp8 keeps
  its own precision, and the choice is kept in `~/.cache/sequence-ai/fp8` for the next start.

[How do I make a PyTorch robot policy faster without changing its outputs?](how-do-i-make-a-pytorch-robot-policy-faster-without-changing-its-outputs.md)
says why its graphs are faster on a PyTorch policy, and how they compare with `torch.compile`.
