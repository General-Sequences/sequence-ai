# Quickstart: from `pip install` to a batched response

Two commands and one short file give you five robots sending requests concurrently, their requests batched, and the
model running in its own process. No GPU, no model and no torch are needed for these steps. The output shown is
from the package built from this repository at 0.62.0, on macOS on Apple silicon with Python 3.12; Python 3.10 or later
behaves the same way.

**How long it takes.** The install took 2 seconds with the packages already downloaded, and the file ran in 0.3 seconds.

## 1. Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install sequence-ai
```

The last line it prints:

```text
Successfully installed annotated-types-0.8.0 anyio-4.16.0 certifi-2026.10.23 h11-0.16.0 httpcore-1.0.9 httpx-0.28.1 idna-3.20 pillow-13.0.0 pydantic-2.14.3 pydantic-core-2.50.4 sequence-ai-0.62.0 sequence-base-0.8.12 typing-extensions-4.16.0 typing-inspection-0.4.5
```

The inference components import only the standard library. `optimize` operates on a torch model, or a JAX one, and
uses the torch or JAX your model already has.

## 2. Run five robots against one model

Save this as `quickstart_batching.py`. It is [`examples/quickstart_batching.py`](../examples/quickstart_batching.py).

```python
import asyncio
import os
import time

from sequence_ai.engine.batcher import Batcher
from sequence_ai.engine.replicas import ReplicaPool


def load_user():
    """Runs in the replica's own process: build your model here, and return what one batch of requests runs."""
    def model_on_a_list_user(observations_user: list) -> list:
        time.sleep(0.05)                                   # one forward pass, whatever the batch: your model's call
        return [{"actions": [[observation_user] * 7] * 8, "pid": os.getpid()} for observation_user in observations_user]

    return model_on_a_list_user


async def main_user() -> None:
    pool_user = ReplicaPool(load_user, replicas=1)        # one replica per GPU; devices=[0] sets the GPU it sees
    batcher_user = Batcher(pool_user.run, sizes=[1, 2, 4, 8], wait_ms=2.0)
    try:
        started_user = time.perf_counter()
        # in your own server, each request's handler awaits batcher(observation)
        outputs_user = await asyncio.gather(*(batcher_user(float(robot_user)) for robot_user in range(5)))
        took_user = (time.perf_counter() - started_user) * 1e3
        print(f"5 robots served in {took_user:.0f} ms, as batches of {dict(batcher_user.sent)}")
        actions_user = outputs_user[3]["actions"]
        print(f"robot 3's own response: {len(actions_user)} actions of {len(actions_user[0])} numbers, "
              f"the first {actions_user[0][:3]}")
        print(f"this process is {os.getpid()}; the model ran in {outputs_user[3]['pid']}")
    finally:
        pool_user.close()


if __name__ == "__main__":
    asyncio.run(main_user())
```

```bash
python quickstart_batching.py
```

It prints (your process IDs will differ):

```text
5 robots served in 113 ms, as batches of {4: 1, 1: 1}
robot 3's own response: 8 actions of 7 numbers, the first [3.0, 3.0, 3.0]
this process is 73513; the model ran in 73515
```

## What just ran

| What | Why | Component |
|---|---|---|
| Five requests went to the model as a batch of 4 and a batch of 1: two forward passes of 50 ms instead of five | Calls waiting for a model are served in one forward pass; only the sizes you list are sent, because a model is traced, compiled or graphed for the shapes it has seen | [`batcher`](../sequence_ai/engine/batcher.py) |
| Robot 3 received its own response, not its batch's | A batch's outputs are split back per call | [`batcher`](../sequence_ai/engine/batcher.py) |
| The model ran in another process | A forward pass never blocks your server's event loop, and a replica that dies is replaced | [`replicas`](../sequence_ai/engine/replicas.py) |

**The 113 ms is the stand-in model's.** Its forward pass sleeps 50 ms regardless of batch size. What batching gains on
a real model depends on where the model spends its time:
[How many robots can one GPU serve?](how-many-robots-can-one-gpu-serve.md)

## Use your own model

1. **Your model.** In `load_user`, build your model and return a function that takes a list of observations and returns
   a list of outputs, one per observation. `load_user` is defined at the top of a module, because each replica imports
   it in its own process.
2. **Your server.** In each request's handler, `output = await batcher(observation)`. Any async server works; the
   components are independent of HTTP.
3. **Your GPU.** `ReplicaPool(load, replicas=2, devices=[0, 1])` gives each replica its own GPU. On one GPU use one
   replica: [Should I run two replicas of a policy on one GPU?](should-i-run-two-replicas-of-a-policy-on-one-gpu.md)
4. **Your batch sizes.** List the sizes your model has been run with, and always include 1.

## Then: a faster forward pass, with the same outputs

If your model is a PyTorch model that calls one module many times per run (a diffusion or flow policy's denoiser),
one more call, `optimize()`, replays that module from CUDA graphs. With torch installed:

```bash
python examples/graph_a_module.py
```

On a machine with no CUDA GPU it leaves the model unchanged, and prints:

```text
cuda_graphs: off - this machine has no CUDA GPU
whole_graph: off - this machine has no CUDA GPU
same outputs, byte for byte: True
```

On a GPU it captures the module, checks the outputs against eager mode's byte for byte, keeps what is faster, and
prints the act before and after. [How do
I make a PyTorch robot policy faster without changing its
outputs?](how-do-i-make-a-pytorch-robot-policy-faster-without-changing-its-outputs.md)

## Where to go next

- [How many robots can one GPU serve?](how-many-robots-can-one-gpu-serve.md)
- [How do I use the inference components with openpi, LeRobot or
  GR00T?](how-do-i-use-the-inference-components-with-openpi-lerobot-or-gr00t.md)
- [How do I get a GPU near my robot?](how-do-i-get-a-gpu-near-my-robot.md)
- A step failed: [FAQ](faq.md)
