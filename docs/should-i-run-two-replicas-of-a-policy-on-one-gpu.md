# Should I run two replicas of a policy on one GPU?

No. Two replicas on one GPU (two loaded copies of the model, each in its own process) do not run concurrently: the
GPU time-slices between the two processes. Measured with π0.5 on an L4, two replicas whose calls overlapped each took
2.1 times as long as one alone, so the GPU's throughput was no higher than with one replica. Give each replica its
own GPU, and on one GPU use one replica and batch its calls.

```python
from sequence_ai.engine.replicas import ReplicaPool

pool = ReplicaPool(load, replicas=2, devices=[0, 1])    # one replica per GPU; load() runs in the replica's process
output = await pool.run(observation)                     # in a request's handler: off the event loop
pool.close()
```

## What was measured

| Setting, π0.5 on one L4 | Each call took |
|---|---|
| One replica | 1 (the baseline) |
| Two replicas, their calls overlapping | 2.1 times as long each |
| Two replicas with the CUDA Multi-Process Service (MPS) enabled | 6 to 9% faster than without it |

One model and one GPU. A model that leaves the GPU idle for most of a call could behave differently; this one did
not.

## What to do on one GPU

- **Batch.** One replica serving several robots in one forward pass:
  [How many robots can one GPU serve?](how-many-robots-can-one-gpu-serve.md)
- **Make the forward pass shorter**, where the model is PyTorch and calls a module many times per run:
  [How do I make a PyTorch robot policy faster without changing its outputs?](how-do-i-make-a-pytorch-robot-policy-faster-without-changing-its-outputs.md)

## What `replicas` is for

- **One replica on each of several GPUs**, each running one call at a time: `devices=[0, 1]`.
- **A model that never blocks your server.** Each replica runs in its own process, so a forward pass does not block
  the server's event loop or hold its interpreter, even with one replica.
- **A replica that dies is replaced.** `ReplicaError` is the model's own error, and its replica takes the next call.
  `ReplicaDied` is a process that has exited: killed, out of memory, or past its deadline. After a few consecutive
  deaths the pool stops replacing, and raises `PoolHalted` once none is left.

`Batcher(pool.run, runners=2)` combines the two components: batches are formed, two run concurrently, one per replica.
[`examples/serve_with_replicas.py`](../examples/serve_with_replicas.py) does it around a small model.

## One requirement

`load` must be a function a new process can import: defined at the top of a module, not a lambda and not a function
inside another. Each replica starts in its own process and imports it by name.
