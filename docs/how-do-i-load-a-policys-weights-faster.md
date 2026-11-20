# How do I load a policy's weights faster?

Pack the checkpoint once into one contiguous file, then load that file in a few long reads at once. Every tensor reads
back with the bytes it was packed with.

```python
from sequence_ai.engine import load

load.pack_from("checkpoint_user/", "model_user.load")                # once: safetensors, .pth or .pt, or an orbax folder
weights_user = load.load("model_user.load", device="cuda")            # {name: torch tensor}, on the GPU
params_user = load.load("model_user.load", device="cuda", to="jax")   # {name: JAX array}, on the GPU
```

`load.nest(params_user)` gives an orbax checkpoint's tree back, `{"a": {"b": ...}}`, from the names `"a/b"`.
`pack_from(..., dtype="bfloat16")` keeps a model's floats in the dtype it runs in, so a float32 checkpoint of a model
that runs in bfloat16 is half the bytes to read.

## What it saves

The first load after a machine starts, with nothing of the files in memory, measured on one L4 whose disk reads at
about 0.5 GB/s, against the way each model reads its checkpoint now:

| Checkpoint | Now | Packed |
|---|---|---|
| π0.5 LIBERO, orbax (12.4 GB in float32, packed in bfloat16), into JAX | 25.0 s | 16.0 s |
| GR00T N1.7 DROID, safetensors (6.9 GB) | 23.1 s | 13.3 s |
| Cosmos-Reason2-2B, safetensors (4.9 GB) | 16.3 s | 9.5 s |
| Wan2.2 VAE, .pth (2.8 GB) | 10.4 s | 5.6 s |

A load that finds the file already in memory is never slower than the checkpoint's own. A checkpoint read tensor by
tensor keeps a disk about half busy; one file read in parallel keeps it busy to its limit.

## How it reads

- **One file.** Its tensors are grouped by dtype into blocks that start at a multiple of 4096 bytes, after a header
  that names each tensor's block, place, dtype and shape.
- **Many reads at once.** Each block is read in 64 MiB pieces by 16 threads, each into a buffer of its own, and copied
  to the GPU while the next piece is read. The tensors are views of their block's memory: no copy after the read.
- **Through the page cache where memory holds the file**, so the next load finds it in memory. Where memory cannot hold
  what the file lacks there, the pieces already cached are read from memory and the others straight from the disk,
  past the cache, so the machine keeps what it had cached.
- **A JAX load** reads into memory and puts each array on the device, in the memory JAX takes for itself when it starts.

## In your policy

Load the packed file where your policy reads its checkpoint, and pack it the first time:

```python
import os

from sequence_ai.engine import load

packed_user = os.path.join(checkpoint_dir_user, "model.load")
if not os.path.exists(packed_user):
    load.pack_from(checkpoint_dir_user, packed_user)
model_user.load_state_dict(load.load(packed_user, device="cuda"))
```

The packed file holds the same bytes as the checkpoint, so it is one more copy of the weights on disk: keep both, or
only the packed file where nothing else reads the checkpoint.
