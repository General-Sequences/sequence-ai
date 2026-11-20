# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
A model served by two replicas, each in its own process, with the calls waiting for it batched and the model made
faster by optimize() where there is a GPU: three components together, around a small model defined here.

    pip install sequence-ai torch
    python examples/serve_with_replicas.py

In your own server, `batcher(observation)` is what each request's handler awaits.
"""
from __future__ import annotations

import asyncio
import os
import time

from sequence_ai.engine.batcher import Batcher
from sequence_ai.engine.replicas import ReplicaPool


def load():
    """Runs in each replica's own process: build the model there, and return what a batch of calls runs."""
    import torch

    from sequence_ai.engine import optimize

    class Denoiser(torch.nn.Module):
        def __init__(self, width: int = 256) -> None:
            super().__init__()
            self.net = torch.nn.Sequential(torch.nn.Linear(width, width), torch.nn.GELU(),
                                           torch.nn.Linear(width, width))

        def forward(self, x, t):
            return x - 0.1 * self.net(x + t)

    class ActionHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.denoiser = Denoiser()

        @torch.no_grad()
        def forward(self, noise):
            x = noise
            for step in range(10):
                x = self.denoiser(x, torch.full_like(x[:, :1], step / 10))
            return x

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ActionHead().to(device).eval()
    batches = [torch.zeros(size, 256, device=device) for size in (1, 2, 4, 8)]        # the batch sizes it will get
    optimize(model, batches, say=lambda line: None)        # on a GPU, faster where the outputs stay the same

    def run(observations: list) -> list:
        """One forward pass for the whole batch; each call gets its own row back, with the replica that ran it."""
        batch = torch.tensor(observations, dtype=torch.float32, device=device)
        actions = model(batch).cpu().tolist()
        return [{"action": row[:3], "pid": os.getpid(), "batch": len(observations)} for row in actions]

    return run


async def main() -> None:
    pool = ReplicaPool(load, replicas=2)                       # two replicas; devices=[0, 1] gives each its own GPU
    batcher = Batcher(pool.run, sizes=[1, 2, 4, 8], wait_ms=2.0, runners=2)
    try:
        observations = [[float(i)] * 256 for i in range(16)]
        started = time.perf_counter()
        outputs = await asyncio.gather(*(batcher(observation) for observation in observations))
        took = (time.perf_counter() - started) * 1e3
        pids = sorted({output["pid"] for output in outputs})
        print(f"16 calls in {took:.0f} ms, as batches of {dict(batcher.sent)}, on replicas {pids}")
        print("the first call's own output:", [round(v, 4) for v in outputs[0]["action"]])
    finally:
        pool.close()


if __name__ == "__main__":
    asyncio.run(main())
