# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
Graph the module a model calls many times per run, in three steps: find it, graph it, check that the outputs are the
same.

    python examples/graph_a_module.py

The model here stands for a flow or diffusion action head: ten denoising steps over the same shapes. On a machine with
no CUDA GPU the same steps run in eager mode, so this shows the flow there and saves no time.
"""
from __future__ import annotations

import time

import torch

from sequence_ai.engine import cuda_graphs


class Denoiser(torch.nn.Module):
    def __init__(self, width: int = 512) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(width, width), torch.nn.GELU(), torch.nn.Linear(width, width))

    def forward(self, x, t):
        return x - 0.1 * self.net(x + t)


class ActionHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.denoiser = Denoiser()

    @torch.no_grad()
    def forward(self, noise):
        x = noise
        for step in range(10):                                   # the same module, the same shapes, ten times per run
            x = self.denoiser(x, torch.full_like(x[:, :1], step / 10))
        return x


def ms_per_run(model, noise, runs: int = 20) -> float:
    if noise.is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        model(noise)
    if noise.is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e3 / runs


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ActionHead().to(device).eval()
    noise = torch.randn(32, 512, device=device)

    # 1. Find the modules worth graphing: called more than once per run, with one shape, the outermost first.
    census = cuda_graphs.Census(cuda_graphs.modules_of(model))
    for _ in range(3):
        model(noise)
        census.end_act()
    census.close()
    found = census.candidates(min_ms=0.0)
    print("candidates:", found)

    # 2. Graph them in place. The model is not rebuilt: each module keeps its class, its weights and its hooks.
    before = model(noise)
    eager_ms = ms_per_run(model, noise)
    cuda_graphs.install(model, [row["path"] for row in found])
    model(noise)                                               # each new shape: run in eager mode, captured, checked

    # 3. The outputs are the same, byte for byte, and on a GPU each run is faster.
    after = model(noise)
    print("same outputs, byte for byte:", cuda_graphs.same(before, after))
    if device == "cuda":
        print(f"eager {eager_ms:.2f} ms per run, graphed {ms_per_run(model, noise):.2f} ms per run")
    else:
        print(f"eager {eager_ms:.2f} ms per run; no CUDA GPU available, so nothing was graphed")


if __name__ == "__main__":
    main()
