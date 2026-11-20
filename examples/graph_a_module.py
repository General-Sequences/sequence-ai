# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
A model made faster without changing its outputs, in one call: optimize() finds the module the model calls many times
per run, replays it from CUDA graphs of its own kernels, and keeps each graph only where the outputs stay the same,
byte for byte, and acts are faster.

    pip install sequence-ai torch
    python examples/graph_a_module.py

The model here stands for a flow or diffusion action head: ten denoising steps over the same shapes. On a machine with
no CUDA GPU nothing is captured, the report says why, and the model runs as it did.
"""
from __future__ import annotations

import torch

from sequence_ai.engine import optimize


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


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ActionHead().to(device).eval()
    examples = [torch.randn(32, 512, device=device) for _ in range(8)]    # what the model is called with
    before = model(examples[0])

    # One call: each speed-up that keeps the outputs exactly the same is tried, checked on the examples and timed.
    # The model is changed in place, so `model` itself runs what is on.
    fast = optimize(model, examples, say=lambda line: None)
    for name in ("cuda_graphs", "whole_graph"):
        switch = fast.report["switches"][name]
        said = f"on, {switch.get('modules') or switch.get('call')}" if switch["on"] else f"off - {switch['why']}"
        print(f"{name}: {said}")

    print("same outputs, byte for byte:", torch.equal(before, model(examples[0])))
    if fast.on:
        act = fast.report["act_ms"]
        print(f"act {act['eager']:.2f} ms before, {act['optimized']:.2f} ms now")


if __name__ == "__main__":
    main()
