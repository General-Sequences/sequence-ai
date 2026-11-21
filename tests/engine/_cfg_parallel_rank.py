# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
One of the two processes tests/engine/test_cfg_parallel.py starts, as torchrun starts each (RANK, WORLD_SIZE,
MASTER_ADDR, MASTER_PORT): a toy classifier-free-guidance sampler run whole, then with its denoiser split by
cfg_parallel, then with the split taken back - each run's actions and the batch each call of the denoiser was given,
written to <out>/<rank>.json.

    RANK=0 WORLD_SIZE=2 MASTER_ADDR=127.0.0.1 MASTER_PORT=29511 python tests/engine/_cfg_parallel_rank.py <out>
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import torch

from sequence_ai.engine import cfg_parallel


class Denoiser(torch.nn.Module):
    """A small denoiser called with the guidance's doubled batch: noisy actions, the step, the prompt's embedding."""

    def __init__(self) -> None:
        super().__init__()
        self.actions = torch.nn.Linear(8, 64)
        self.prompt = torch.nn.Linear(4, 64)
        self.head = torch.nn.Linear(64, 8)
        self.seen: list = []

    def forward(self, actions, step, prompt):
        self.seen.append(int(actions.shape[0]))
        hidden = torch.tanh(self.actions(actions) + self.prompt(prompt) + step[:, None])
        return {"velocity": self.head(hidden), "step": step}


def sample(model: Denoiser, noise, prompt, unconditioned, *, steps: int = 4, scale: float = 3.0):
    """The guidance's loop as a pipeline writes it: the batch doubled, its unconditioned half first."""
    actions = noise
    for i in range(steps):
        step = torch.full((2 * actions.shape[0],), i / steps)
        out = model(torch.cat([actions, actions]), step, torch.cat([unconditioned, prompt]))
        free, held = out["velocity"].chunk(2)
        actions = actions - (free + scale * (held - free)) / steps
    return actions


def main(out: pathlib.Path) -> None:
    torch.manual_seed(0)
    model = Denoiser()
    noise, prompt, unconditioned = torch.randn(3, 8), torch.randn(3, 4), torch.zeros(3, 4)
    with torch.inference_mode():
        alone = sample(model, noise, prompt, unconditioned)
        seen_alone, model.seen = model.seen, []
        said: list = []
        patches = cfg_parallel.install({"model": model}, ["model", "model.nothing"], say=said.append)
        split = sample(model, noise, prompt, unconditioned)
        seen_split, model.seen = model.seen, []
        cfg_parallel.uninstall(patches)
        again = sample(model, noise, prompt, unconditioned)
    rank = int(os.environ["RANK"])
    (out / f"{rank}.json").write_text(json.dumps({
        "rank": rank, "alone": alone.tolist(), "split": split.tolist(), "again": again.tolist(),
        "seen_alone": seen_alone, "seen_split": seen_split, "seen_again": model.seen, "said": said,
        "patched": [p.what for p in patches], "restored": "forward" not in vars(model),
        "max_abs": float((split - alone).abs().max()), "exact": bool(torch.equal(split, alone))}))
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))
