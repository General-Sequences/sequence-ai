# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
One of the two processes tests/engine/test_pipeline_parallel.py starts, as torchrun starts each (RANK, WORLD_SIZE,
MASTER_ADDR, MASTER_PORT): a toy diffusion transformer's sampler run whole, then with its blocks split by
pipeline_parallel in each form - each run's actions, the calls run whole and in parts, the shares timed - and the
blocks' own forwards after, written to <out>/<rank>.json.

    RANK=0 WORLD_SIZE=2 MASTER_ADDR=127.0.0.1 MASTER_PORT=29511 python tests/engine/_pipeline_parallel_rank.py <out>
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import torch
import torch.nn.functional as F

from sequence_ai.engine import pipeline_parallel

WIDTH, HEADS, TOKENS, BLOCKS = 32, 4, 8, 4


class Block(torch.nn.Module):
    """A diffusion transformer's block: its condition scales and shifts its norm, its self-attention through
    scaled_dot_product_attention."""

    def __init__(self) -> None:
        super().__init__()
        self.norm1, self.norm2 = torch.nn.LayerNorm(WIDTH), torch.nn.LayerNorm(WIDTH)
        self.qkv, self.proj = torch.nn.Linear(WIDTH, 3 * WIDTH), torch.nn.Linear(WIDTH, WIDTH)
        self.ada = torch.nn.Linear(WIDTH, 2 * WIDTH)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(WIDTH, 2 * WIDTH), torch.nn.GELU(),
                                       torch.nn.Linear(2 * WIDTH, WIDTH))

    def forward(self, x, c):
        shift, scale = self.ada(c)[:, None].chunk(2, -1)
        h = self.norm1(x) * (1 + scale) + shift
        q, k, v = self.qkv(h).view(x.shape[0], x.shape[1], 3, HEADS, WIDTH // HEADS).permute(2, 0, 3, 1, 4)
        x = x + self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(x.shape))
        return x + self.mlp(self.norm2(x))


class DiT(torch.nn.Module):
    """Noisy actions as tokens, the step and the condition in; their velocity out."""

    def __init__(self) -> None:
        super().__init__()
        self.into, self.out = torch.nn.Linear(8, WIDTH), torch.nn.Linear(WIDTH, 8)
        self.cond = torch.nn.Linear(4, WIDTH)
        self.pos = torch.nn.Parameter(0.1 * torch.randn(1, TOKENS, WIDTH))
        self.blocks = torch.nn.ModuleList([Block() for _ in range(BLOCKS)])

    def forward(self, x, t, cond):
        c = self.cond(cond) + t[:, None]
        h = self.into(x) + self.pos
        for block in self.blocks:
            h = block(h, c)
        return self.out(h)


def sample(model: DiT, noise, cond, *, steps: int = 4):
    """One act: a few steps of the diffusion from `noise`, as serve() begins each act."""
    pipeline_parallel.begin()
    x = noise
    for i in range(steps):
        x = x - 0.25 * model(x, torch.full((x.shape[0],), i / steps), cond)
    return x


def main(out: pathlib.Path) -> None:
    rank = int(os.environ["RANK"])
    torch.manual_seed(0)
    model = DiT().eval()
    noise, cond = torch.randn(4, TOKENS, 8), torch.randn(4, 4)
    found: dict = {}
    with torch.inference_mode():
        alone = sample(model, noise, cond)

        layers = pipeline_parallel.install({"model": model}, "model.blocks", form="layers", parts=2, stages=[2, 2])
        split_layers = sample(model, noise, cond)
        found["layers_ran"] = dict(layers.ran)
        pipeline_parallel.uninstall(layers)

        timed = pipeline_parallel.install({"model": model}, "model.blocks", form="layers", parts=2)
        sample(model, noise, cond)
        found["stages"] = timed.stages
        pipeline_parallel.uninstall(timed)

        patches = pipeline_parallel.install({"model": model}, "model.blocks", form="patches", parts=2, stages=[2, 2])
        split_patches = sample(model, noise, cond)
        found["patches_ran"] = dict(patches.ran)
        t = torch.zeros(4)
        pipeline_parallel.begin()
        once = model(noise, t, cond)               # the act's first call: whole, its keys and values kept
        again = model(noise, t, cond)              # the same input in patches: every patch's keys and values fresh
        pipeline_parallel.uninstall(patches)
        want = model(noise, t, cond)

        restored = all("forward" not in vars(block) for block in model.blocks)
        after = sample(model, noise, cond)

        order = pipeline_parallel.install({"model": model}, "model.blocks", form="layers", parts=2, stages=[2, 2])
        try:
            model.blocks[1](noise.new_zeros(4, TOKENS, WIDTH), torch.zeros(4, WIDTH))
            found["order"] = None
        except RuntimeError as e:
            found["order"] = str(e)
        pipeline_parallel.uninstall(order)

        failing = pipeline_parallel.install({"model": model}, "model.blocks", form="layers", parts=2, stages=[2, 2])

        def boom(*args, **kwargs):
            raise ValueError("this block fails on purpose")

        failing._own[3] = boom                        # block 3 is rank 1's: rank 0 waits on its part
        try:
            sample(model, noise, cond)
            found["failed"] = None
        except RuntimeError as e:
            found["failed"] = str(e)
        pipeline_parallel.uninstall(failing)

    if rank == 0:
        found["layers_max_abs"] = float((split_layers - alone).abs().max())
        found["patches_max_abs"] = float((split_patches - alone).abs().max())
        found["again_max_abs"] = float((again - want).abs().max())
        found["once_max_abs"] = float((once - want).abs().max())
    found["restored"] = restored
    found["after_same"] = bool(torch.equal(after, alone))
    (out / f"{rank}.json").write_text(json.dumps(found))


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))
