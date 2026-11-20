# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
optimize() on a real CUDA GPU: the capture itself, which the package's tests replace with a stand-in.

    python examples/check_on_a_gpu.py

What it checks, each printed as one RESULT line of JSON:
  env        the GPU, the driver's CUDA, torch
  on         optimize() turns cuda_graphs on for the denoiser, called ten times per run with one shape
  same       five runs on five inputs give the same outputs, byte for byte, as eager mode; four of them are inputs
             optimize() never saw
  noise      two runs in a row draw the noise eager mode would have drawn: what was captured leaves the generator as
             eager mode leaves it
  shape      a new batch size is captured on its first call and gives eager mode's output, then and replayed
  speed      eager mode against optimized, ms per run
"""
from __future__ import annotations

import copy
import json
import time

import torch

from sequence_ai.engine import optimize

FAILED: list[str] = []


def result(check: str, ok: bool, **fields) -> None:
    if not ok:
        FAILED.append(check)
    print("RESULT " + json.dumps({"check": check, "ok": ok, **fields}, default=str), flush=True)


class Block(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(width)
        self.a = torch.nn.Linear(width, width)
        self.b = torch.nn.Linear(width, width)

    def forward(self, x):
        return x + self.b(torch.nn.functional.gelu(self.a(self.norm(x))))


class Denoiser(torch.nn.Module):
    """Many small kernels per call, as in a diffusion or flow action head's denoiser at batch 1."""

    def __init__(self, width: int = 256, depth: int = 12) -> None:
        super().__init__()
        self.time = torch.nn.Linear(1, width)
        self.blocks = torch.nn.ModuleList(Block(width) for _ in range(depth))
        self.out = torch.nn.Linear(width, width)

    def forward(self, x, t):
        h = x + self.time(t)
        for block in self.blocks:
            h = block(h)
        return self.out(h)


class ActionHead(torch.nn.Module):
    steps = 10

    def __init__(self) -> None:
        super().__init__()
        self.denoiser = Denoiser()

    @torch.no_grad()
    def forward(self, condition):
        x = torch.randn(condition.shape, device=condition.device)      # the model's noise, from the GPU's generator
        for step in range(self.steps):
            t = torch.full((condition.shape[0], 1), step / self.steps, device=condition.device)
            x = x - 0.1 * self.denoiser(x + condition, t)
        return x


def same(a, b) -> bool:
    """Equal byte for byte: the same shape, dtype and values."""
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)


def ms_per_run(model, condition, runs: int = 200) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        model(condition)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e3 / runs


def main() -> int:
    result("env", torch.cuda.is_available(), torch=torch.__version__, cuda=torch.version.cuda,
           gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    if not torch.cuda.is_available():
        print("no CUDA GPU available: nothing to check")
        return 2
    torch.manual_seed(0)
    model = ActionHead().cuda().eval()
    twin = copy.deepcopy(model)                              # stays in eager mode: the reference for the optimized one
    conditions = [torch.randn(1, 256, device="cuda") for _ in range(5)]

    # ── eager mode's outputs, before optimize() runs ──
    eager = []
    for seed, condition in enumerate(conditions):
        torch.manual_seed(seed)
        eager.append(twin(condition))

    # ── on ──: optimize() sees the first condition alone
    printed: list[str] = []
    fast = optimize(model, conditions[:1], say=printed.append)
    graphs = fast.report["switches"]["cuda_graphs"]
    result("on", graphs["on"] and any(path.endswith("denoiser") for path in graphs.get("modules", [])), on=fast.on,
           cuda_graphs=graphs, whole_graph=fast.report["switches"]["whole_graph"], printed=printed)

    # ── same ──: the other four conditions were never seen by optimize()
    each = []
    for seed, condition in enumerate(conditions):
        torch.manual_seed(seed)
        each.append(same(eager[seed], model(condition)))
    result("same", all(each), each=each)

    # ── noise ──: no reseeding between two runs
    torch.manual_seed(123)
    a1, a2 = twin(conditions[1]), twin(conditions[1])
    torch.manual_seed(123)
    b1, b2 = model(conditions[1]), model(conditions[1])
    result("noise", same(a1, b1) and same(a2, b2), first=same(a1, b1), second=same(a2, b2))

    # ── shape ──: a batch of 4, never seen
    wide = torch.randn(4, 256, device="cuda")
    torch.manual_seed(7)
    want = twin(wide)
    torch.manual_seed(7)
    first = model(wide)                                      # a new shape: captured, and checked against eager mode
    torch.manual_seed(7)
    again = model(wide)                                      # replayed
    result("shape", same(want, first) and same(want, again), first=same(want, first), again=same(want, again))

    # ── speed ──
    for _ in range(20):
        model(conditions[0])
        twin(conditions[0])
    eager_ms, optimized_ms = ms_per_run(twin, conditions[0]), ms_per_run(model, conditions[0])
    result("speed", optimized_ms < eager_ms, eager_ms=round(eager_ms, 3), optimized_ms=round(optimized_ms, 3),
           times=round(eager_ms / optimized_ms, 2))

    print("RESULT " + json.dumps({"check": "all", "ok": not FAILED, "failed": FAILED}), flush=True)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
