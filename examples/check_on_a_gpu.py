# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
cuda_graphs on a real CUDA GPU: the capture itself, which the package's tests replace with a fake.

    python examples/check_on_a_gpu.py

What it checks, each printed as one RESULT line of JSON:
  env        the GPU, the driver's CUDA, torch
  census     the census finds the denoiser, called ten times per run with one shape
  same       five runs on five inputs give the same outputs, byte for byte, as eager mode did before anything was
             installed; four of them are inputs the capture never saw
  noise      two consecutive runs draw the noise eager mode would have drawn: a capture leaves the generator unchanged
  shape      a new batch size is captured automatically and gives the same output
  report     the shapes captured, and none left in eager mode
  speed      eager mode against graphed, ms per run
"""
from __future__ import annotations

import copy
import json
import time

import torch

from sequence_ai.engine import cuda_graphs

FAILED: list[str] = []


def result(check: str, ok: bool, **fields) -> None:
    if not ok:
        FAILED.append(check)
    print("RESULT " + json.dumps({"check": check, "ok": ok, **fields}), flush=True)


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
    twin = copy.deepcopy(model)                              # stays in eager mode: the reference for the graphed one
    conditions = [torch.randn(1, 256, device="cuda") for _ in range(5)]

    # ── census ──
    census = cuda_graphs.Census(cuda_graphs.modules_of(model))
    for _ in range(3):
        model(conditions[0])
        census.end_act()
    census.close()
    found = census.candidates()
    result("census", [row["path"] for row in found] == ["denoiser"] and found[0]["calls"] == 10, found=found)

    # ── eager mode's outputs, before anything is installed ──
    eager = []
    for seed, condition in enumerate(conditions):
        torch.manual_seed(seed)
        eager.append(twin(condition))

    printed: list[str] = []
    graphs = cuda_graphs.install(model, ["denoiser"], say=printed.append)

    # ── same ──: the first run captures on conditions[0]; the other four were never seen by the capture
    same = []
    for seed, condition in enumerate(conditions):
        torch.manual_seed(seed)
        same.append(cuda_graphs.same(eager[seed], model(condition)))
    result("same", all(same), each=same, printed=printed)

    # ── noise ──: no reseeding between two runs, on a newly graphed model so its first run is a capture
    fresh = copy.deepcopy(twin)
    torch.manual_seed(123)
    a1, a2 = twin(conditions[0]), twin(conditions[0])
    cuda_graphs.install(fresh, ["denoiser"], say=printed.append)
    torch.manual_seed(123)
    b1, b2 = fresh(conditions[0]), fresh(conditions[0])
    result("noise", cuda_graphs.same(a1, b1) and cuda_graphs.same(a2, b2),
           first=cuda_graphs.same(a1, b1), second=cuda_graphs.same(a2, b2))

    # ── shape ──: a batch of 4, never seen
    wide = torch.randn(4, 256, device="cuda")
    torch.manual_seed(7)
    want = twin(wide)
    torch.manual_seed(7)
    captured = model(wide)                                   # run in eager mode, captured, checked
    torch.manual_seed(7)
    replayed = model(wide)
    result("shape", cuda_graphs.same(want, captured) and cuda_graphs.same(want, replayed),
           captured=cuda_graphs.same(want, captured), replayed=cuda_graphs.same(want, replayed))

    report = graphs["denoiser"].report()
    result("report", report["shapes"] == 2 and report["eager"] == [], report=report, printed=printed)

    # ── speed ──
    for _ in range(20):
        model(conditions[0])
    cuda_graphs.ACTIVE = False
    eager_ms = ms_per_run(model, conditions[0])
    cuda_graphs.ACTIVE = True
    graphed_ms = ms_per_run(model, conditions[0])
    result("speed", graphed_ms < eager_ms, eager_ms=round(eager_ms, 3), graphed_ms=round(graphed_ms, 3),
           times=round(eager_ms / graphed_ms, 2))

    print("RESULT " + json.dumps({"check": "all", "ok": not FAILED, "failed": FAILED}), flush=True)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
