# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
One process of tests/engine/test_cfg_parallel.py's servers: serve() of a toy guidance policy - its denoiser called with
the doubled batch, its noise drawn from the observation - with cfg_parallel asked and a tolerance, alone or as one rank
of two, as the environment says (torchrun's: WORLD_SIZE, RANK, MASTER_ADDR, MASTER_PORT, SEQ_SERVE_RANKS_PORT).
`kept`: optimize.MIN_SPEEDUP 0, so the split is kept whatever it saves on a CPU.

    python tests/engine/_cfg_parallel_server.py <port> [kept]
"""
from __future__ import annotations

import importlib
import sys

import numpy as np
import torch

from sequence_ai.engine.serve import serve

EXAMPLE = {"state": np.linspace(-1.0, 1.0, 7).astype(np.float32)}


class Denoiser(torch.nn.Module):
    """Noisy actions, the step and the prompt's embedding in; their velocity out - each row apart from the others."""

    def __init__(self) -> None:
        super().__init__()
        self.actions = torch.nn.Linear(8, 64)
        self.prompt = torch.nn.Linear(4, 64)
        self.head = torch.nn.Linear(64, 8)

    def forward(self, actions, step, prompt):
        return self.head(torch.tanh(self.actions(actions) + self.prompt(prompt) + step[:, None]))


def guided():
    """The policy: four steps of guidance from noise the observation seeds, the batch doubled for each, the
    unconditioned half first."""
    torch.manual_seed(0)
    model = Denoiser()

    def policy(observation: dict) -> dict:
        state = torch.as_tensor(observation["state"], dtype=torch.float32)
        seed = int(abs(float(state.sum())) * 1e6) % (2 ** 31)
        actions = torch.randn(1, 8, generator=torch.Generator().manual_seed(seed))
        prompt = state[None, :4]
        with torch.inference_mode():
            for i in range(4):
                step = torch.full((2,), i / 4)
                both = model(torch.cat([actions, actions]), step, torch.cat([torch.zeros_like(prompt), prompt]))
                free, held = both.chunk(2)
                actions = actions - (free + 3.0 * (held - free)) / 4
        return {"actions": actions[0].numpy()}

    return policy


if __name__ == "__main__":
    port = int(sys.argv[1])
    if "kept" in sys.argv[2:]:
        importlib.import_module("sequence_ai.engine.optimize").MIN_SPEEDUP = 0.0      # the module, not its function
    serve(guided(), host="127.0.0.1", port=port, example=EXAMPLE, cfg_parallel=True, tolerance=1e-4,
          log_level="info")
