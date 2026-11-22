# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
One process of tests/engine/test_pipeline_parallel.py's servers: serve() of a toy diffusion transformer's policy - its
noise drawn from the observation, its blocks called one after another - with pipeline_parallel asked and a tolerance,
alone or as one rank of two, as the environment says (torchrun's: WORLD_SIZE, RANK, MASTER_ADDR, MASTER_PORT,
SEQ_SERVE_RANKS_PORT). `kept`: optimize.MIN_SPEEDUP 0, so the split is kept whatever it saves on a CPU.

    python tests/engine/_pipeline_parallel_server.py <port> [kept]
"""
from __future__ import annotations

import importlib
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _pipeline_parallel_rank import TOKENS, DiT  # noqa: E402

from sequence_ai.engine.serve import serve  # noqa: E402

EXAMPLE = {"state": np.linspace(-1.0, 1.0, 7).astype(np.float32)}


def denoised():
    """The policy: four steps of the diffusion from noise the observation seeds, conditioned on its state."""
    torch.manual_seed(0)
    model = DiT().eval()

    def policy(observation: dict) -> dict:
        state = torch.as_tensor(observation["state"], dtype=torch.float32)
        seed = int(abs(float(state.sum())) * 1e6) % (2 ** 31)
        x = torch.randn(1, TOKENS, 8, generator=torch.Generator().manual_seed(seed))
        cond = state[None, :4]
        with torch.inference_mode():
            for i in range(4):
                x = x - 0.25 * model(x, torch.full((1,), i / 4), cond)
        return {"actions": x[0, :, :7].numpy()}

    return policy


if __name__ == "__main__":
    port = int(sys.argv[1])
    if "kept" in sys.argv[2:]:
        importlib.import_module("sequence_ai.engine.optimize").MIN_SPEEDUP = 0.0      # the module, not its function
    serve(denoised(), host="127.0.0.1", port=port, example=EXAMPLE, pipeline_parallel=True, tolerance=0.2,
          log_level="info")
