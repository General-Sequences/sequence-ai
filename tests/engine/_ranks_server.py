# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
One process of tests/engine/test_serve_ranks.py's servers: serve() of a toy policy, alone or as one rank of a group, as
the environment says (WORLD_SIZE, RANK, MASTER_ADDR, MASTER_PORT, SEQ_SERVE_RANKS_PORT - torchrun's).

    python tests/engine/_ranks_server.py cfg <port>      # a classifier-free-guidance policy, its two branches split
                                                         # across two ranks over gloo (torch), or both here alone
    python tests/engine/_ranks_server.py plain <port>    # numpy alone: each rank the same, every call it ran written
                                                         # to $SEQ_TEST_CALLS.<rank>
    python tests/engine/_ranks_server.py class <port>    # a @seq.policy class under seq serve (its observe function
                                                         # and on_episode), each rank's calls written there too
"""
from __future__ import annotations

import os
import sys

import numpy as np

from sequence_ai.engine.serve import serve

EXAMPLE = {"state": np.zeros(7, np.float32)}


def cfg_policy():
    """A batch function whose answer is uncond + 7.5 (cond - uncond), each branch a layer of fixed weights: alone, both
    branches here; as rank 0 or 1 of two, the conditional branch on rank 0, the unconditional on rank 1, each rank's
    sent to the other (torch.distributed's all_gather, over gloo) - a classifier-free guidance split, on a CPU."""
    import torch
    import torch.distributed as dist

    world, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))
    if world > 1:
        dist.init_process_group("gloo", init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
                                rank=rank, world_size=world)
    weights = torch.arange(49, dtype=torch.float32).reshape(7, 7) / 49.0

    def branch(states, conditional: bool):
        return torch.tanh(states @ weights + (0.5 if conditional else -0.25))

    def policy_on_a_list(observations: list) -> list:
        states = torch.stack([torch.as_tensor(o["state"], dtype=torch.float32) for o in observations])
        if world == 1:
            cond, uncond = branch(states, True), branch(states, False)
        else:
            mine = branch(states, rank == 0)
            both = [torch.empty_like(mine) for _ in range(world)]
            dist.all_gather(both, mine)
            cond, uncond = both[0], both[1]
        out = uncond + 7.5 * (cond - uncond)
        return [{"actions": out[i].numpy()} for i in range(len(observations))]

    return policy_on_a_list


def plain_policy():
    """A batch function of numpy alone, each rank the same; every call it ran written down, one line each."""
    rank = os.environ.get("RANK", "0")
    calls = os.environ.get("SEQ_TEST_CALLS")

    def policy_on_a_list(observations: list) -> list:
        if calls:
            with open(f"{calls}.{rank}", "a") as f:
                f.write(f"{len(observations)} {float(observations[0]['state'][0])}\n")
        return [{"actions": np.tanh(np.asarray(o["state"], np.float32) * 3.0)} for o in observations]

    return policy_on_a_list


# A class whose act answers what its observe function kept of its episode, every observe, act and episode's end written
# down by the rank that ran it.
COUNTING = '''
import os

import numpy as np
import seq


def _written(line):
    calls = os.environ.get("SEQ_TEST_CALLS")
    if calls:
        with open(calls + "." + os.environ.get("RANK", "0"), "a") as f:
            f.write(line + "\\n")


@seq.policy(name="counting_user", hz=10)
class CountingUser(seq.Policy):
    @seq.sample
    def sample(self) -> list:
        return [("observe", {"state": np.zeros(7, np.float32)}), ("act",)]

    @seq.observe
    def observe(self, obs: dict, episode) -> None:
        episode.state.setdefault("seen_user", []).append(float(obs["state"][0]))
        _written(f"observe {episode.id} {float(obs['state'][0])}")

    @seq.on_episode
    def on_episode(self, episode) -> None:
        _written(f"episode {episode.id} {episode.phase}")

    @seq.infer
    def infer(self, episode) -> dict:
        seen = episode.state.get("seen_user", [])
        _written(f"act {episode.id} {seen}")
        return {"actions": np.asarray([sum(seen), len(seen)], np.float32)}
'''


def serve_class(port: int) -> None:
    """COUNTING under `seq serve`, its folder in $SEQ_TEST_FOLDER."""
    import pathlib

    from seq import serve_local

    folder = pathlib.Path(os.environ["SEQ_TEST_FOLDER"]) / os.environ.get("RANK", "0")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "policy.py").write_text(COUNTING)
    serve_local.serve(folder, host="127.0.0.1", port=port)


if __name__ == "__main__":
    mode, port = sys.argv[1], int(sys.argv[2])
    if mode == "class":
        serve_class(port)
    else:
        policy = cfg_policy() if mode == "cfg" else plain_policy()
        serve(policy, host="127.0.0.1", port=port, batch=[1, 2], example=EXAMPLE, log_level="warning")
