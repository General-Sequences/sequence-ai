# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
Quickstart: five robots send requests at the same time, their requests reach the model as batches, and the model runs
in its own process. Standard library only: no GPU and no torch needed to see it work.

    pip install sequence-ai
    python quickstart.py

Put your model where `model_on_a_list_user` is. docs/quickstart.md walks through it.
"""
import asyncio
import os
import time

from sequence_ai.engine.batcher import Batcher
from sequence_ai.engine.replicas import ReplicaPool


def load_user():
    """Runs in the replica's own process: build your model here, and return what one batch of requests runs."""
    def model_on_a_list_user(observations_user: list) -> list:
        time.sleep(0.05)                                   # one forward pass, whatever the batch: your model's call
        return [{"actions": [[observation_user] * 7] * 8, "pid": os.getpid()} for observation_user in observations_user]

    return model_on_a_list_user


async def main_user() -> None:
    pool_user = ReplicaPool(load_user, replicas=1)        # one replica per GPU; devices=[0] sets the GPU it sees
    batcher_user = Batcher(pool_user.run, sizes=[1, 2, 4, 8], wait_ms=2.0)
    try:
        started_user = time.perf_counter()
        # in your own server, each request's handler awaits batcher(observation)
        outputs_user = await asyncio.gather(*(batcher_user(float(robot_user)) for robot_user in range(5)))
        took_user = (time.perf_counter() - started_user) * 1e3
        print(f"5 robots served in {took_user:.0f} ms, as batches of {dict(batcher_user.sent)}")
        actions_user = outputs_user[3]["actions"]
        print(f"robot 3's own response: {len(actions_user)} actions of {len(actions_user[0])} numbers, "
              f"the first {actions_user[0][:3]}")
        print(f"this process is {os.getpid()}; the model ran in {outputs_user[3]['pid']}")
    finally:
        pool_user.close()


if __name__ == "__main__":
    asyncio.run(main_user())
