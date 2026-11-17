# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
Quickstart: five robots send requests at the same time, their requests reach the model as batches, and the model runs
in its own process. Standard library only: no GPU and no torch needed to see it work.

    pip install sequence-ai
    python quickstart.py

Put your model where `model_on_a_list` is. docs/quickstart.md walks through it.
"""
import asyncio
import os
import time

from sequence_ai.engine.batcher import Batcher
from sequence_ai.engine.replicas import ReplicaPool


def load():
    """Runs in the replica's own process: build your model here, and return what one batch of requests runs."""
    def model_on_a_list(observations: list) -> list:
        time.sleep(0.05)                                   # one forward pass, whatever the batch: your model's call
        return [{"actions": [[observation] * 7] * 8, "pid": os.getpid()} for observation in observations]

    return model_on_a_list


async def main() -> None:
    pool = ReplicaPool(load, replicas=1)                   # one replica per GPU; devices=[0] sets the GPU it sees
    batcher = Batcher(pool.run, sizes=[1, 2, 4, 8], wait_ms=2.0)
    try:
        started = time.perf_counter()
        # in your own server, each request's handler awaits batcher(observation)
        outputs = await asyncio.gather(*(batcher(float(robot)) for robot in range(5)))
        took = (time.perf_counter() - started) * 1e3
        print(f"5 robots served in {took:.0f} ms, as batches of {dict(batcher.sent)}")
        actions = outputs[3]["actions"]
        print(f"robot 3's own response: {len(actions)} actions of {len(actions[0])} numbers, "
              f"the first {actions[0][:3]}")
        print(f"this process is {os.getpid()}; the model ran in {outputs[3]['pid']}")
    finally:
        pool.close()


if __name__ == "__main__":
    asyncio.run(main())
