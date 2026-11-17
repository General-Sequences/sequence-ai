# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
batcher, alone: the calls waiting for a model batched into one call to it, in the sizes the model lists.

    python examples/components/batcher.py
"""
import asyncio
import time

from sequence_ai.engine.batcher import Batcher


def model_on_a_list(observations: list) -> list:
    time.sleep(0.05)                               # one forward pass, whatever the batch: your model's call here
    return [observation * 2 for observation in observations]


async def main() -> None:
    batcher = Batcher(model_on_a_list, sizes=[1, 2, 4, 8], wait_ms=2.0)
    started = time.perf_counter()
    outputs = await asyncio.gather(*(batcher(i) for i in range(11)))       # what each request's handler awaits
    took = (time.perf_counter() - started) * 1e3
    print(f"11 calls, each with its own output {outputs[:4]}..., in {took:.0f} ms as batches of {dict(batcher.sent)}")


asyncio.run(main())
