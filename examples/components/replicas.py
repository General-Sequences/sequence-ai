# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
replicas, alone: replicas of a model, each in its own process, with the model's own error distinguished from a replica
that died.

    python examples/components/replicas.py
"""
import os

from sequence_ai.engine.replicas import ReplicaError, ReplicaPool


def load():
    """Runs in each replica's own process: build your model here and return what a call runs."""
    def run(payload):
        if payload == "bad input":
            raise ValueError("the model's own error")
        return {"output": payload * 2, "pid": os.getpid()}

    return run


if __name__ == "__main__":
    pool = ReplicaPool(load, replicas=2)           # devices=[0, 1] gives each replica its own GPU
    try:
        print("outputs:", [pool.call(i)["output"] for i in range(4)], "from", pool.size, "replicas")
        try:
            pool.call("bad input")
        except ReplicaError as error:
            print("the model raised, and its replica takes the next call:", str(error).strip().splitlines()[-1])
        print("still serving:", pool.call(21)["output"])
    finally:
        pool.close()
