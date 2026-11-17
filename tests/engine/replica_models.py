# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""Models for tests/engine/test_replicas.py, here so a replica's new process can import them without pytest."""
from __future__ import annotations

import os
import time


def load_echo():
    """Returns who ran the call, on which GPU, and how many calls this replica has run."""
    calls = [0]

    def model(payload):
        calls[0] += 1
        if payload == "raise":
            raise ValueError("the model refuses this")
        if payload == "die":
            os._exit(3)
        if isinstance(payload, dict) and "sleep" in payload:
            time.sleep(payload["sleep"])
        return {"pid": os.getpid(), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "calls": calls[0],
                "payload": payload}
    return model


def load_that_raises():
    raise RuntimeError("no weights here")


def load_then_always_dies():
    def model(payload):
        os._exit(4)
    return model
