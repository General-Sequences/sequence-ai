# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
Quickstart, server side: a policy server in one call. Put your model where `policy_user` is.

    pip install sequence-ai
    python quickstart_server.py                  # every address of this machine, port 8000
    python quickstart_server.py 127.0.0.1 8000   # this machine only

serve() answers POST /act with what `policy_user` returns, and GET /ping at once; it runs `policy_user` on
`example_user` before it says it is ready. docs/quickstart.md walks through it with examples/quickstart_robot.py.
"""
import sys

import numpy as np

import sequence_ai


def policy_user(observation_user: dict) -> dict:
    """Your model here: an observation in, a chunk of actions out. This one returns 8 actions of 7 numbers."""
    return {"actions": np.full((8, 7), float(observation_user["image"].mean()), dtype=np.float32)}


# an observation as your robot sends it: serve() runs the policy on it before it says it is ready
example_user = {"image": np.zeros((224, 224, 3), dtype=np.uint8), "state": np.zeros(7, dtype=np.float32)}

if __name__ == "__main__":
    host_user = sys.argv[1] if len(sys.argv) > 1 else None
    port_user = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    sequence_ai.serve(policy_user, host=host_user, port=port_user, example=example_user)
