# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
Quickstart, client side: the first action from a policy server, in three lines of sequence-ai.

    pip install sequence-ai
    python quickstart_robot.py                           # against examples/quickstart_server.py on this machine
    python quickstart_robot.py http://YOUR_GPU:8000      # against the same server on your GPU

docs/quickstart.md walks through it.
"""
import sys

import numpy as np

import sequence_ai

url_user = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
frame_user = np.full((224, 224, 3), 128, dtype=np.uint8)        # your camera's frame here
state_user = np.zeros(7, dtype=np.float32)                      # your robot's joints here

with sequence_ai.connect(url_user) as link_user:                # BBR, TCP keepalive, a ping when idle, msgpack: all on
    for step_user in range(3):
        actions_user = link_user.act({"image": frame_user, "state": state_user})["actions"]
        print(f"request {step_user}: {actions_user.shape[0]} actions of {actions_user.shape[1]} numbers, "
              f"the first {actions_user[0, :3]}")
    timing_user = link_user.stats()["latency_us"]["act_us"]
    print(f"{timing_user['n']} requests, the slowest took {timing_user['max'] / 1000:.1f} ms")
