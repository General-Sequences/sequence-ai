# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
Quickstart, robot side: the robot's own loop. Each answer is played a step at a time at the policy's hz, the next asked
for in the background at its replan step, so the robot never waits; the steps that passed on the way are skipped, and
the arm is held still while the server cannot answer.

    python quickstart_robot.py                       # against `seq serve examples/quickstart_policy` on this machine
    python quickstart_robot.py http://YOUR_GPU:8000  # against the same policy on your GPU

An answer's first dimension is time: its row i is the step to play i steps after the observation it answers.
docs/quickstart.md walks through it.
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import sequence_ai

HZ_USER = 15            # the policy's hz: the steps a second
VALID_USER = 8          # its action_chunk_valid_len: the steps of an answer the robot plays
REPLAN_AT_USER = 5      # its action_chunk_replan_at: the step at which the robot asks for the next answer

url_user = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"


def observation_now_user() -> dict:
    """Your robot here: its camera's frame and its joints."""
    return {"image": np.full((224, 224, 3), 128, dtype=np.uint8), "state": np.zeros(7, dtype=np.float32)}


def move_user(step_user: np.ndarray) -> None:
    """Your robot here: one step's 7 joint targets."""


def hold_user() -> None:
    """Your robot here: hold still where it is."""


def ask_user(policy_user) -> tuple:
    """The next answer's valid steps, and when it was asked for."""
    asked_user = time.monotonic()
    return policy_user.act()["actions"][:VALID_USER], asked_user


with sequence_ai.connect(url_user) as policy_user, ThreadPoolExecutor(1) as background_user:
    policy_user.wait_until_ready()                                         # once, before the robot moves
    policy_user.observe(observation_now_user())
    chunk_user, asked_user = ask_user(policy_user)
    print(f"the first answer: {len(chunk_user)} steps of {chunk_user.shape[1]} joints, the first {chunk_user[0, :3]}")
    played_user = 0
    while played_user < 3 * HZ_USER:                                       # 3 s of steps
        started_user, next_user = time.monotonic(), None
        for i_user, step_user in enumerate(chunk_user):
            if i_user == REPLAN_AT_USER:
                next_user = background_user.submit(ask_user, policy_user)  # 1. ask early, in the background
            move_user(step_user)
            played_user += 1
            policy_user.observe(observation_now_user())                    # 2. the newest observation, each step
            time.sleep(max(0.0, started_user + (i_user + 1) / HZ_USER - time.monotonic()))   # 3. on to the next step
        try:
            chunk_user, asked_user = (next_user or background_user.submit(ask_user, policy_user)).result()
            chunk_user = chunk_user[int((time.monotonic() - asked_user) * HZ_USER):]       # 4. skip the steps passed
        except Exception:                          # 5. a dropped link, a restarted server, the policy raising:
            hold_user()                            #    hold still,
            policy_user.wait_until_ready()         #    wait until the server answers again,
            policy_user.observe(observation_now_user())
            chunk_user, asked_user = ask_user(policy_user)                                  #    and go on from here
    acts_user = policy_user.stats()["latency_us"]["act_us"]
    print(f"{played_user} steps played at {HZ_USER} a second; {acts_user['n']} answers, the slowest in "
          f"{acts_user['max'] / 1000:.1f} ms")
