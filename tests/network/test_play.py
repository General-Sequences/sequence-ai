# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
play(): each answer's steps played while the next is asked for, switched to at the step the robot has reached since
that act was sent - never a step twice - and waited for when the valid steps run out first.
"""
import threading
import time

import numpy as np
import pytest

from sequence_ai.network import play
from sequence_ai.network.loop import steps


class Policy:
    """observe() then act(): its k-th answer is {"actions": [(k, 0), (k, 1), ...]}, `n` steps, after `act_s`."""

    def __init__(self, n: int = 10, act_s: float = 0.0, fail_at: int = -1) -> None:
        self.n, self.act_s, self.fail_at = n, act_s, fail_at
        self.acts, self.observed, self.ready = 0, [], 0
        self.lock = threading.Lock()

    def wait_until_ready(self) -> None:
        self.ready += 1

    def observe(self, observation) -> None:
        self.observed.append(observation)

    def act(self):
        with self.lock:
            k, self.acts = self.acts, self.acts + 1
        if k == self.fail_at:
            raise RuntimeError("the server is gone")
        time.sleep(self.act_s)
        return {"actions": [(k, i) for i in range(self.n)]}


def run(policy, *, count: int, **kw) -> list:
    played: list = []
    took = play(policy, read=lambda: {"t": time.monotonic()}, step=played.append,
                done=lambda: len(played) >= count, **kw)
    assert took == len(played) == count
    return played


def answers(played: list) -> dict:
    by: dict = {}
    for k, i in played:
        by.setdefault(k, []).append(i)
    return by


def test_no_step_plays_twice_and_each_answer_plays_its_steps_in_order_from_where_the_robot_is():
    policy = Policy(n=10, act_s=0.03)
    played = run(policy, count=40, hz=200, replan_at=2)
    assert len(set(played)) == len(played)
    for k, idx in answers(played).items():
        assert idx == list(range(idx[0], idx[0] + len(idx))), (k, idx)
        if k > 0:
            assert idx[0] >= 1, (k, idx)            # its act was in flight while the robot played on
    assert policy.ready == 1 and len(policy.observed) == policy.acts


def test_when_the_valid_steps_run_out_before_the_replan_step_the_robot_waits_and_starts_each_answer_at_its_first():
    policy = Policy(n=10)
    played = run(policy, count=12, hz=500, replan_at=8, valid_steps=3)
    assert played == [(k, i) for k in range(4) for i in range(3)]


def test_an_error_of_an_act_is_raised_and_ready_false_asks_nothing_first():
    policy = Policy(n=10, fail_at=1)
    with pytest.raises(RuntimeError, match="the server is gone"):
        play(policy, read=dict, step=lambda a: None, hz=500, replan_at=2, ready=False)
    assert policy.ready == 0


def test_steps_are_the_answer_itself_or_its_one_list_or_its_actions():
    chunk = np.zeros((5, 7))
    assert steps([1, 2]) == [1, 2] and steps(chunk) is chunk
    assert steps({"actions": chunk, "state": "x"}) is chunk
    assert steps({"actions": chunk, "gripper": np.zeros(5)}) is chunk
    assert steps({"arm": chunk}) is chunk
    with pytest.raises(ValueError, match="steps_of"):
        steps({"left": chunk, "right": chunk})
    with pytest.raises(ValueError, match="hz"):
        play(Policy(), read=dict, step=print, hz=0, replan_at=1)
