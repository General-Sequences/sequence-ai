# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
What a plan is handed (sequence_ai/engine/robot_view.py): one robot's episode as it is the moment the plan is called -
its state copied and read-only, the server's three keys as tuples oldest first, the newest observation as its act got
it in place of the one the robot sent - so the act, writing the live state on another thread, never meets the plan.

    python -m pytest -q tests/engine/test_robot_view.py
"""
from __future__ import annotations

import pytest

from sequence_ai.engine import record
from sequence_ai.engine.robot_view import Robot, view


class Episode:
    """An episode as the view reads one: its id and its state."""

    def __init__(self, id: str) -> None:  # noqa: A002 - the episode's id
        self.id, self.state = id, {}


def test_the_view_is_the_episodes_state_read_only_its_records_oldest_first_the_newest_as_its_act_got_it():
    episode = Episode("ep-1")
    episode.state.update({"task_user": "pick", record.OBSERVATIONS: {3: {"x": "3 as it came"}, 1: {"x": 1}},
                          record.ACTIONS: {3: [3], 1: [1]}, record.PLANS: [{"step": 1, "value": "go"}]})
    robot = view(episode, latest={"x": "3 decoded"})
    assert isinstance(robot, Robot) and robot.id == "ep-1" and repr(robot) == "Robot('ep-1')"
    assert robot.state["task_user"] == "pick"
    assert robot.state[record.OBSERVATIONS] == ({"x": 1}, {"x": "3 decoded"})     # oldest first, the newest decoded
    assert robot.state[record.ACTIONS] == ([1], [3])
    assert robot.state[record.PLANS] == ({"step": 1, "value": "go"},)
    with pytest.raises(TypeError):
        robot.state["task_user"] = "drop"                                         # read-only
    episode.state[record.PLANS].append({"step": 3, "value": "lift"})
    assert len(robot.state[record.PLANS]) == 1                          # a copy: the act's writes stay its own
    assert isinstance(episode.state[record.OBSERVATIONS], dict)                   # the live state as it was


def test_an_episode_with_nothing_kept_yet_is_seen_with_three_empty_records():
    robot = view(Episode("ep-2"))
    assert [robot.state[k] for k in (record.OBSERVATIONS, record.ACTIONS, record.PLANS)] == [(), (), ()]
