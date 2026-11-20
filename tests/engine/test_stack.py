# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
A @seq.concurrent policy's batch stacked by the server (sequence_ai/engine/stack.py): arrays along a new first axis,
numbers too, text a list; a first axis that differs padded, its mask beside it; a key not every act sends a list with
None where one sent none; and the answer split back one an act - a list taken as one an act, never one object handed to
every act.

    python -m pytest -q tests/engine/test_stack.py
"""
from __future__ import annotations

import numpy as np
import pytest

from sequence_ai.engine import stack


def _act(i, history=2):
    """One act's inputs as infer is handed them: two cameras, a state vector, its instruction, its step, a history."""
    return {"front": np.full((4, 6, 3), i, np.uint8), "wrist": np.full((4, 6, 3), 10 + i, np.uint8),
            "state": np.arange(3, dtype=np.float32) + i, "prompt": f"pick {i}", "step": i,
            "history": np.ones((history, 2), np.float32) * i}


def test_arrays_stack_along_a_new_first_axis_numbers_too_and_text_stays_a_list():
    b = stack.stack([_act(i) for i in range(3)])
    assert b["front"].shape == (3, 4, 6, 3) and b["front"].dtype == np.uint8 and b["front"][2, 0, 0, 0] == 2
    assert b["state"].shape == (3, 3) and b["state"].dtype == np.float32
    assert b["step"].tolist() == [0, 1, 2] and b["prompt"] == ["pick 0", "pick 1", "pick 2"]
    assert "history_mask" not in b                 # every history as long: nothing padded, no mask


def test_a_first_axis_that_differs_is_padded_with_zeros_and_its_mask_says_what_is_real():
    b = stack.stack([_act(1, history=1), _act(2, history=3), _act(3, history=2)])
    assert b["history"].shape == (3, 3, 2)
    assert b["history_mask"].tolist() == [[True, False, False], [True, True, True], [True, True, False]]
    assert b["history"][0, 1:].tolist() == [[0, 0], [0, 0]] and b["history"][1].tolist() == [[2, 2]] * 3


def test_a_key_not_every_act_sends_comes_as_a_list_with_none_where_one_sent_none():
    """pi05-droid-qwen's robots send a prompt connected to it, none through its planner, and share a batch: the prompt
    comes as a list, None for the act that sent none - and an array no act's own is never made up."""
    b = stack.stack([_act(0), {k: v for k, v in _act(1).items() if k not in ("prompt", "wrist")}])
    assert b["prompt"] == ["pick 0", None] and b["wrist"][0].shape == (4, 6, 3) and b["wrist"][1] is None
    assert b["front"].shape == (2, 4, 6, 3)                          # what both sent, stacked as ever


def test_an_answer_splits_back_one_an_act_and_one_object_is_never_every_acts():
    acts = [_act(i) for i in range(3)]
    b = stack.stack(acts)
    answers = stack.unstack({"actions": b["state"] * 2, "gripper": b["front"][:, 0, 0, 0], "note": "ok"}, 3)
    assert [a["actions"].tolist() for a in answers] == [(r["state"] * 2).tolist() for r in acts]
    assert [int(a["gripper"]) for a in answers] == [0, 1, 2] and [a["note"] for a in answers] == ["ok"] * 3
    assert stack.unstack([{"a": 1}, {"a": 2}], 2) == [{"a": 1}, {"a": 2}]       # a list: one an act already
    with pytest.raises(stack.Unstackable, match="2 rows for a batch of 3 acts"):
        stack.unstack(np.zeros((2, 4)), 3)
    with pytest.raises(stack.Unstackable, match="a list of 1 for a batch of 2 acts"):
        stack.unstack([{"a": 1}], 2)

    class Chunk:                                    # a robot-arm preset's ActionChunk: one act's answer
        pass

    with pytest.raises(stack.Unstackable, match="one Chunk for a batch of 2 acts: answer a list of 2"):
        stack.unstack(Chunk(), 2)


def test_what_cannot_be_one_batch_says_why_and_names_collate_false():
    with pytest.raises(stack.Unstackable,
                       match=r"front, arrays of shapes \(4, 6, 3\), \(5, 7, 3\) cannot be one batch"):
        stack.stack([_act(0), {**_act(1), "front": np.zeros((5, 7, 3), np.uint8)}])
    with pytest.raises(stack.Unstackable, match="one act has int where another has a dict.*collate=False"):
        stack.stack([_act(0), 1])
    with pytest.raises(stack.Unstackable, match="history_mask is a key of the acts already"):
        stack.stack([{**_act(0, history=1), "history_mask": 1}, {**_act(1, history=2), "history_mask": 1}])
