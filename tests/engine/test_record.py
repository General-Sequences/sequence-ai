# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
The robot's record in its episode (sequence_ai/engine/record.py), the one a deployment's worker and serve() share: its
observations as the robot sent them, compressed, and its acts' answers, in episode.state["seq.observations"] /
["seq.actions"] by step - about its last minute (RECORD_S), the oldest steps first, cut at a keyframe - and, at the end
of an episode a server saves, what else its state holds, none of it cut, made storable once.

    python -m pytest -q tests/engine/test_record.py
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from sequence_ai.engine import record


class Episode:
    """An episode as the record reads one: its id and its state (a deployment's seq.Episode has more)."""

    def __init__(self, id: str) -> None:  # noqa: A002 - the episode's id
        self.id, self.state = id, {}


def _encoder():
    """A camera's H.264 encoder as a robot marks its frames with (sequence_base.codec.marked, which needs PyAV)."""
    marked = pytest.importorskip("sequence_base.codec.marked")
    pytest.importorskip("av")
    return marked


def _video(n: int) -> list[dict]:
    """`n` frames of one camera as the robot marks them: H.264, a keyframe first, its clock still: a second of a busy
    machine's would be one more (marked.KEYFRAME_S)."""
    enc = _encoder().encoder("video", clock=lambda: 0.0)
    frames = []
    for i in range(n):
        img = np.zeros((48, 64, 3), np.uint8)
        img[:, : 4 * (i + 1)] = 200
        frames.append(enc.encode(img))
    return frames


@pytest.fixture
def clock(monkeypatch):
    """The record's clock, set by hand: `clock.t` seconds."""
    c = types.SimpleNamespace(t=0.0)
    monkeypatch.setattr(record, "time", types.SimpleNamespace(monotonic=lambda: c.t))
    return c


# ── the record ───────────────────────────────────────────────────────────────────────────────────────────────────────

def test_observations_by_step_and_each_answer_at_the_newest_step_before_it():
    ep, rec = Episode("e1"), record.Record()
    record.frame(ep, rec, 0, {"x": 0})
    record.answer(ep, rec, [0.1])
    record.frame(ep, rec, 1, {"x": 1})
    record.frame(ep, rec, 2, {"x": 2})
    record.answer(ep, rec, [0.2])
    record.answer(ep, rec, [0.3])                        # a second act at one step: the newer kept
    record.frame(ep, rec, 1, {"x": "again"})             # a step sent again: the newer, in its place
    assert ep.state[record.OBSERVATIONS] == {0: {"x": 0}, 1: {"x": "again"}, 2: {"x": 2}}
    assert ep.state[record.ACTIONS] == {0: [0.1], 2: [0.3]}
    assert list(rec.kept_at) == [0, 1, 2]


def test_past_a_minute_the_oldest_steps_go_first_then_up_to_their_videos_next_keyframe(clock):
    ep, rec = Episode("e1"), record.Record()
    video = _video(5)
    assert [bool(v.get("kf")) for v in video][:2] == [True, False]
    big = {"pad": np.zeros(2**20, np.uint8)}             # no size lets a step go, only its age
    for step in range(5):
        clock.t = float(step)
        record.frame(ep, rec, step, {"cam": video[step], **big})
    assert list(ep.state[record.OBSERVATIONS]) == [0, 1, 2, 3, 4]
    clock.t = record.RECORD_S + 0.5                        # step 0 past the minute: it goes, and 1-4 start mid-stream
    record.trim(ep, rec)
    assert list(ep.state[record.OBSERVATIONS]) == [4]     # never the newest, though it starts mid-stream too
    ep2, rec2 = Episode("e2"), record.Record()
    enc = _encoder().encoder("video", clock=lambda: 0.0)
    for step in range(4):
        if step == 2:
            enc.force_keyframe()
        clock.t = float(step)
        record.frame(ep2, rec2, step, {"cam": enc.encode(np.full((48, 64, 3), step * 40, np.uint8))})
    clock.t = record.RECORD_S + 0.5
    record.trim(ep2, rec2)
    assert list(ep2.state[record.OBSERVATIONS]) == [2, 3]  # cut where a decoder can begin
    assert list(rec2.kept_at) == [2, 3]


def test_what_is_stored_is_data_its_tensors_and_arrays_copied_and_what_is_no_data_named():
    arr = np.arange(4)
    kept, skipped = record.storable({1: arr, "b": b"\x00\x01", "f": open, "n": [1, None, object()], "t": (1, 2.5)})
    arr[0] = 99
    assert kept["1"].tolist() == [0, 1, 2, 3]               # a copy, its keys strings
    assert (kept["b"], kept["t"], kept["n"][:2]) == (b"\x00\x01", [1, 2.5], [1, None])
    assert "f" not in kept and skipped == ["f", "n[2]"]


def test_a_torch_tensor_is_stored_as_a_numpy_array_bfloat16_as_float32():
    torch = pytest.importorskip("torch")
    got, none = record.storable({"w": torch.ones(2, dtype=torch.bfloat16)})
    assert none == [] and got["w"].dtype == np.float32 and got["w"].tolist() == [1.0, 1.0]


def test_an_episodes_state_at_its_end_is_all_its_code_kept_and_its_plans_not_the_record_again():
    ep, rec = Episode("e1"), record.Record()
    record.frame(ep, rec, 0, {"x": 0})
    record.answer(ep, rec, [0.1])
    ep.state["task"] = "pick up the cup"
    ep.state["seq.plans"] = [{"step": 0, "value": {"instruction": "reach"}}]
    ep.state["model"] = object()
    ep.state["big"] = np.zeros(2**20, np.uint8)
    got = record.state_of(ep)
    big = got["state"].pop("big")
    assert got["state"] == {"task": "pick up the cup", "seq.plans": [{"step": 0, "value": {"instruction": "reach"}}]}
    assert big.nbytes == 2**20 and got["skipped"] == ["model"]   # none of it cut: only what is no data is left out
