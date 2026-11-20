# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
What a policy of its own inputs and outputs declares of them (sequence_ai/engine/convert.py), done where its code runs,
last before its code and first after it: an observation annotated torch.Tensor arrives a torch tensor on its device,
float64 as float32, a batch dimension of 1 first - not for a @seq.concurrent policy, whose list is its batch - and each
tensor it answers with leaves as an array, that dimension taken off. A policy that takes no tensors is handed its
inputs untouched. The tests of tensors need torch: they are skipped without it.

    python -m pytest -q tests/engine/test_convert.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from sequence_ai.engine import convert
from sequence_ai.engine.convert import Conversions, InputsRejected


class _Annotated:
    """infer methods annotated as a policy's code annotates them: torch tensors, alone, in a dict, or by name."""

    def alone(self, obs: "torch.Tensor", *, seed=None): ...  # noqa: F821 - read as text, never imported

    def keyed(self, obs: "dict[str, torch.Tensor]"): ...  # noqa: F821

    def named(self, obs: "Tensor"): ...  # noqa: F821

    def plain(self, obs, *, seed=None): ...

    def arrays(self, obs: "dict[str, np.ndarray]"): ...

    def other(self, obs: "tf.Tensor"): ...  # noqa: F821


def test_an_infer_annotated_torch_tensor_is_handed_tensors_and_any_other_nothing():
    """Whether a policy's inputs reach it as torch tensors is read off its infer - or its observe method's observation
    where it has one - with nothing declared beside."""
    a = _Annotated()
    assert [convert.wants_tensors(m) for m in (a.alone, a.keyed, a.named)] == [True, True, True]
    assert [convert.wants_tensors(m) for m in (a.plain, a.arrays, a.other)] == [False, False, False]
    assert convert.of(a.plain) is None and convert.of(a.alone, observe=a.plain) is None    # observe's, if any


def test_what_cannot_be_converted_is_refused_at_load():
    with pytest.raises(TypeError, match="written against the robot-arm preset"):
        convert.of(_Annotated().alone, io_mode="robot_arm")


def _tensors(batch: bool):
    return Conversions(tensors=True, batch=batch)


def test_each_array_reaches_the_policy_a_tensor_on_its_device_float64_as_float32():
    torch = pytest.importorskip("torch")

    c = _tensors(batch=False)
    out = c.observation({"head": np.zeros((6, 8, 3), np.uint8), "state": np.array([0.25, 0.5]), "step": np.array([3]),
                         "nested": {"q": [np.zeros(2, dtype=np.float16)]}, "task": "stack", "gains": [1.0, 2.0]})
    assert (out["head"].dtype, tuple(out["head"].shape)) == (torch.uint8, (6, 8, 3))             # as it came: HWC
    assert out["state"].dtype == torch.float32 and out["state"].tolist() == [0.25, 0.5]
    assert out["step"].dtype == torch.int64 and out["nested"]["q"][0].dtype == torch.float16    # other dtypes kept
    assert (out["task"], out["gains"]) == ("stack", [1.0, 2.0])                                  # no array: untouched
    assert {t.device.type for t in (out["head"], out["state"])} == {"cpu"}                       # no card here


def test_a_batch_dimension_comes_first():
    torch = pytest.importorskip("torch")

    out = _tensors(batch=True).observation({"head": np.zeros((6, 8, 3), np.uint8), "state": np.array([1.0, 2.0])})
    assert (out["head"].dtype, tuple(out["head"].shape)) == (torch.uint8, (1, 6, 8, 3))
    assert tuple(out["state"].shape) == (1, 2)


def test_an_array_read_off_the_wire_is_copied_not_shared_read_only():
    pytest.importorskip("torch")
    c = _tensors(batch=False)
    wire = np.frombuffer(np.array([1.0, 2.0], dtype=np.float32).tobytes(), dtype=np.float32)
    assert not wire.flags.writeable
    with warnings.catch_warnings():
        warnings.simplefilter("error")                  # torch warns of a tensor over a read-only array
        t = c.observation({"x": wire})["x"]
    t.add_(1)
    assert wire.tolist() == [1.0, 2.0]


def test_the_device_is_the_card_when_there_is_one(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert Conversions(tensors=True)._device.type == "cuda"


def test_each_tensor_it_answers_with_leaves_an_array_the_batch_dimension_taken_off():
    torch = pytest.importorskip("torch")

    out = _tensors(batch=True).answer({"actions": torch.ones(1, 4, 7, requires_grad=True), "value": torch.tensor(0.5),
                                       "half": [torch.zeros(1, 2, dtype=torch.bfloat16)], "id": "a", "raw": np.ones(2)})
    assert isinstance(out["actions"], np.ndarray) and out["actions"].shape == (4, 7)
    assert out["value"].shape == () and float(out["value"]) == 0.5                  # no dimension to take off
    assert out["half"][0].dtype == np.float32 and out["half"][0].shape == (2,)      # numpy has no bfloat16
    assert out["id"] == "a" and out["raw"].tolist() == [1.0, 1.0]
    assert _tensors(batch=False).answer(torch.zeros(4, 7)).shape == (4, 7)          # none was added: none taken off


def test_a_tensor_answer_that_does_not_lead_with_its_batch_dimension_is_refused_naming_its_shape():
    torch = pytest.importorskip("torch")

    with pytest.raises(ValueError, match=r"the answer\.actions has shape \(4, 7\)"):
        _tensors(batch=True).answer({"actions": torch.zeros(4, 7)})


def test_without_tensors_an_answer_leaves_as_the_policy_returned_it():
    answer = {"actions": [[0.0] * 7]}
    assert Conversions().answer(answer) is answer


def test_a_conversion_error_is_the_callers_inputs_rejected_with_its_reason():
    err = convert.ConversionError("the observation has no 'state'", "inputs")
    assert isinstance(err, InputsRejected) and isinstance(err, ValueError)
    assert (str(err), err.reason) == ("the observation has no 'state'", "inputs")
    assert InputsRejected("x").reason == "observation"
