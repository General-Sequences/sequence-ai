# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
A module's forward compiled, built by TensorRT or run in float16 (compiled.py). Each test pins one guarantee:
  a float16 copy runs the module's inputs cast in and its outputs cast back, the module keeping its own weights, and
  the copy takes no forward put in front of a module's own;
  torch.compile is given the module's own forward and the cache it is to keep;
  an engine's key changes with the weights and the shapes it was built for;
  TensorRT not installed is said with the torch-tensorrt made for the torch installed here;
  a module of keyword arguments alone is built from its tensors in order, with no precision named beside explicit
  typing, and a call unlike the one it was built for runs the module's own forward;
  tensors inside other values are refused before anything is built.

    python -m pytest -q tests/engine/test_compiled.py
"""
from __future__ import annotations

import importlib.util
import re
import sys

import pytest

torch = pytest.importorskip("torch")

from sequence_ai.engine import compiled  # noqa: E402


def test_a_float16_copy_runs_in_place_of_the_module_and_the_module_keeps_its_weights():
    torch.manual_seed(0)
    module = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.GELU(), torch.nn.Linear(8, 4))
    module[0].forward = lambda x: x                                   # put in front of its own: not copied
    copy = compiled.dtype_copy(module, torch.float16)
    assert "forward" in vars(module[0]) and "forward" not in vars(copy[0])
    assert copy[0].weight.dtype == torch.float16 and module[0].weight.dtype == torch.float32
    del module[0].forward
    forward = compiled.cast_around(copy, torch.float32)
    x = torch.randn(3, 8)
    got = forward(x)
    assert got.dtype == torch.float32 and torch.allclose(got, module(x), atol=1e-2)


def test_torch_compile_is_given_the_modules_own_forward_and_its_cache(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    monkeypatch.setattr(torch, "compile", lambda fn, **options: seen.update(fn=fn, **options) or fn)
    module = torch.nn.Linear(2, 2)
    module.forward = lambda x: x
    compiled.torch_compile(module, cache=tmp_path)
    assert seen["fn"].__func__ is torch.nn.Linear.forward and seen["dynamic"] is False
    import os

    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(tmp_path)


def test_an_engines_key_changes_with_its_weights_and_shapes():
    module = torch.nn.Linear(4, 4)
    key = compiled._engine_key(module, (torch.zeros(2, 4),), {}, "fp32", "1")
    assert key == compiled._engine_key(module, (torch.ones(2, 4),), {}, "fp32", "1")      # values are not shapes
    assert key != compiled._engine_key(module, (torch.zeros(3, 4),), {}, "fp32", "1")
    assert key != compiled._engine_key(module, (torch.zeros(2, 4),), {}, "fp16", "1")
    with torch.no_grad():
        module.weight.add_(1.0)
    assert key != compiled._engine_key(module, (torch.zeros(2, 4),), {}, "fp32", "1")


def test_tensorrt_not_installed_says_the_torch_tensorrt_made_for_the_torch_installed(monkeypatch):
    if importlib.util.find_spec("torch_tensorrt") is not None:
        pytest.skip("TensorRT is installed here")
    with pytest.raises(RuntimeError, match=re.escape(f"TensorRT is not installed: {compiled.tensorrt_install()}")):
        compiled.tensorrt(torch.nn.Linear(2, 2), (torch.zeros(1, 2),), {})
    # one torch-tensorrt release for each torch minor version, read from torch's own metadata
    for installed, minor in (("2.9.1", "2.9"), ("2.10.0+cu128", "2.10"), ("2.11.0.dev20251201+cpu", "2.11")):
        monkeypatch.setattr(compiled.importlib.metadata, "version", lambda name, v=installed: v)
        said = compiled.tensorrt_install()
        assert said == f'pip install "torch-tensorrt=={minor}.*" (the one for your torch {minor})'

    def missing(name):
        raise compiled.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(compiled.importlib.metadata, "version", missing)
    assert compiled.tensorrt_install().startswith("pip install torch first")


class _FakeTensorRT:
    """torch_tensorrt as far as tensorrt() uses it: compile keeps its settings and gives back the module it was given,
    run eagerly, counting its calls; nothing can be saved."""
    __version__ = "fake"

    def __init__(self) -> None:
        self.settings, self.calls = None, 0

    def compile(self, module, **settings):
        self.settings = settings
        import inspect

        # TensorRT's export gives each parameter of the forward its input: one positional parameter a tensor, and the
        # module exports with them
        params = list(inspect.signature(module.forward).parameters.values())
        assert all(p.kind is p.POSITIONAL_OR_KEYWORD for p in params) and len(params) == len(settings["inputs"])
        torch.export.export(module, tuple(settings["inputs"]))

        def engine(*tensors):
            self.calls += 1
            return module(*tensors)

        return engine

    def save(self, built, path, **kwargs):
        raise OSError("nowhere to keep it")


class _KeywordsOnly(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, *, hidden, scale=1.0):
        return self.linear(hidden) * scale


def test_a_module_of_keyword_arguments_alone_is_built_from_its_tensors_in_order(monkeypatch, tmp_path):
    fake = _FakeTensorRT()
    monkeypatch.setitem(sys.modules, "torch_tensorrt", fake)
    module, hidden = _KeywordsOnly(), torch.randn(2, 4)
    run = compiled.tensorrt(module, (), {"hidden": hidden, "scale": 2.0}, cache=tmp_path, say=lambda line: None)
    assert [tuple(t.shape) for t in fake.settings["inputs"]] == [(2, 4)]
    assert fake.settings["use_explicit_typing"] and "enabled_precisions" not in fake.settings
    assert torch.equal(run(hidden=hidden, scale=2.0), module(hidden=hidden, scale=2.0)) and fake.calls == 1
    wider = torch.ones(3, 4)                                     # another shape, then another value: the module's own
    assert torch.equal(run(hidden=wider, scale=2.0), module(hidden=wider, scale=2.0))
    assert torch.equal(run(hidden=hidden, scale=3.0), module(hidden=hidden, scale=3.0)) and fake.calls == 1


def test_tensors_inside_other_values_are_refused_before_anything_is_built(monkeypatch, tmp_path):
    fake = _FakeTensorRT()
    monkeypatch.setitem(sys.modules, "torch_tensorrt", fake)
    with pytest.raises(ValueError, match="tensors inside other values"):
        compiled.tensorrt(torch.nn.Linear(4, 4), ({"x": torch.zeros(1, 4)},), {}, cache=tmp_path, say=lambda line: None)
    assert fake.settings is None


def test_a_float16_copys_outputs_come_back_in_the_modules_dtype_inside_a_models_output_objects():
    import collections

    class Output(collections.OrderedDict):                 # as transformers' ModelOutput is
        pass

    Pair = collections.namedtuple("Pair", "a b")

    def fn(x):
        return Output(hidden=x * 2, pair=Pair(x, x.int()), rest=[x])

    out = compiled.cast_around(fn, torch.bfloat16)(torch.ones(2, dtype=torch.bfloat16))
    assert type(out) is Output and out["hidden"].dtype == torch.bfloat16 and type(out["pair"]) is Pair
    assert out["pair"].a.dtype == torch.bfloat16 and out["pair"].b.dtype == torch.int32
    assert out["rest"][0].dtype == torch.bfloat16
