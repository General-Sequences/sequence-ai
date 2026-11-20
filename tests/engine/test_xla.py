# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
A JAX policy's switches (xla.py, and optimize()'s path for it), on jax's CPU backend. Each test pins one guarantee:
  the functions jax.jit made are found where an openpi policy holds them - in a closure beside its frozen parameters -
  with the typed random key it splits on every call, and put back as they were;
  a function compiled with XLA's command-buffer options answers as the function itself does;
  with its key set from each run's seed, two eager runs of an observation draw alike, so the graphs can be checked;
  the command buffers stay on where they are faster and the outputs the same, with what each asked for named;
  precision casts the parameters the jitted function is given, judged within the tolerance, and put back when off;
  torch_compile, fused_kernels and tensorrt say what XLA does already, or that it has no counterpart.

    python -m pytest -q tests/engine/test_xla.py
"""
from __future__ import annotations

import threading
import time

import pytest

jax = pytest.importorskip("jax")
np = pytest.importorskip("numpy")
import jax.numpy as jnp  # noqa: E402

from sequence_ai.engine import xla  # noqa: E402
from sequence_ai.engine.optimize import apply, optimize  # noqa: E402

FAST = threading.local()


def _module_jit(fn, state):
    """openpi's module_jit, as far as it matters here: jax.jit of a function of the model's state, the state frozen in
    the closure of the wrapper the policy calls."""
    jitted_fn = jax.jit(fn)

    def wrapper(*args):
        return jitted_fn(state, *args)

    return wrapper


def _sample(state, key, x):
    noise = jax.random.normal(key, x.shape)
    for _ in range(4):
        noise = noise + 0.1 * jnp.tanh(noise @ state["w"] + x)
    return noise


class Policy:
    """openpi's Policy, as far as optimize() sees it: its jitted sampling and a typed key split on every call."""

    def __init__(self) -> None:
        state = {"w": jax.random.normal(jax.random.key(1), (64, 64)) * 0.1}
        self._sample_actions = _module_jit(_sample, state)
        self._rng = jax.random.key(0)

    def infer(self, observation):
        self._rng, key = jax.random.split(self._rng)
        return {"actions": np.asarray(self._sample_actions(key, jnp.asarray(observation["state"])))}


def _examples(n=3):
    return [{"state": np.full((1, 64), i / 10, dtype=np.float32)} for i in range(n)]


def test_the_jitted_function_its_parameters_and_its_key_are_found_where_the_policy_holds_them():
    policy = Policy()
    found = xla.find({"self": policy})
    assert len(found.jitted) == 1 and found.jitted[0].kind == "cell"
    assert len(found.trees) == 1 and found.trees[0].kind == "cell"
    assert [(p.kind, p.key) for p in found.keys] == [("attr", "_rng")]
    first = policy._rng
    found.seed(5)
    assert not bool(jnp.all(jax.random.key_data(policy._rng) == jax.random.key_data(first)))
    found.put_back_keys()
    assert policy._rng is first


def test_a_function_compiled_with_command_buffer_options_answers_as_the_function_does():
    policy = Policy()
    place = xla.find({"self": policy}).jitted[0]
    buffered = xla.Buffered(place.value, xla.GRAPHS)
    state = xla.find({"self": policy}).trees[0].value
    key, x = jax.random.key(3), jnp.ones((1, 64))
    assert np.array_equal(np.asarray(buffered(state, key, x)), np.asarray(place.value(state, key, x)))
    assert buffered.jitted is not None                       # jitted again from the function jax.jit was given
    unwrapped = xla.Buffered(_Opaque(place.value), xla.GRAPHS)             # one whose original is not to be had
    assert np.array_equal(np.asarray(unwrapped(state, key, x)), np.asarray(place.value(state, key, x)))
    unwrapped(state, key, x)
    assert len(unwrapped.compiled) == 1                      # lowered and compiled once for one signature


def test_whole_graph_takes_the_first_way_of_naming_loops_that_puts_a_loop_in_a_command_buffer(monkeypatch):
    tried = []

    def buffers(options):
        tried.append(options["xla_gpu_enable_command_buffer"])
        return options["xla_gpu_enable_command_buffer"] == "6"              # an XLA that reads only numbers

    xla._WHOLE.clear()
    monkeypatch.setattr(xla, "_buffers_loops", buffers)
    assert xla.whole() == {"xla_gpu_enable_command_buffer": "6", "xla_gpu_graph_min_graph_size": 1}
    assert xla.whole() is xla.whole() and tried == ["WHILE", "6"]           # asked once
    xla._WHOLE.clear()
    monkeypatch.setattr(xla, "_buffers_loops", lambda options: False)
    assert xla.whole() is None                                               # no GPU, or an XLA that cannot
    xla._WHOLE.clear()


def test_a_loop_is_in_no_command_buffer_on_a_cpu_and_a_kind_xla_refuses_is_no_error():
    assert not xla._buffers_loops({"xla_gpu_enable_command_buffer": "WHILE", **xla.GRAPHS})
    assert not xla._buffers_loops({"xla_gpu_enable_command_buffer": "NO_SUCH_KIND", **xla.GRAPHS})


class _Opaque:
    """A jitted function that does not say what it jitted."""

    def __init__(self, fn) -> None:
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def lower(self, *args, **kwargs):
        return self.fn.lower(*args, **kwargs)

    def eval_shape(self, *args, **kwargs):
        return self.fn.eval_shape(*args, **kwargs)


class _Quick:
    """A stand-in for Buffered: the same function without the time its launches take (FAST)."""

    def __init__(self, fn, options) -> None:
        self.fn, self.options = fn, dict(options)

    def __call__(self, *args, **kwargs):
        FAST.on = True
        try:
            return self.fn(*args, **kwargs)
        finally:
            FAST.on = False

    def lower(self, *args, **kwargs):
        return self.fn.lower(*args, **kwargs)

    def eval_shape(self, *args, **kwargs):
        return self.fn.eval_shape(*args, **kwargs)


class _Launches:
    """A jitted function that waits as its launches would, unless replayed from a command buffer (FAST)."""

    def __init__(self, fn) -> None:
        self.fn = fn

    def __call__(self, *args, **kwargs):
        if not getattr(FAST, "on", False):
            time.sleep(0.004)
        return self.fn(*args, **kwargs)

    def lower(self, *args, **kwargs):
        return self.fn.lower(*args, **kwargs)

    def eval_shape(self, *args, **kwargs):
        return self.fn.eval_shape(*args, **kwargs)


def _slow_policy():
    policy = Policy()
    place = xla.find({"self": policy}).jitted[0]
    place.set(_Launches(place.value))
    return policy


def test_command_buffers_stay_on_where_they_are_faster_and_the_outputs_the_same(monkeypatch):
    monkeypatch.setattr(xla, "gpu", lambda: True)
    monkeypatch.setattr(xla, "Buffered", _Quick)
    monkeypatch.setattr(xla, "whole", lambda: {"xla_gpu_enable_command_buffer": "WHILE", **xla.GRAPHS})
    policy = _slow_policy()
    fast = optimize(policy.infer, _examples(), cuda_graphs=True, whole_graph=True, say=lambda line: None)
    switch = fast.report["switches"]["cuda_graphs"]
    assert switch["on"] and switch["xla_options"] == {"xla_gpu_graph_min_graph_size": 1}
    # its while loops add nothing here: judged, and no faster than the straight-line buffers
    assert not fast.report["switches"]["whole_graph"]["on"] and "faster" in fast.report["switches"]["whole_graph"]["why"]
    assert "not_compared" not in fast.report                  # the key set from each seed: two eager runs drew alike


def test_precision_casts_the_parameters_judged_within_the_tolerance_and_put_back_when_off():
    policy = Policy()
    tree = xla.find({"self": policy}).trees[0]
    fast = optimize(policy.infer, _examples(), precision="fp16", tolerance=0.05, cuda_graphs=False,
                    whole_graph=False, say=lambda line: None)
    switch = fast.report["switches"]["precision"]
    assert switch["precision"] == "fp16" and switch["trees"] == 1 and 0 < switch["differences"]["max_abs"] < 0.05
    if not switch["on"]:
        assert tree.get() is tree.value and tree.get()["w"].dtype == jnp.float32


def test_the_switches_without_a_counterpart_in_xla_say_so():
    fast = optimize(Policy().infer, _examples(), torch_compile=True, fused_kernels=True, tensorrt=True, tolerance=0.1,
                    cuda_graphs=False, whole_graph=False, say=lambda line: None)
    whys = {name: fast.report["switches"][name]["why"] for name in ("torch_compile", "fused_kernels", "tensorrt")}
    assert whys == {"torch_compile": "a JAX model: XLA compiles each jitted function whole already",
                    "fused_kernels": "a JAX model: XLA fuses its kernels already",
                    "tensorrt": "a JAX model: TensorRT builds torch modules, and has no counterpart for JAX"}
    fp8 = optimize(Policy().infer, _examples(), precision="fp8", tolerance=0.1, cuda_graphs=False, whole_graph=False,
                   say=lambda line: None)
    assert fp8.report["switches"]["precision"]["why"].startswith("a JAX model: fp8 needs scaled multiplies")


def test_apply_compiles_a_jax_policys_functions_with_the_options_a_check_kept():
    report = {"switches": {"cuda_graphs": {"on": True, "xla_options": dict(xla.GRAPHS)},
                           "whole_graph": {"on": False, "why": "no faster with it"}}}
    policy, other = Policy(), Policy()
    applied = apply(policy.infer, report, say=lambda line: None)
    assert applied.on == ["cuda_graphs"] and applied.report["switches"]["cuda_graphs"]["xla_options"] == xla.GRAPHS
    assert isinstance(xla.find({"self": policy}).jitted[0].value, xla.Buffered)
    assert np.array_equal(policy.infer(_examples()[1])["actions"], other.infer(_examples()[1])["actions"])


def test_a_jax_policys_command_buffers_are_tried_only_when_asked_and_nothing_is_run_or_compiled_for_them(monkeypatch):
    from sequence_ai.engine.optimize import JAX_GRAPHS

    monkeypatch.setattr(xla, "gpu", lambda: True)
    monkeypatch.setattr(xla, "Buffered", lambda *a, **k: (_ for _ in ()).throw(AssertionError("compiled")))
    policy, said = Policy(), []
    calls = []
    infer = policy.infer
    fast = optimize(lambda observation: calls.append(1) or infer(observation), _examples(), say=said.append)
    assert fast.on == [] and calls == []                                       # nothing run, nothing compiled
    for name in ("cuda_graphs", "whole_graph"):
        assert fast.report["switches"][name]["why"] == JAX_GRAPHS
    assert said == [f"optimize: {name} off - {JAX_GRAPHS}" for name in ("cuda_graphs", "whole_graph")]
