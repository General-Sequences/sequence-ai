# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
A policy made faster, each switch checked on its own observations before it is turned on (optimize.py). Each test pins
one guarantee, with the CUDA graph replaced by a stand-in (`capture`), so they run with no GPU:
  both switches turn on where the outputs stay the same, bit for bit, and the act is at least 10% faster - the modules
  the model calls again and again (cuda_graphs), and the run's whole call, its noise drawn in it (whole_graph);
  a switch whose outputs differ on an observation it was not captured on, or that is no faster, is taken out again, the
  report saying why, and the model is as it was;
  a whole call whose replay does not draw the random numbers eager mode drew is never kept;
  a switch that changes the outputs is timed with cuda_graphs on in both turns where cuda_graphs is tried - as it will
  run - and eager mode's time is still its own; where those graphs do not stay on, the act is timed again as it runs;
  a policy whose eager outputs change from run to run, one with no torch module, one with no observation to check on,
  and a machine with no GPU turn nothing on, each saying why;
  a batch function is checked at each of its sizes;
  the random states after optimize() are what they would have been without it;
  the modules are found in a function's closure and globals, a bound method's object and a partial's arguments.

    python -m pytest -q tests/engine/test_optimize.py
"""
from __future__ import annotations

import functools
import json
import sys
import threading
import time

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from sequence_ai.engine import cuda_graphs  # noqa: E402
from sequence_ai.engine.optimize import Optimized, _namespace, apply, differences, optimize  # noqa: E402

# While set, the stand-in modules skip their sleep: what a replayed graph saves, the time of launching its kernels.
FAST = threading.local()
SLEEP_S = 0.004


def _slow() -> None:
    if not getattr(FAST, "on", False):
        time.sleep(SLEEP_S)


class _Replay:
    """A graph that replays its call by running it again, without the time its kernels take to launch (FAST)."""

    def __init__(self, inner, args, kwargs, pool) -> None:
        self.inner = inner

    def replay(self, args, kwargs):
        FAST.on = True
        try:
            return self.inner(*args, **kwargs)
        finally:
            FAST.on = False


class _Slowly(_Replay):
    """A graph that saves nothing: its replay takes as long as eager mode."""

    def replay(self, args, kwargs):
        return self.inner(*args, **kwargs)


class _Captured(_Replay):
    """A graph that returns what it computed on the inputs it was captured on, whatever it is given: right on those
    inputs, wrong on every other - and drawing no random number when replayed."""

    def __init__(self, inner, args, kwargs, pool) -> None:
        super().__init__(inner, args, kwargs, pool)
        FAST.on = True
        try:
            self.out = inner(*args, **kwargs)
        finally:
            FAST.on = False

    def replay(self, args, kwargs):
        return self.out.clone()


class _Denoiser(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(8, 8)

    def forward(self, x):
        _slow()
        return x + 0.1 * torch.tanh(self.lin(x))


class _Encoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(4, 8)

    def forward(self, state):
        _slow()
        return self.lin(state)


class _Flow(torch.nn.Module):
    """A flow policy's model: its run is a method, not its forward - the noise drawn in it, the state encoded once,
    the denoiser called 4 times."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.encoder, self.dit = _Encoder(), _Denoiser()

    def sample(self, state):
        x = torch.randn(state.shape[0], 8)
        c = self.encoder(state)
        for _ in range(4):
            x = self.dit(x + c)
        return x


def _policy_of(model):
    def policy(observation):
        with torch.no_grad():
            return {"actions": model.sample(torch.as_tensor(observation["state"])[None])[0]}
    return policy


def _examples(n=4):
    return [{"state": np.full(4, i / 10, dtype=np.float32)} for i in range(n)]


def _eager(policy, examples, seeds):
    out = []
    for obs, seed in zip(examples, seeds):
        torch.manual_seed(seed)
        out.append(policy(dict(obs))["actions"].clone())
    return out


def test_both_switches_turn_on_where_the_outputs_stay_the_same_and_the_act_is_faster():
    model = _Flow().eval()
    policy, examples = _policy_of(model), _examples()
    want = _eager(policy, examples, [7, 8, 9, 10])
    said = []
    fast = optimize(policy, examples, capture=_Replay, say=said.append)
    assert isinstance(fast, Optimized) and fast.on == ["cuda_graphs", "whole_graph"]
    report = fast.report["switches"]
    assert report["cuda_graphs"]["modules"] == ["model.dit"] and report["whole_graph"]["call"] == "model.sample"
    assert report["cuda_graphs"]["speedup"] >= 1.1 and report["whole_graph"]["speedup"] >= 1.1
    assert fast.report["note"].startswith("checked on 4 observations")
    assert isinstance(vars(model.dit)["forward"], cuda_graphs.Graphed)           # in place: the policy runs it too
    assert isinstance(vars(model)["sample"], cuda_graphs.Graphed)
    assert [line.split(" - ")[0] for line in said] == ["optimize: cuda_graphs on", "optimize: whole_graph on"]
    assert all(torch.equal(a, b) for a, b in zip(_eager(fast, examples, [7, 8, 9, 10]), want))


def test_a_graph_right_only_for_the_inputs_it_was_captured_on_is_taken_out_and_the_model_is_as_it_was():
    model = _Flow().eval()
    policy = _policy_of(model)
    fast = optimize(policy, _examples(), capture=_Captured, say=lambda line: None)
    assert fast.on == []
    for name in ("cuda_graphs", "whole_graph"):
        assert not fast.report["switches"][name]["on"]
    # its first call in run 1 captured, its 3 other calls in that run replayed on inputs it was not captured on
    assert "differed from eager mode's on run 1 of 8" in fast.report["switches"]["cuda_graphs"]["why"]
    assert "forward" not in vars(model.dit) and "sample" not in vars(model)


def test_a_switch_that_is_no_faster_is_taken_out():
    global SLEEP_S
    SLEEP_S, kept = 0.01, SLEEP_S
    try:
        model = _Flow().eval()
        fast = optimize(_policy_of(model), _examples(2), capture=_Slowly, say=lambda line: None, whole_graph=False)
    finally:
        SLEEP_S = kept
    why = fast.report["switches"]["cuda_graphs"]["why"]
    assert not fast.report["switches"]["cuda_graphs"]["on"] and ("no faster" in why or "under the 10%" in why)
    assert "forward" not in vars(model.dit)


def test_a_whole_call_whose_replay_does_not_draw_the_noise_eager_mode_drew_is_never_kept():
    model = _Flow().eval()
    fast = optimize(_policy_of(model), _examples(), capture=_Captured, cuda_graphs=False, say=lambda line: None)
    switch = fast.report["switches"]["whole_graph"]
    assert not switch["on"] and "differed" in switch["why"]
    assert "sample" not in vars(model)


def test_a_policy_whose_eager_outputs_change_turns_nothing_on():
    model = _Flow().eval()

    def policy(observation):
        torch.manual_seed(time.perf_counter_ns())                 # its own seed each call: never the same twice
        return {"actions": model.sample(torch.as_tensor(observation["state"])[None])}

    fast = optimize(policy, _examples(), capture=_Replay, say=lambda line: None)
    assert fast.on == [] and "differ from one eager run to the next" in fast.report["switches"]["cuda_graphs"]["why"]


def test_nothing_is_turned_on_with_no_torch_module_no_observation_or_every_switch_off():
    def numpy_policy(observation):
        return {"actions": observation["state"] * 2}

    said = []
    assert optimize(numpy_policy, _examples(), say=said.append).report["switches"]["cuda_graphs"]["why"] \
        .startswith("the policy holds no torch module")          # ", and no function jax.jit made" where jax is in
    policy = _policy_of(_Flow().eval())
    assert optimize(policy, None, capture=_Replay, say=said.append).report["switches"]["whole_graph"]["why"] \
        .startswith("no observation to check the outputs on")
    off = optimize(policy, _examples(), capture=_Replay, cuda_graphs=False, whole_graph=False, say=said.append)
    assert off.on == [] and off.report["switches"]["cuda_graphs"]["why"] == "turned off"


def test_with_no_gpu_nothing_is_turned_on():
    if torch.cuda.is_available():
        pytest.skip("this machine has a GPU")
    fast = optimize(_policy_of(_Flow().eval()), _examples(), say=lambda line: None)
    assert fast.on == [] and fast.report["switches"]["cuda_graphs"]["why"] == "this machine has no CUDA GPU"


def test_a_batch_function_is_checked_at_each_of_its_sizes():
    model = _Flow().eval()

    def batched(observations):
        with torch.no_grad():
            states = torch.as_tensor(np.stack([o["state"] for o in observations]))
            return [{"actions": row} for row in model.sample(states)]

    fast = optimize(batched, _examples(), batch=[1, 2, 4], capture=_Replay, say=lambda line: None)
    assert "cuda_graphs" in fast.on
    graphed = vars(model.dit)["forward"]
    assert graphed.report()["shapes"] == 3                    # a graph for each batch size: 1, 2 and 4
    assert fast.report["runs"] == 8 + 2


def test_the_random_states_after_it_are_what_they_would_have_been():
    policy = _policy_of(_Flow().eval())                 # made first: the model seeds its own weights
    torch.manual_seed(3)
    np.random.seed(3)
    want = (torch.rand(3), np.random.rand(3))
    torch.manual_seed(3)
    np.random.seed(3)
    optimize(policy, _examples(), capture=_Replay, say=lambda line: None)
    got = (torch.rand(3), np.random.rand(3))
    assert torch.equal(got[0], want[0]) and np.array_equal(got[1], want[1])


class _Holder:
    def __init__(self) -> None:
        self.model = _Flow().eval()

    def act(self, observation):
        return self.model.sample(torch.as_tensor(observation["state"])[None])


GLOBAL_MODEL = _Flow().eval()


def _uses_the_global(observation):
    return GLOBAL_MODEL.sample(torch.as_tensor(observation["state"])[None])


def _with_model(model, observation):
    return model.sample(torch.as_tensor(observation["state"])[None])


def test_the_modules_are_found_wherever_the_policy_reaches_them():
    holder = _Holder()
    assert cuda_graphs.modules_of(_namespace(holder.act)) == {"self.model": holder.model}
    assert cuda_graphs.modules_of(_namespace(_uses_the_global)) == {"GLOBAL_MODEL": GLOBAL_MODEL}
    model = _Flow().eval()
    assert cuda_graphs.modules_of(_namespace(_policy_of(model))) == {"model": model}
    assert cuda_graphs.modules_of(_namespace(functools.partial(_with_model, model))) == {"arg0": model}
    # seq serve's call: a function holding the policy's infer, bound to the policy
    infer = holder.act
    assert cuda_graphs.modules_of(_namespace(lambda observation: infer(observation))) == {"infer.model": holder.model}


def test_differences_measure_how_far_two_outputs_are():
    a = {"actions": np.array([1.0, 2.0, 3.0]), "done": False}
    b = {"actions": np.array([1.0, 2.5, 3.0]), "done": False}
    found = differences(a, b)
    assert found["max_abs"] == 0.5 and abs(found["mean_abs"] - 0.5 / 3) < 1e-12 and 0.99 < found["cosine"] < 1.0
    assert differences(a, {"actions": np.zeros(2), "done": False}) is None          # their structures differ
    assert differences(a, a)["cosine"] == 1.0


# ── the switches that change the outputs ────────────────────────────────────────────────────────────────────────────
from sequence_ai.engine import compiled  # noqa: E402
from sequence_ai.engine.optimize import check_switches  # noqa: E402


def test_a_switch_that_changes_the_outputs_is_refused_without_a_tolerance_and_a_tolerance_is_read():
    for switches in ({"torch_compile": True}, {"fused_kernels": True}, {"tensorrt": True}, {"precision": "fp16"},
                     {"precision": "fp32"}):
        with pytest.raises(ValueError, match="change.* the policy's outputs: give tolerance="):
            check_switches(**switches)
    with pytest.raises(ValueError, match="precision='bf16'"):
        check_switches(precision="bf16")
    with pytest.raises(ValueError, match="with tensorrt, precision is"):
        check_switches(tensorrt=True, precision="fp8", tolerance=0.1)
    with pytest.raises(ValueError, match="its limits are max_abs, mean_abs and cosine, not atol"):
        check_switches(torch_compile=True, tolerance={"atol": 1e-3})
    with pytest.raises(ValueError, match="cosine=1.5"):
        check_switches(torch_compile=True, tolerance={"cosine": 1.5})
    assert check_switches(torch_compile=True, tolerance=1e-3)["tolerance"] == {"max_abs": 1e-3}
    assert check_switches(tolerance={"max_abs": 0.02, "cosine": 0.999})["tolerance"] == {"max_abs": 0.02,
                                                                                          "cosine": 0.999}
    with pytest.raises(ValueError, match="give tolerance="):
        optimize(lambda o: o, [{}], torch_compile=True)


def _compile_that(shift: float):
    """A stand-in for torch.compile: the module's own forward, without the time its kernels take to launch, its
    outputs moved by `shift` (a compiler's other order of the arithmetic)."""
    def torch_compile(module, **_):
        own = compiled.own_forward(module)

        def forward(*args, **kwargs):
            FAST.on = True
            try:
                return own(*args, **kwargs) + shift
            finally:
                FAST.on = False
        return forward
    return torch_compile


# While set, the heavy denoiser's arithmetic takes no time: a switch that computes less.
LIGHT = threading.local()


class _HeavyDenoiser(_Denoiser):
    """A denoiser whose kernels take time to run as well as to launch: a graph saves the launching (FAST), not the
    running."""

    def forward(self, x):
        if not getattr(LIGHT, "on", False):
            time.sleep(SLEEP_S)
        return super().forward(x)


class _HeavyFlow(_Flow):
    def __init__(self) -> None:
        super().__init__()
        self.dit = _HeavyDenoiser()


def _lighter_that(shift: float, launches: int):
    """A stand-in for torch.compile that computes less - the module's own forward without its arithmetic's time
    (LIGHT) - and launches `launches` waits' worth of kernels in place of its own, saved only in a graph (FAST); its
    outputs moved by `shift`."""
    def torch_compile(module, **_):
        own = compiled.own_forward(module)

        def forward(*args, **kwargs):
            for _ in range(launches):
                _slow()
            fast = getattr(FAST, "on", False)
            LIGHT.on = FAST.on = True                # its own launches are the `launches` above
            try:
                return own(*args, **kwargs) + shift
            finally:
                LIGHT.on, FAST.on = False, fast
        return forward
    return torch_compile


def test_a_switch_within_the_tolerance_stays_on_and_the_graphs_are_held_to_what_it_runs(monkeypatch):
    monkeypatch.setattr(compiled, "torch_compile", _lighter_that(1e-6, launches=1))
    model = _HeavyFlow().eval()
    fast = optimize(_policy_of(model), _examples(), torch_compile=True, tolerance=1e-3, capture=_Replay,
                    say=lambda line: None)
    switch = fast.report["switches"]["torch_compile"]
    assert switch["on"] and switch["modules"] == ["model.dit"] and 0 < switch["differences"]["max_abs"] <= 1e-3
    # the denoiser's graphs are of the compiled denoiser, checked bit for bit against it; the whole call's on top
    assert fast.on == ["cuda_graphs", "whole_graph", "torch_compile"]
    graphed = vars(model.dit)["forward"]
    assert isinstance(graphed, cuda_graphs.Graphed) and "_lighter_that" in graphed.inner.__qualname__


def test_a_switch_that_changes_the_outputs_is_timed_with_cuda_graphs_on_in_both_turns_as_it_will_run(monkeypatch):
    # launching more kernels and computing less: slower alone, faster under the graphs, which take its launches away
    monkeypatch.setattr(compiled, "torch_compile", _lighter_that(1e-6, launches=4))
    alone = optimize(_policy_of(_HeavyFlow().eval()), _examples(), torch_compile=True, tolerance=1e-3,
                     capture=_Replay, cuda_graphs=False, whole_graph=False, say=lambda line: None)
    switch = alone.report["switches"]["torch_compile"]
    assert alone.on == [] and switch["why"].startswith("no faster with it (act ") and "timed_with" not in switch
    said: list = []
    under = optimize(_policy_of(_HeavyFlow().eval()), _examples(), torch_compile=True, tolerance=1e-3,
                     capture=_Replay, whole_graph=False, say=said.append)
    switch = under.report["switches"]["torch_compile"]
    assert under.on == ["cuda_graphs", "torch_compile"] and switch["timed_with"] == "cuda_graphs"
    assert any(line.startswith("optimize: torch_compile on - ") and line.endswith(" ms with cuda_graphs on in both")
               and " runs (largest differences: max_abs " in line for line in said)     # what `seq serve` prints
    # eager mode's time is its own, timed in the same turns, and not the graphs' the switch was timed against
    assert under.report["act_ms"]["eager"] > switch["act_ms"]["before"]
    # saving only launches: faster alone, and no faster under the graphs, which save them anyway
    monkeypatch.setattr(compiled, "torch_compile", _compile_that(1e-6))
    model = _Flow().eval()
    graphed = optimize(_policy_of(model), _examples(), torch_compile=True, tolerance=1e-3, capture=_Replay,
                       say=lambda line: None)
    switch = graphed.report["switches"]["torch_compile"]
    assert not switch["on"] and switch["timed_with"] == "cuda_graphs" and "with cuda_graphs on in both" in switch["why"]
    assert graphed.on == ["cuda_graphs", "whole_graph"]
    assert vars(model.dit)["forward"].before is cuda_graphs._NONE          # graphed over the denoiser's own forward


def test_a_switch_kept_under_graphs_that_do_not_stay_on_has_the_act_timed_again_as_it_runs(monkeypatch):
    # computing less and launching nothing: as fast under the graphs as alone, so they add nothing once it is on
    monkeypatch.setattr(compiled, "torch_compile", _lighter_that(1e-6, launches=0))
    module = sys.modules["sequence_ai.engine.optimize"]
    turns, arms = module._Trial.turns, []
    monkeypatch.setattr(module._Trial, "turns", lambda self, given: arms.append(len(given)) or turns(self, given))
    fast = optimize(_policy_of(_HeavyFlow().eval()), _examples(), torch_compile=True, tolerance=1e-3, capture=_Replay,
                    whole_graph=False, say=lambda line: None)
    assert fast.on == ["torch_compile"] and fast.report["switches"]["torch_compile"]["timed_with"] == "cuda_graphs"
    assert "faster" in fast.report["switches"]["cuda_graphs"]["why"]
    assert arms == [3, 2, 1]        # the switch under its graphs, with eager mode; cuda_graphs; what runs now, alone
    assert fast.report["act_ms"]["eager"] > fast.report["act_ms"]["optimized"] > 0


def test_a_switch_past_the_tolerance_is_taken_out_and_the_module_runs_its_own_forward(monkeypatch):
    monkeypatch.setattr(compiled, "torch_compile", _compile_that(0.1))
    model = _Flow().eval()
    fast = optimize(_policy_of(model), _examples(), torch_compile=True, tolerance=1e-3, capture=_Replay,
                    cuda_graphs=False, whole_graph=False, say=lambda line: None)
    why = fast.report["switches"]["torch_compile"]["why"]
    assert fast.on == [] and why.startswith("its max_abs from eager mode's was ") and \
        why.endswith("on run 1 of 8, past the 0.001 of the tolerance")
    assert "forward" not in vars(model.dit)


def test_tensorrt_not_installed_is_said(monkeypatch):
    import importlib.util

    if importlib.util.find_spec("torch_tensorrt") is not None:
        pytest.skip("TensorRT is installed here")
    fast = optimize(_policy_of(_Flow().eval()), _examples(), tensorrt=True, tolerance=1e-3, cuda_graphs=False,
                    whole_graph=False, say=lambda line: None)
    assert "TensorRT is not installed: pip install sequence-ai[tensorrt]" in fast.report["switches"]["tensorrt"]["why"]


def test_fp8_on_a_machine_without_it_says_why():
    if torch.cuda.is_available():
        pytest.skip("this machine has a GPU")
    fast = optimize(_policy_of(_Flow().eval()), _examples(), precision="fp8", tolerance=0.5, cuda_graphs=False,
                    whole_graph=False, say=lambda line: None)
    assert fast.report["switches"]["precision"]["why"] == "fp8 needs a CUDA GPU"


class _RMSNorm(torch.nn.Module):
    """An RMSNorm as a language model's code writes it: float32 inside, its weight times the normalized row."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight, self.variance_epsilon = torch.nn.Parameter(torch.rand(width) + 0.5), 1e-6

    def forward(self, x):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        return self.weight * (x.float() * torch.rsqrt(variance + self.variance_epsilon)).to(x.dtype)


class _MLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = (torch.nn.Linear(16, 32), torch.nn.Linear(16, 32),
                                                        torch.nn.Linear(32, 16))
        self.act_fn = torch.nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class _Block(torch.nn.Module):
    """A transformer block's memory-bound steps around its projections: an RMSNorm, a gated MLP."""

    def __init__(self) -> None:
        super().__init__()
        self.norm, self.mlp = _RMSNorm(16), _MLP()

    def forward(self, x):
        _slow()
        return x + self.mlp(self.norm(x))


def _block_policy(block):
    def policy(observation):
        with torch.no_grad():
            x = torch.as_tensor(observation["state"]).repeat(4)[None]
            for _ in range(3):
                x = block(x)
            return {"actions": x[0]}
    return policy


def test_fused_kernels_found_in_the_modules_are_judged_and_taken_out_when_no_faster():
    torch.manual_seed(0)
    block = _Block().eval()
    fast = optimize(_block_policy(block), _examples(), fused_kernels=True, tolerance=1e-4, cuda_graphs=False,
                    whole_graph=False, say=lambda line: None)
    switch = fast.report["switches"]["fused_kernels"]
    # on the CPU the reference runs, as fast as the model's own code: found, within the tolerance, and no faster
    assert switch["replaced"] == ["block.norm: RMSNorm", "block.mlp: gated MLP (silu)"]
    assert not switch["on"] and "faster" in switch["why"] and switch["differences"]["max_abs"] <= 1e-4
    assert "forward" not in vars(block.norm) and "forward" not in vars(block.mlp)


def test_fp8_layers_are_calibrated_once_and_judged(monkeypatch, tmp_path):
    import sequence_ai.engine.fp8 as module

    monkeypatch.setattr(module, "unsupported", lambda device=None: None)     # its CPU reference stands in for the GPU
    calls = []
    real, install = module._measure, module.install
    monkeypatch.setattr(module, "_measure", lambda *a: calls.append(1) or real(*a))
    monkeypatch.setattr(module, "install", lambda m, **k: install(m, **{**k, "cache": tmp_path}))
    torch.manual_seed(0)
    block = _Block().eval()
    for _ in range(2):
        fast = optimize(_block_policy(block), _examples(), precision="fp8", tolerance=0.5, cuda_graphs=False,
                        whole_graph=False, say=lambda line: None)
        switch = fast.report["switches"]["precision"]
        assert switch["precision"] == "fp8" and 1 <= switch["layers"] <= 3
    assert calls == [1] and len(list(tmp_path.glob("*.json"))) == 1            # calibrated once, read the second time
    if not switch["on"]:
        assert all("forward" not in vars(layer) for layer in (block.mlp.gate_proj, block.mlp.up_proj,
                                                              block.mlp.down_proj))


def test_a_policy_that_takes_seed_is_given_each_runs_so_its_fresh_noise_repeats():
    """A template's infer draws fresh noise when it is given no seed, and noise from its seed when it is - as pi0.5's
    do: optimize() gives each run its seed, so eager mode repeats and the graphs can be checked."""
    model = _Flow().eval()

    def policy(observation, seed=None):
        rng = np.random.default_rng(seed)                         # fresh entropy with no seed
        noise = torch.as_tensor(rng.standard_normal((1, 8)), dtype=torch.float32)
        with torch.no_grad():
            c = model.encoder(torch.as_tensor(observation["state"])[None])
            x = noise
            for _ in range(4):
                x = model.dit(x + c)
        return {"actions": x[0]}

    fast = optimize(policy, _examples(), capture=_Replay, whole_graph=False, say=lambda line: None)
    assert fast.on == ["cuda_graphs"], fast.report["switches"]["cuda_graphs"]


def test_apply_puts_what_a_check_kept_on_the_policy_built_again_without_checking_it_again():
    fast = optimize(_policy_of(_Flow().eval()), _examples(), capture=_Replay, say=lambda line: None)
    report = json.loads(json.dumps(fast.report))                  # as another process gets it
    model = _Flow().eval()                                        # the policy built again, where it is served
    policy = _policy_of(model)
    want = _eager(policy, _examples(), [7, 8, 9, 10])
    said = []
    applied = apply(policy, report, capture=_Replay, say=said.append)
    assert applied.on == ["cuda_graphs", "whole_graph"] and applied.report["applied"]
    assert applied.report["switches"]["cuda_graphs"]["modules"] == ["model.dit"]
    assert applied.report["switches"]["whole_graph"]["call"] == "model.sample"
    assert isinstance(vars(model.dit)["forward"], cuda_graphs.Graphed)
    assert isinstance(vars(model)["sample"], cuda_graphs.Graphed)
    assert said == ["optimize: cuda_graphs, whole_graph on, as an earlier check kept them"]
    assert all(torch.equal(a, b) for a, b in zip(_eager(applied, _examples(), [7, 8, 9, 10]), want))


def test_apply_leaves_off_what_changes_the_outputs_and_what_a_machine_cannot_graph():
    report = {"switches": {"torch_compile": {"on": True, "modules": ["model.dit"]},
                           "cuda_graphs": {"on": True, "modules": ["model.dit"]},
                           "whole_graph": {"on": False, "why": "no faster with it"}}}
    model = _Flow().eval()
    applied = apply(_policy_of(model), report, say=lambda line: None)          # no capture stand-in, no CUDA GPU here
    whys = {name: applied.report["switches"][name]["why"] for name in ("torch_compile", "cuda_graphs", "whole_graph")}
    assert applied.on == [] and "forward" not in vars(model.dit)
    assert whys == {"torch_compile": "it changes the outputs: optimize() turns it on where it checks it",
                    "cuda_graphs": "this machine has no CUDA GPU", "whole_graph": "turned off"}
