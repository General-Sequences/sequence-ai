# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
A torch model's repeated module replayed from graphs of its own kernels (cuda_graphs.py). Each test pins one
guarantee, with the graph itself replaced by a fake (`capture`), so they run with no GPU:
  a shape's first call runs in eager mode, its graph is kept only when its output for that call is the same, byte for
  byte, and it is replayed for that shape afterwards;
  a shape whose graph's output differs, or whose capture fails, runs in eager mode permanently, reported once, and the
  random state is as it would be had nothing been captured;
  each shape is captured separately; a call with an argument no graph can be keyed on runs in eager mode;
  the census finds the outermost module called more than once per run with one shape, never one called once or with
  shapes that change;
  a module is graphed in place: the same object of the same class, its state_dict, parameters and other methods its
  own, its hooks run on each call; a forward that calls itself runs those calls within its own;

    python -m pytest -q tests/engine/test_cuda_graphs.py
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from sequence_ai.engine import cuda_graphs  # noqa: E402


class _Fake:
    """A graph whose output is `answers` (eager mode's own output by default), and which counts its replays."""

    made: list = []

    def __init__(self, inner, args, kwargs, pool, *, answer=None, fail=None) -> None:
        if fail is not None:
            torch.rand(4)                                 # a capture that draws from the generator before it fails
            raise fail
        self.inner, self.answer, self.replays = inner, answer, 0
        _Fake.made.append(self)

    def replay(self, args, kwargs):
        self.replays += 1
        return self.answer if self.answer is not None else self.inner(*args, **kwargs)


class _Double(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, x, *, scale=1.0):
        self.calls += 1
        return x * 2 * scale


class _Same:
    """A value no graph can be keyed on, which a module can still use."""

    def __rmul__(self, other):
        return other


def _graphed(capture, said):
    return cuda_graphs.Graphed(_Double(), "head", say=said.append, capture=capture)


def test_a_shapes_first_call_runs_in_eager_mode_and_its_kept_graph_serves_later_calls():
    said: list = []
    _Fake.made = []
    g = _graphed(lambda inner, a, k, pool: _Fake(inner, a, k, pool), said)
    x = torch.ones(3)
    assert torch.equal(g(x), x * 2) and g.inner.calls == 2                # eager run, then the graph's check of it
    assert torch.equal(g(x + 1), (x + 1) * 2)
    assert len(_Fake.made) == 1 and _Fake.made[0].replays == 2 and g.inner.calls == 3
    assert said == ["cuda graphs: captured head for a new shape in 0.0 s"]


def test_a_graph_whose_output_differs_leaves_its_shape_in_eager_mode_reported_once():
    said: list = []
    g = _graphed(lambda inner, a, k, pool: _Fake(inner, a, k, pool, answer=torch.zeros(3)), said)
    x = torch.ones(3)
    for _ in range(3):
        assert torch.equal(g(x), x * 2)
    assert said == ["cuda graphs: head runs in eager mode for one of its shapes: "
                    "its graph's output differed from eager mode's"]
    assert len(g.eager) == 1 and not g.graphs


def test_a_capture_that_fails_leaves_its_shape_in_eager_mode_and_the_random_state_unchanged():
    said: list = []
    g = _graphed(lambda inner, a, k, pool: _Fake(inner, a, k, pool, fail=RuntimeError("operation not permitted")), said)
    torch.manual_seed(7)
    want = (torch.rand(2), torch.rand(2))
    torch.manual_seed(7)
    got_first = torch.rand(2)
    assert torch.equal(g(torch.ones(3)), torch.full((3,), 2.0))
    assert torch.equal(got_first, want[0]) and torch.equal(torch.rand(2), want[1])   # its draws undone
    assert said == ["cuda graphs: head runs in eager mode for one of its shapes: its capture failed "
                    "(RuntimeError: operation not permitted)"]
    g(torch.ones(3))
    assert len(said) == 1                                                    # reported once


def test_each_shape_is_captured_separately_and_an_argument_no_graph_is_keyed_on_runs_in_eager_mode():
    said: list = []
    _Fake.made = []
    g = _graphed(lambda inner, a, k, pool: _Fake(inner, a, k, pool), said)
    g(torch.ones(3))
    g(torch.ones(4))
    g(torch.ones(3), scale=2.0)                                              # a different value: a different shape
    assert len(_Fake.made) == 3
    calls = g.inner.calls
    assert torch.equal(g(torch.ones(3), scale=_Same()), torch.full((3,), 2.0))
    assert g.inner.calls == calls + 1 and len(_Fake.made) == 3               # eager mode, nothing captured


def test_outputs_are_the_same_only_byte_for_byte():
    nan = torch.tensor([float("nan"), 1.0])
    assert cuda_graphs.same(nan, nan.clone())
    assert not cuda_graphs.same(torch.tensor([0.0]), torch.tensor([-0.0]))
    assert not cuda_graphs.same(torch.ones(2), torch.ones(2, dtype=torch.float64))
    assert cuda_graphs.same({"a": (torch.ones(1), 3)}, {"a": (torch.ones(1), 3)})


class _Head(torch.nn.Module):
    """A denoising head: its step called `steps` times per run with one shape, its encoder once."""

    def __init__(self, steps: int = 4) -> None:
        super().__init__()
        self.encoder, self.step, self.steps = torch.nn.Linear(8, 8), torch.nn.Sequential(torch.nn.Linear(8, 8)), steps

    def forward(self, x):
        x = self.encoder(x)
        for _ in range(self.steps):
            x = x + self.step(x)
        return x


class _Changing(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inner = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.inner(x) + self.inner(torch.cat([x, x]))[:1]


def test_the_census_finds_the_outermost_module_called_again_with_one_shape():
    head, changing = _Head(), _Changing()
    census = cuda_graphs.Census({"head": head, "changing": changing})
    for _ in range(3):
        head(torch.ones(1, 8))
        changing(torch.ones(1, 4))
        census.end_act()
    census.close()
    found = [c["path"] for c in census.candidates(min_ms=0.0)]
    assert found == ["head.step"]                     # not head.step.0 inside it, nor the encoder, nor changing.inner
    assert census.candidates(min_ms=0.0)[0]["calls"] == 4
    assert not head._forward_pre_hooks and not head.step._forward_hooks       # its hooks removed


class _Policy:
    """A policy that holds its model inside its own object, as GR00T's template holds Isaac-GR00T's policy."""

    def __init__(self) -> None:
        self._policy = type("Inner", (), {})()
        self._policy.model = _Head()
        self.name = "toy"

    def act(self, x):
        return self._policy.model(x)


def _fake(inner, args, kwargs, pool):
    return _Fake(inner, args, kwargs, pool)


def test_a_module_is_graphed_where_the_policy_holds_it_the_same_object_of_its_own_class():
    """[fails before: the module was replaced by a wrapper of another class: `type(model.step)` was GraphedModule,
    its state_dict's keys under `inner.`]"""
    policy, said, fired = _Policy(), [], []
    assert list(cuda_graphs.modules_of(policy)) == ["_policy.model"]
    model = policy._policy.model
    step, encoder = model.step, model.encoder
    keys, params = list(model.state_dict()), list(model.parameters())
    step.register_forward_hook(lambda module, args, out: fired.append(module))
    want = policy.act(torch.ones(1, 8))
    _Fake.made = []
    got = cuda_graphs.install(policy, ["_policy.model.step", "_policy.model.encoder", "_policy.model.nothing"],
                              say=said.append, capture=_fake)
    assert list(got) == ["_policy.model.step", "_policy.model.encoder"]
    assert said == ["cuda graphs: _policy.model.nothing is not in this model: not captured"]
    assert model.step is step and type(model.step) is torch.nn.Sequential and model.encoder is encoder
    assert list(model.state_dict()) == keys and [id(p) for p in model.parameters()] == [id(p) for p in params]
    fired.clear()
    assert torch.equal(policy.act(torch.ones(1, 8)), want) and len(_Fake.made) == 2   # one shape each through the call
    assert fired == [step] * 4                          # its hook around each of its 4 calls, the replayed ones too
    assert cuda_graphs.install(policy, ["_policy.model.step"], capture=_fake)["_policy.model.step"] \
        is got["_policy.model.step"]                    # named again: graphed once


class _Prepared(torch.nn.Module):
    """A module whose policy also calls one of its own methods on it, besides its forward."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(8, 8)

    def forward(self, x):
        return self.lin(x)

    def forward_prepared(self, x):
        return self.lin(x) * 2


def test_a_method_the_policy_calls_on_the_module_is_its_own():
    holder = SimpleNamespace(dit=_Prepared())
    _Fake.made = []
    cuda_graphs.install(holder, ["dit"], capture=_fake)
    x = torch.ones(1, 8)
    assert torch.equal(holder.dit.forward_prepared(x), holder.dit.lin(x) * 2) and not _Fake.made
    assert torch.equal(holder.dit(x), holder.dit.lin(x)) and len(_Fake.made) == 1        # its calls alone graphed


class _Halves(torch.nn.Module):
    """A module whose forward calls its own forward again, on each half of its input."""

    def forward(self, x):
        if x.shape[0] > 1:
            return torch.cat([self.forward(x[:1]), self.forward(x[1:])])
        return x * 3


def test_a_forward_that_calls_itself_runs_those_calls_within_its_own():
    holder = SimpleNamespace(net=_Halves())
    _Fake.made = []
    got = cuda_graphs.install(holder, ["net"], capture=_fake)
    done: list = []
    called = threading.Thread(target=lambda: done.append(holder.net(torch.ones(2, 4))), daemon=True)
    called.start()
    called.join(10)
    assert done and torch.equal(done[0], torch.full((2, 4), 3.0))        # never waiting on itself
    assert len(_Fake.made) == 1 and got["net"].report()["shapes"] == 1     # the call it received, its halves within


class _Net(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(8, 8)

    def forward(self, x):
        return x + self.lin(x)


class _Checking:
    """A policy that checks what it holds before it runs, as a flow policy's reference path does (FLUX's
    `assert isinstance(self.dit, JointSingleSeq), "call prepare_inference before prediction"`, and its type itself),
    then draws its noise from the random state passed to its call and calls its net 4 times on it."""

    def __init__(self) -> None:
        self.net = _Net()

    def act(self):
        assert isinstance(self.net, _Net) and type(self.net) is _Net, "call prepare_inference before prediction"
        y = torch.randn(1, 8)
        for _ in range(4):
            y = self.net(y)
        return y


class _Denoiser(torch.nn.Module):
    """A small flow model's denoiser: 12 blocks of a norm, a 256-wide linear layer and a GELU, which is many small
    kernels per step that a graph replays without launching each."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            torch.nn.Sequential(torch.nn.LayerNorm(256), torch.nn.Linear(256, 256), torch.nn.GELU()) for _ in range(12))

    def forward(self, x):
        for block in self.blocks:
            x = x + 0.1 * block(x)
        return x


class _Flow:
    """A policy on the GPU that checks what it holds before it runs, as FLUX's reference path does, then denoises the
    noise its call draws in 10 steps."""

    def __init__(self) -> None:
        self.dit = _Denoiser().cuda().eval()

    def act(self):
        assert isinstance(self.dit, _Denoiser), "call prepare_inference before prediction"
        with torch.no_grad():
            x = torch.randn(1, 256, device="cuda")
            for _ in range(10):
                x = self.dit(x)
            return x.cpu()


class _Strict(_Checking):
    """A policy whose own code rejects a net whose forward is not its class's, as code that inspects or swaps its
    modules' forward may."""

    def act(self):
        if "forward" in vars(self.net):
            raise RuntimeError("this net's forward is not its own")
        return super().act()


def test_a_model_that_is_itself_a_module_is_counted_by_its_own_paths():
    """modules_of(a module) is {"": it}; its census names each module by its path from it, which install takes."""
    model = torch.nn.Sequential(_Double(), _Double())
    assert list(cuda_graphs.modules_of(model)) == [""]
    census = cuda_graphs.Census(cuda_graphs.modules_of(model))
    for _ in range(2):
        model(torch.ones(2))
        model[0](torch.ones(2))                                       # module "0" twice per run, "1" once
        census.end_act()
    census.close()
    assert [row["path"] for row in census.candidates(min_ms=0.0)] == ["0"]
    assert list(cuda_graphs.install(model, ["0"], say=lambda line: None, capture=_Fake)) == ["0"]


def test_with_no_gpu_a_model_is_left_unchanged_and_that_is_reported():
    if torch.cuda.is_available():
        pytest.skip("this machine has a GPU")
    model, said = torch.nn.Sequential(_Double()), []
    assert cuda_graphs.install(model, ["0"], say=said.append) == {}
    assert "forward" not in vars(model[0]) and said == ["cuda graphs: no CUDA GPU available: nothing captured"]


def test_with_no_gpu_a_mistyped_path_is_still_reported():
    import torch

    if torch.cuda.is_available():
        pytest.skip("this machine has a GPU: the path is reported as it is graphed")
    said = []
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    assert cuda_graphs.install(model, ["0", "action_head.denoiser"], say=said.append) == {}
    assert said == ["cuda graphs: action_head.denoiser is not in this model: not captured",
                    "cuda graphs: no CUDA GPU available: nothing captured"]
    assert "forward" not in vars(model[0])


def test_uninstall_puts_back_a_modules_forward_and_a_method_graphed_by_its_path():
    holder = SimpleNamespace(dit=_Prepared())
    got = cuda_graphs.install(holder, ["dit", "dit.forward_prepared"], capture=_fake)
    assert list(got) == ["dit", "dit.forward_prepared"]
    assert isinstance(vars(holder.dit)["forward_prepared"], cuda_graphs.Graphed)
    x = torch.ones(1, 8)
    assert torch.equal(holder.dit.forward_prepared(x), holder.dit.lin(x) * 2)
    assert cuda_graphs.uninstall(holder, ["dit", "dit.forward_prepared", "dit.nothing"]) == \
        ["dit", "dit.forward_prepared"]
    assert "forward" not in vars(holder.dit) and "forward_prepared" not in vars(holder.dit)
    assert torch.equal(holder.dit(x), holder.dit.lin(x))


def test_a_path_that_names_neither_a_module_nor_a_method_of_one_is_reported():
    said: list = []
    holder = SimpleNamespace(dit=_Prepared(), name="toy")
    assert cuda_graphs.install(holder, ["name"], say=said.append, capture=_fake) == {}
    assert said == ["cuda graphs: name is not a torch module or a method of one: not captured"]


class _Sampler(torch.nn.Module):
    """A flow model whose run is a method: its noise drawn in it, its denoiser called 3 times."""

    def __init__(self) -> None:
        super().__init__()
        self.dit = _Net()

    def sample(self, c):
        x = torch.randn(1, 8)
        for _ in range(3):
            x = self.dit(x + c)
        return x


class _Cached(_Fake):
    """A graph that returns what it computed when it was made, drawing nothing when replayed."""

    def __init__(self, inner, args, kwargs, pool) -> None:
        super().__init__(inner, args, kwargs, pool)
        self.out = inner(*args, **kwargs)

    def replay(self, args, kwargs):
        self.replays += 1
        return self.out.clone()


def test_a_whole_call_is_kept_when_its_replay_draws_what_eager_mode_drew_and_never_otherwise():
    model, said = _Sampler(), []
    graphed = cuda_graphs.install(model, ["sample"], say=said.append, capture=_fake, whole=True)["sample"]
    assert graphed.bound
    c = torch.ones(1, 8)
    torch.manual_seed(5)
    eager = _Sampler.sample(model, c)                        # the class's own method: eager mode
    torch.manual_seed(5)
    assert torch.equal(model.sample(c), eager) and graphed.report()["shapes"] == 1
    torch.manual_seed(5)
    assert torch.equal(model.sample(c), eager)               # replayed, from the same random state: the same noise
    stale = _Sampler()
    stale.dit = model.dit
    graphed = cuda_graphs.install(stale, ["sample"], say=said.append, capture=_Cached, whole=True)["sample"]
    torch.manual_seed(5)
    stale.sample(c)
    assert graphed.report()["shapes"] == 0 and graphed.report()["eager"] == [["differs", ""]]


def test_a_module_called_inside_another_graphs_capture_runs_in_eager_mode():
    model = _Sampler()
    _Fake.made = []
    inner = cuda_graphs.install(model, ["dit"], capture=_fake)["dit"]
    outer = cuda_graphs.install(model, ["sample"], capture=_fake, whole=True)["sample"]
    model.sample(torch.ones(1, 8))
    assert outer.report()["shapes"] == 1 and inner.report()["shapes"] == 0     # inside the whole call's capture
    assert len(_Fake.made) == 2                                  # the whole call's two captures, and nothing else
    cuda_graphs.uninstall(model, ["sample"])
    model.sample(torch.ones(1, 8))
    assert inner.report()["shapes"] == 1                       # alone again, the module captured on its own


class _Features(dict):
    """A model's feature dict: a dict of another class, its fields read as attributes too."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


class _Reads(torch.nn.Module):
    def forward(self, features):
        assert isinstance(features, _Features), "features must be the model's own class"
        return {"out": features.x * 2}


def test_a_mapping_of_another_class_is_keyed_on_and_given_back_as_its_own_class():
    holder = SimpleNamespace(net=_Reads())
    _Fake.made = []
    got = cuda_graphs.install(holder, ["net"], capture=_fake)["net"]
    out = holder.net(_Features(x=torch.ones(2)))
    assert torch.equal(out["out"], torch.full((2,), 2.0)) and got.report()["shapes"] == 1 and got.unkeyed == 0
    assert isinstance(cuda_graphs._static(_Features(x=torch.ones(2))), _Features)


def test_calls_finds_the_outermost_method_first_then_the_calls_within_it():
    model = _Sampler()
    calls = cuda_graphs.Calls(cuda_graphs.modules_of({"model": model}))
    for _ in range(2):
        with calls.run():
            model.sample(torch.ones(1, 8))
    found = [row["path"] for row in calls.outermost()]
    assert found == ["model.sample", "model.dit"]
    assert calls.outermost()[1]["calls"] == 3


def test_the_census_takes_every_candidate_when_asked_for_all():
    heads = torch.nn.ModuleDict({f"h{i}": _Head() for i in range(6)})
    census = cuda_graphs.Census({"heads": heads})
    for _ in range(2):
        for head in heads.values():
            head(torch.ones(1, 8))
        census.end_act()
    census.close()
    assert len(census.candidates(min_ms=0.0)) == 4 and len(census.candidates(min_ms=0.0, most=None)) == 6
