# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
An openpi policy's infer() on a batch (examples/serve_openpi_batched.py): each observation passes through the input
transforms, they are stacked, the model samples them in one call with a key split from the policy's own, and each row
passes through the output transforms; one output per observation, the dict infer() gives it. A PyTorch model runs one
observation at a time, through infer(). A policy without infer()'s attributes is refused.

jax and openpi are stood in for by modules with the few calls the adapter makes, so the adapter's own steps are checked
here without either installed; a real policy's batch is checked against its infer() on a GPU.

    python -m pytest -q tests/engine/test_openpi.py
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

import numpy as np
import pytest

# the batch function of the example that shows how to write one for openpi's policy (it is not part of the package)
_EXAMPLE = pathlib.Path(__file__).resolve().parents[2] / "examples" / "serve_openpi_batched.py"
_SPEC = importlib.util.spec_from_file_location("serve_openpi_batched", _EXAMPLE)
openpi = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(openpi)


def _tree_map(f, tree, *rest):
    if isinstance(tree, dict):
        return {key: _tree_map(f, tree[key], *(other[key] for other in rest)) for key in tree}
    if isinstance(tree, (list, tuple)):
        return type(tree)(_tree_map(f, item, *(other[i] for other in rest)) for i, item in enumerate(tree))
    return f(tree, *rest)


@pytest.fixture
def stand_ins(monkeypatch):
    """jax, jax.numpy and openpi.models.model, as far as the adapter uses them; the keys the policy's rng was split
    into."""
    splits = []

    def split(key):
        splits.append(key)
        return key + 1, 1000 + key

    jnp = types.ModuleType("jax.numpy")
    jnp.asarray, jnp.stack = np.asarray, np.stack
    jax = types.ModuleType("jax")
    jax.numpy = jnp
    jax.tree = types.SimpleNamespace(map=_tree_map)
    jax.random = types.SimpleNamespace(split=split)
    model = types.ModuleType("openpi.models.model")
    model.Observation = types.SimpleNamespace(from_dict=lambda batch: dict(batch))
    models = types.ModuleType("openpi.models")
    models.model = model
    package = types.ModuleType("openpi")
    package.models = models
    for name, module in {"jax": jax, "jax.numpy": jnp, "openpi": package, "openpi.models": models,
                         "openpi.models.model": model}.items():
        monkeypatch.setitem(sys.modules, name, module)
    return splits


class Policy:
    """openpi's Policy as infer() reads it: input transforms that scale the state and add a token, a model whose actions
    are each row's state and its own noise, output transforms that keep 8 of the action's numbers."""

    def __init__(self):
        self._rng = 7
        self._sample_kwargs = {"num_steps": 10}
        self._input_transform = lambda x: {"state": np.asarray(x["state"]) * 2, "token": np.int32(3)}
        self._output_transform = lambda x: {"state": x["state"], "actions": x["actions"][:, :8]}
        self.calls = []

    def _sample_actions(self, rng, observation, **kwargs):
        self.calls.append({"rng": rng, "batch": len(observation["state"]), **kwargs})
        state = np.asarray(observation["state"])
        noise = np.arange(len(state))[:, None, None] * 0.5            # each row its own noise
        return np.broadcast_to(state.sum(axis=1)[:, None, None], (len(state), 10, 32)) + noise


def test_a_batch_samples_once_and_each_observation_gets_the_output_infer_would_give_it(stand_ins):
    policy = Policy()
    infer_batch = openpi.batched(policy)
    observations = [{"state": np.full(4, float(i))} for i in range(3)]
    outputs = infer_batch(observations)

    assert len(outputs) == 3 and len(policy.calls) == 1                          # one call to the model
    assert policy.calls[0] == {"rng": 1007, "batch": 3, "num_steps": 10}           # a key split from the policy's
    assert stand_ins == [7] and policy._rng == 8                                   # which moves on, as infer()'s does
    for i, output in enumerate(outputs):
        assert set(output) == {"state", "actions", "policy_timing"}
        assert np.array_equal(output["state"], np.full(4, 2.0 * i))               # its own row, transformed
        assert output["actions"].shape == (10, 8)                                  # through the output transform
        assert np.allclose(output["actions"], 8.0 * i + 0.5 * i)                  # its state and its own noise
        assert output["policy_timing"]["infer_ms"] >= 0
    assert outputs[0]["policy_timing"] == outputs[2]["policy_timing"]              # the batch's time, shared


def test_the_observations_are_copied_before_the_transforms(stand_ins):
    policy = Policy()

    def in_place(x):
        x["state"] = x["state"] + 1                       # a transform that writes into what it is given
        return {"state": x["state"], "token": np.int32(0)}

    policy._input_transform = in_place
    observation = {"state": np.zeros(2)}
    openpi.batched(policy)([observation])
    assert np.array_equal(observation["state"], np.zeros(2))


def test_a_pytorch_model_runs_one_observation_at_a_time_through_infer():
    class TorchPolicy:
        _is_pytorch_model = True

        def __init__(self):
            self.seen = []

        def infer(self, observation):
            self.seen.append(observation["i"])
            return {"actions": observation["i"]}

    policy = TorchPolicy()
    assert openpi.batched(policy)([{"i": 1}, {"i": 2}]) == [{"actions": 1}, {"actions": 2}]
    assert policy.seen == [1, 2]


def test_what_is_not_an_openpi_policy_is_refused_naming_what_it_lacks():
    class Other:
        def infer(self, observation):
            return observation

    with pytest.raises(TypeError, match="takes an openpi Policy .* this Other has no _input_transform, "
                                        "_output_transform, _sample_actions, _sample_kwargs, _rng"):
        openpi.batched(Other())


def test_served_with_batch_the_waiting_observations_reach_the_model_together(stand_ins):
    """The adapter as serve() calls it: the app's thread, its batches - the observations that waited while the policy
    ran reach it together, none held for others - and each robot its own row back."""
    pytest.importorskip("sequence_base")                 # serve() takes each robot's stream, which needs sequence-base
    import asyncio
    import json
    import threading

    from sequence_ai.engine.serve import App
    from sequence_ai.network import msgpack_io

    policy = Policy()
    gate, started, sample = threading.Event(), threading.Event(), policy._sample_actions

    def held_first(rng, observation, **kwargs):
        if not started.is_set():
            started.set()
            gate.wait(5)                            # the first act holds the policy while the others come
        return sample(rng, observation, **kwargs)

    policy._sample_actions = held_first

    async def go():
        app = App(openpi.batched(policy), batch=[1, 2, 4])
        batcher, reached = app._batcher, []

        class Reached:                              # the app's batcher, counting the calls that reached it
            async def __call__(self, *args, **kwargs):
                reached.append(1)
                return await batcher(*args, **kwargs)

            def __getattr__(self, name):
                return getattr(batcher, name)

        app._batcher = Reached()
        try:
            async def request(path, body, kind):
                sent: list = []
                parts = [body]

                async def receive():
                    return {"type": "http.request", "body": parts.pop(0) if parts else b"", "more_body": False}

                async def send(message):
                    sent.append(message)

                scope = {"type": "http", "method": "POST", "path": path,
                         "headers": [(b"content-length", str(len(body)).encode()), (b"content-type", kind),
                                     (b"accept", msgpack_io.MEDIA_TYPE.encode())]}
                await app(scope, receive, send)
                return sent[1]["body"]

            async def observe(i):                   # each robot's observation, on its own episode
                frame = {"seq": 0, "inputs": {"state": np.full(4, float(i))}}
                await request("/observe", msgpack_io.pack({"episode_id": f"robot{i}", "frames": [frame], "skipped": [],
                                                           "blocks": {}}), msgpack_io.MEDIA_TYPE.encode())

            async def act(i):                       # and its act after it
                call = json.dumps({"call": "act", "episode_id": f"robot{i}", "seq": 1}).encode()
                return msgpack_io.unpack(await request("/act", call, b"application/json"))["outputs"]

            await asyncio.gather(*(observe(i) for i in range(5)))
            first = asyncio.ensure_future(act(4))
            while not started.is_set():
                await asyncio.sleep(0.01)
            rest = [asyncio.ensure_future(act(i)) for i in range(4)]
            # the four wait for the policy, each in the batcher before it is let go: a loaded machine hands them over
            # slowly, and a fixed sleep let it go with three in (MARKET 11-20 saw them split 1, 2, 1)
            while len(reached) < 5:
                await asyncio.sleep(0.01)
            gate.set()
            return await asyncio.gather(*rest), await first
        finally:
            app.close()

    outputs, alone = asyncio.run(asyncio.wait_for(go(), 10))
    assert [c["batch"] for c in policy.calls] == [1, 4]      # the first alone; the four that waited for it, together
    assert [float(o["actions"][0, 0]) for o in outputs] == [8.0 * i + 0.5 * i for i in range(4)]
    assert float(alone["actions"][0, 0]) == 8.0 * 4
