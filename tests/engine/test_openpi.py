# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
An openpi policy's infer() on a batch (sequence_ai/engine/openpi.py): each observation passes through the input
transforms, they are stacked, the model samples them in one call with a key split from the policy's own, and each row
passes through the output transforms; one output per observation, the dict infer() gives it. A PyTorch model runs one
observation at a time, through infer(). A policy without infer()'s attributes is refused.

jax and openpi are stood in for by modules with the few calls the adapter makes, so the adapter's own steps are checked
here without either installed; a real policy's batch is checked against its infer() on a GPU.

    python -m pytest -q tests/engine/test_openpi.py
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from sequence_ai.engine import openpi


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
    """The adapter as serve() calls it: the app's thread, its batches, and each robot its own row back."""
    import asyncio

    from sequence_ai.engine.serve import App
    from sequence_ai.network import msgpack_io

    policy = Policy()

    async def go():
        app = App(openpi.batched(policy), batch=[1, 2, 4])
        try:
            async def act(i):
                sent: list = []
                body = msgpack_io.pack({"state": np.full(4, float(i))})
                parts = [body]

                async def receive():
                    return {"type": "http.request", "body": parts.pop(0), "more_body": False}

                async def send(message):
                    sent.append(message)

                scope = {"type": "http", "method": "POST", "path": "/act",
                         "headers": [(b"content-length", str(len(body)).encode())]}
                await app(scope, receive, send)
                return msgpack_io.unpack(sent[1]["body"])

            return await asyncio.gather(*(act(i) for i in range(4)))
        finally:
            app.close()

    outputs = asyncio.run(asyncio.wait_for(go(), 10))
    assert [c["batch"] for c in policy.calls] == [4]
    assert [float(o["actions"][0, 0]) for o in outputs] == [8.0 * i + 0.5 * i for i in range(4)]
