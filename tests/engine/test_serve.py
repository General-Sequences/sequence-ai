# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
serve() and its app (sequence_ai/engine/serve.py), reached as a robot reaches it: an observation posted as it is made
(POST /observe) and the act after it (POST /act), answered with the policy's output and the server's time, in msgpack;
the policy on one thread of its own, one call at a time, or on a batch of the observations waiting; /ping answered
while the policy runs, and 503 until the warm-up on the example is done; what every response says the server is; the
token; a body over the limit or one that does not read, a policy that raises or returns what cannot be sent; torch and
JAX outputs as numpy; the listening sockets; and the server on a real port. serve() needs sequence-base (Python 3.10 or
later): skipped where it is not installed.

    python -m pytest -q tests/engine/test_serve.py
"""
from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import os
import socket
import threading
import time

import numpy as np
import pytest

from sequence_ai.engine import serve as serving
from sequence_ai.engine.serve import App
from sequence_ai.network import msgpack_io

pytest.importorskip("sequence_base")         # serve() takes each robot's stream, which needs sequence-base

# what a robot's requests carry: its observations in msgpack, its act in JSON, the act's answer asked for in msgpack
_OBSERVE = [("Content-Type", msgpack_io.MEDIA_TYPE)]
_ACT = [("Content-Type", "application/json"), ("Accept", msgpack_io.MEDIA_TYPE)]
_EPISODES = itertools.count()                # each _act() an episode of its own


def _run(coroutine, timeout: float = 10.0):
    return asyncio.run(asyncio.wait_for(coroutine, timeout))


async def _request(app: App, method: str, path: str, body: bytes = b"", *, headers=(), chunks=None,
                   length: bool = True):
    """One request through the ASGI app: (status, headers, body)."""
    parts = list(chunks) if chunks is not None else [body]
    sent: list = []

    async def receive():
        if parts:
            return {"type": "http.request", "body": parts.pop(0), "more_body": bool(parts)}
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    given = [(k.lower().encode(), v.encode()) for k, v in headers]
    if length and chunks is None:
        given.append((b"content-length", str(len(body)).encode()))
    await app({"type": "http", "method": method, "path": path, "headers": given}, receive, send)
    start, *rest = sent
    return (start["status"], {k.decode(): v.decode() for k, v in start["headers"]},
            b"".join(m.get("body", b"") for m in rest))


def _observed(episode: str, observation) -> bytes:
    """A POST /observe body as a robot sends it: `observation`, the first of `episode`, in msgpack."""
    return msgpack_io.pack({"episode_id": episode, "frames": [{"seq": 0, "inputs": observation}], "skipped": []})


def _act_call(episode: str) -> bytes:
    """The POST /act body of the act after that observation: one observation of `episode` came before it."""
    return json.dumps({"call": "act", "episode_id": episode, "seq": 1}).encode()


async def _act(app: App, observation, headers=()):
    """`observation` as a robot sends it - posted as the first of an episode of its own, then the act after it, each
    request with `headers` - and the act's (status, headers, body), a 200's body the policy's output alone, in msgpack;
    the post's answer where the post is refused."""
    episode = f"e-{next(_EPISODES)}"
    posted = await _request(app, "POST", "/observe", _observed(episode, observation), headers=[*_OBSERVE, *headers])
    if posted[0] != 200:
        return posted
    status, said, body = await _request(app, "POST", "/act", _act_call(episode), headers=[*_ACT, *headers])
    if status == 200:
        body = msgpack_io.pack(msgpack_io.unpack(body)["outputs"])
    return status, said, body


def _picture() -> np.ndarray:
    """A 64 x 48 camera frame, with detail in it."""
    y, x = np.mgrid[0:48, 0:64]
    return np.stack([(x * 4) % 256, (y * 5) % 256, (x + y) * 2 % 256], axis=-1).astype(np.uint8)


# ── an observation in, the output out ──────────────────────────────────────────────────────────────────────────────
def test_an_observation_posted_and_the_act_after_it_get_the_policys_output_and_the_servers_time():
    def policy(observation):
        return {"actions": np.full((8, 7), observation["state"].sum(), dtype=np.float32), "note": "ok"}

    observation = {"state": np.arange(4, dtype=np.float32)}

    async def go():
        app = App(policy)
        try:
            posted = await _request(app, "POST", "/observe", _observed("e-first", observation), headers=_OBSERVE)
            return posted, await _request(app, "POST", "/act", _act_call("e-first"), headers=_ACT)
        finally:
            app.close()

    (posted, posted_headers, through), (status, headers, body) = _run(go())
    assert posted == 200 and posted_headers["content-type"] == "application/json"
    assert json.loads(through) == {"through": 0}               # the episode is whole through its first observation
    assert status == 200 and headers["content-type"] == msgpack_io.MEDIA_TYPE
    answer = msgpack_io.unpack(body)
    out = answer["outputs"]
    assert out["note"] == "ok" and out["actions"].shape == (8, 7) and float(out["actions"][0, 0]) == 6.0
    # the server's time, in the answer: from the act's last byte to its answer, which the robot takes off its round
    # trip, and the policy call's, within it
    perf = answer["perf"]
    assert sorted(perf) == ["handler_ms", "inference_ms"] and 0 <= perf["inference_ms"] <= perf["handler_ms"]


def test_the_policys_arrays_are_its_own_to_write_into():
    """msgpack reads arrays in place from the request's bytes, read-only: torch.from_numpy warns on each, and a policy
    that writes into its observation would fail."""
    def policy(observation):
        observation["state"][0] = 9.0
        observation["camera"][0][0, 0, 0] = 1
        return {"state": observation["state"], "writable": observation["camera"][0].flags.writeable}

    async def go():
        app = App(policy)
        try:
            return await _act(app, {"state": np.zeros(3, dtype=np.float32), "camera": [_picture()]})
        finally:
            app.close()

    status, _, body = _run(go())
    assert status == 200, body
    out = msgpack_io.unpack(body)
    assert out["state"].tolist() == [9.0, 0.0, 0.0] and out["writable"] is True


# ── the policy's thread ─────────────────────────────────────────────────────────────────────────────────────────────
def test_the_policy_runs_one_call_at_a_time_on_one_thread_of_its_own():
    running, most, names, lock = [0], [0], set(), threading.Lock()

    def policy(observation):
        with lock:
            running[0] += 1
            most[0] = max(most[0], running[0])
            names.add(threading.current_thread().name)
        time.sleep(0.02)
        with lock:
            running[0] -= 1
        return observation["i"]

    async def go():
        app = App(policy)
        try:
            answers = await asyncio.gather(*(_act(app, {"i": i}) for i in range(6)))
            return [msgpack_io.unpack(body) for _, _, body in answers], threading.current_thread().name
        finally:
            app.close()

    outputs, loop_thread = _run(go())
    assert outputs == list(range(6))                       # each request its own output
    assert most[0] == 1                                    # never two calls at once
    (name,) = names                                        # always the same thread, and not the event loop's
    assert name.startswith("sequence_ai-policy") and name != loop_thread


def test_a_ping_is_answered_while_the_policy_runs():
    started, gate = threading.Event(), threading.Event()

    def policy(observation):
        started.set()
        gate.wait(5)
        return 1

    async def go():
        app = App(policy)
        try:
            acting = asyncio.ensure_future(_act(app, {"x": 1}))
            while not started.is_set():                    # the act is in the policy's thread, which waits on the gate
                await asyncio.sleep(0.005)
            began = time.perf_counter()
            pinged = await _request(app, "GET", "/ping")
            took = time.perf_counter() - began
            assert not acting.done()
            gate.set()
            return pinged, took, await acting
        finally:
            gate.set()
            app.close()

    (status, _, body), took, (act_status, _, _) = _run(go())
    assert status == 200 and json.loads(body)["server"] == "sequence-ai" and took < 0.05
    assert act_status == 200


def test_every_response_says_it_is_serve_and_the_ping_says_more():
    def my_policy(observation):
        return 1

    async def go():
        app = App(my_policy, token="t")
        try:
            refused = await _request(app, "GET", "/ping")
            pinged = await _request(app, "GET", "/ping", headers=[("Authorization", "Bearer t")])
            acted = await _act(app, {}, headers=[("Authorization", "Bearer t")])
            missing = await _request(app, "GET", "/nothing", headers=[("Authorization", "Bearer t")])
            return refused, pinged, acted, missing
        finally:
            app.close()

    answers = _run(go())
    version = serving._version()
    for _, headers, _ in answers:
        assert headers["x-seq-server"] == f"sequence-ai/{version}"
        # and how a robot sends to it: on its stream, the policy reading each act's newest observation
        assert headers["x-seq-act-formats"] == f"{serving.STREAM_FORMAT}, {serving.LATEST_FORMAT}"
    (_, _, refused_body), (status, headers, body) = answers[0], answers[1]
    said = json.loads(body)
    assert status == 200 and headers["content-type"] == "application/json"
    assert said["server"] == "sequence-ai" and said["version"] == version and said["pid"] == os.getpid()
    assert said["policy"].endswith("my_policy") and 0 <= said["up_s"] < 60
    assert b"sequence-ai" not in refused_body                  # without the token, nothing but the refusal


def test_outputs_of_other_libraries_come_back_as_numpy_arrays():
    class Lazy:                                            # an array of another library, as JAX's reads into numpy
        def __array__(self, dtype=None, copy=None):
            return np.arange(3, dtype=np.float32)

    def policy(observation):
        return {"jax": Lazy(), "nested": [Lazy(), (Lazy(),)], "scalar": np.float32(2.5), "text": "t"}

    async def go():
        app = App(policy)
        try:
            return await _act(app, {})
        finally:
            app.close()

    status, _, body = _run(go())
    out = msgpack_io.unpack(body)
    assert status == 200
    assert np.array_equal(out["jax"], [0, 1, 2]) and np.array_equal(out["nested"][1][0], [0, 1, 2])
    assert float(out["scalar"]) == 2.5 and out["text"] == "t"


def test_torch_tensors_come_back_as_numpy_arrays_bfloat16_as_float32():
    torch = pytest.importorskip("torch")

    def policy(observation):
        return {"actions": torch.ones(2, 3), "half": torch.ones(2, dtype=torch.bfloat16),
                "list": [torch.zeros(1, dtype=torch.int64)]}

    async def go():
        app = App(policy)
        try:
            return await _act(app, {})
        finally:
            app.close()

    status, _, body = _run(go())
    out = msgpack_io.unpack(body)
    assert status == 200
    assert out["actions"].dtype == np.float32 and out["actions"].shape == (2, 3)
    assert out["half"].dtype == np.float32 and out["half"].tolist() == [1.0, 1.0]
    assert out["list"][0].dtype == np.int64


def test_jax_bfloat16_comes_back_as_float32():
    ml_dtypes = pytest.importorskip("ml_dtypes")

    async def go():
        app = App(lambda o: np.ones(3, dtype=ml_dtypes.bfloat16))
        try:
            return await _act(app, {})
        finally:
            app.close()

    status, _, body = _run(go())
    assert status == 200 and msgpack_io.unpack(body).dtype == np.float32


# ── batches ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_with_batch_the_observations_waiting_reach_the_policy_as_one_list_and_each_gets_its_own_output():
    gate, seen = threading.Event(), []

    def policy_on_a_list(observations):
        seen.append([o["i"] for o in observations])
        gate.wait(5)
        return [{"i": o["i"], "twice": o["i"] * 2} for o in observations]

    async def go():
        app = App(policy_on_a_list, batch=[1, 2, 4])
        try:
            first = asyncio.ensure_future(_act(app, {"i": 0}))
            while not seen:                                # the first observation is in the policy, alone
                await asyncio.sleep(0.005)
            rest = [asyncio.ensure_future(_act(app, {"i": i})) for i in range(1, 6)]
            while len(app._batcher._open) < 5:             # five more wait while the policy runs
                await asyncio.sleep(0.005)
            gate.set()
            answers = await asyncio.gather(first, *rest)
            return [msgpack_io.unpack(body) for _, _, body in answers], dict(app._batcher.sent)
        finally:
            gate.set()
            app.close()

    outputs, sent = _run(go())
    assert [o["twice"] for o in outputs] == [0, 2, 4, 6, 8, 10]
    assert seen == [[0], [1, 2, 3, 4], [5]]                # the largest listed size the waiting fill, then the rest
    assert sent == {1: 2, 4: 1}


@pytest.mark.parametrize("returned, said", [
    (lambda os: {"actions": np.zeros((len(os), 3))}, "returned a dict for a batch of 1: it returns a list"),
    (lambda os: [], "returned 0 outputs for a batch of 1: one output for each observation"),
])
def test_a_batch_function_that_returns_other_than_one_output_for_each_observation_is_told_so(returned, said):
    sizes, gate = [], threading.Event()

    def policy_on_a_list(observations):
        sizes.append(len(observations))
        if len(sizes) == 1:
            gate.wait(5)                        # the first act holds the policy while two more wait: a batch of 2
        return returned(observations)

    async def go():
        app = App(policy_on_a_list, batch=[1, 2])
        try:
            first = asyncio.ensure_future(_act(app, {"i": 0}))
            for _ in range(500):
                if sizes:
                    break
                await asyncio.sleep(0.01)
            rest = [asyncio.ensure_future(_act(app, {"i": i})) for i in (1, 2)]
            await asyncio.sleep(0.2)
            gate.set()
            return await asyncio.gather(first, *rest)
        finally:
            app.close()

    answers = _run(go())
    # the first alone; the batch of the 2 that waited for it failed, so each was run again alone, and failed alone, saying
    # why
    assert sizes == [1, 2, 1, 1]
    for status, _, body in answers:
        assert status == 500 and said in json.loads(body)["detail"]


# ── warm-up ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_the_warm_up_runs_the_example_twice_at_each_batch_size_and_ping_says_503_until_it_is_done():
    gate, seen = threading.Event(), []
    example = {"image": _picture(), "state": np.zeros(7, dtype=np.float32)}

    def policy_on_a_list(observations):
        gate.wait(5)
        image = observations[0]["image"]
        seen.append((len(observations), image.flags.writeable, np.array_equal(image, example["image"])))
        return [{"actions": np.zeros(3)} for _ in observations]

    async def go():
        app = App(policy_on_a_list, batch=[1, 2], example=example)
        try:
            warming = await _request(app, "GET", "/ping")         # the first request starts it, with no lifespan
            gate.set()
            for _ in range(200):
                if app.ready:
                    break
                await asyncio.sleep(0.01)
            return warming, await _request(app, "GET", "/ping"), await _request(app, "GET", "/example")
        finally:
            gate.set()
            app.close()

    (status, headers, body), (ready_status, _, _), (example_status, example_headers, example_body) = _run(go())
    assert status == 503 and headers["retry-after"] == "1" and b"warming up" in body
    assert ready_status == 200
    # twice at each size, on the example as a robot's observation reaches the policy: read back from msgpack, the
    # example's arrays as they were, each the policy's own
    assert seen == [(1, True, True), (1, True, True), (2, True, True), (2, True, True)]
    assert example_status == 200 and example_headers["content-type"] == msgpack_io.MEDIA_TYPE
    assert np.array_equal(msgpack_io.unpack(example_body)["image"], example["image"])


def test_a_policy_that_fails_on_the_example_stops_the_server_and_its_error_is_kept():
    stopped = []

    def policy(observation):
        raise RuntimeError("the checkpoint is missing")

    async def go():
        app = App(policy, example={"x": np.zeros(2)})
        app.on_failure = lambda: stopped.append(True)
        try:
            await _request(app, "GET", "/ping")
            for _ in range(200):
                if app.failure is not None:
                    break
                await asyncio.sleep(0.01)
            return app.failure, await _request(app, "GET", "/ping")
        finally:
            app.close()

    failure, (status, _, body) = _run(go())
    assert isinstance(failure, RuntimeError) and stopped == [True]
    assert status == 500 and b"RuntimeError: the checkpoint is missing" in body


def test_an_output_that_cannot_be_sent_fails_the_warm_up_not_the_first_robot():
    async def go():
        app = App(lambda o: {"actions": object()}, example={"x": 1})
        try:
            await _request(app, "GET", "/ping")
            for _ in range(200):
                if app.failure is not None:
                    break
                await asyncio.sleep(0.01)
            return app.failure
        finally:
            app.close()

    failure = _run(go())
    assert isinstance(failure, msgpack_io.MsgpackError) and "object cannot be sent as msgpack" in str(failure)


def test_without_an_example_the_server_is_ready_at_once_has_none_to_hand_out_and_says_what_that_leaves_out(caplog):
    async def go():
        app = App(lambda o: 1)
        try:
            return await _request(app, "GET", "/ping"), await _request(app, "GET", "/example")
        finally:
            app.close()

    with caplog.at_level("WARNING", logger="sequence_ai.serve"):
        (status, _, _), (example_status, _, body) = _run(go())
    assert status == 200 and example_status == 404 and b"serve(example=...)" in body
    assert [r.getMessage() for r in caplog.records if r.name == "sequence_ai.serve"] == [serving.NO_EXAMPLE]
    assert "not warmed up" in serving.NO_EXAMPLE and serving.NO_EXAMPLE.endswith("add a @seq.sample method to the policy")


def test_the_lifespan_starts_the_warm_up_and_its_end_stops_the_policys_thread():
    async def go():
        app = App(lambda o: 1, example={"x": 1})
        messages = [{"type": "lifespan.startup"}, {"type": "lifespan.shutdown"}]
        sent = []

        async def receive():
            if len(messages) == 1:
                for _ in range(200):                       # the shutdown comes once the warm-up is done
                    if app.ready:
                        break
                    await asyncio.sleep(0.01)
            return messages.pop(0)

        async def send(message):
            sent.append(message["type"])

        await app({"type": "lifespan"}, receive, send)
        return app.ready, sent, app._thread._shutdown

    ready, sent, shut = _run(go())
    assert ready and sent == ["lifespan.startup.complete", "lifespan.shutdown.complete"] and shut


# ── refusals ────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_with_a_token_every_request_needs_it():
    async def go():
        app = App(lambda o: 1, token="s3cret")
        try:
            return [await _request(app, "GET", "/ping"),
                    await _request(app, "GET", "/ping", headers=[("Authorization", "Bearer wrong")]),
                    await _act(app, {"x": 1}),                                      # refused at its observation
                    await _request(app, "POST", "/act", _act_call("e-alone"), headers=_ACT),
                    await _request(app, "GET", "/ping", headers=[("Authorization", "Bearer s3cret")]),
                    await _act(app, {"x": 1}, headers=[("Authorization", "bearer s3cret")])]
        finally:
            app.close()

    answers = _run(go())
    assert [status for status, _, _ in answers] == [401, 401, 401, 401, 200, 200]
    assert answers[0][1]["www-authenticate"] == "Bearer" and b"Bearer <token>" in answers[0][2]


def test_a_body_over_the_limit_is_refused_before_it_is_read_and_the_connection_closed():
    async def go():
        app = App(lambda o: 1, limit=1000)
        try:
            answers = []
            for path in ("/observe", "/act"):
                answers.append(await _request(app, "POST", path, b"x" * 2000))                      # declared
                answers.append(await _request(app, "POST", path, chunks=[b"x" * 600, b"x" * 600]))  # streamed
            return answers
        finally:
            app.close()

    for status, headers, body in _run(go()):
        assert status == 413 and headers["connection"] == "close" and b"limit of 1000 bytes" in body


@pytest.mark.parametrize("path, body, kind, reason, said", [
    ("/observe", b"\xc1", msgpack_io.MEDIA_TYPE, "observe", "msgpack type 0xc1 is not supported"),
    ("/observe", msgpack_io.pack({"state": [1.0]}), msgpack_io.MEDIA_TYPE, "observe", "episode_id"),
    ("/observe", b"{", "application/json", "observe", "Expecting property name"),
    ("/act", b"not json", "application/json", "body", "the act does not read"),
])
def test_a_body_that_does_not_read_is_a_400_that_says_why(path, body, kind, reason, said):
    async def go():
        app = App(lambda o: 1)
        try:
            return await _request(app, "POST", path, body, headers=[("Content-Type", kind)])
        finally:
            app.close()

    status, headers, text = _run(go())
    refused = json.loads(text)
    assert status == 400 and headers["content-type"] == "application/json"
    assert refused["code"] == "invalid_request" and refused["details"] == {"reason": reason}
    assert said in refused["detail"], refused["detail"]


def test_a_policy_that_raises_is_a_500_with_its_message_and_the_server_goes_on():
    def policy(observation):
        if observation["bad"]:
            raise ValueError("no camera named wrist")
        return "fine"

    async def go():
        app = App(policy)
        try:
            return await _act(app, {"bad": True}), await _act(app, {"bad": False})
        finally:
            app.close()

    (status, headers, body), (after, _, fine) = _run(go())
    refused = json.loads(body)
    assert status == 500 and headers["content-type"] == "application/json" and refused["code"] == "policy_error"
    assert refused["detail"] == "the policy raised ValueError: no camera named wrist"
    assert refused["details"] == {"exception": {"type": "ValueError", "message": "no camera named wrist"}}
    assert after == 200 and msgpack_io.unpack(fine) == "fine"


def test_an_output_that_cannot_be_sent_is_a_500_that_says_so():
    async def go():
        app = App(lambda o: {"actions": {1.5: "a float key"}})
        try:
            return await _act(app, {})
        finally:
            app.close()

    status, _, body = _run(go())
    assert status == 500 and b"cannot go back as msgpack" in body


def test_other_paths_and_methods():
    async def go():
        app = App(lambda o: 1)
        try:
            return [await _request(app, "GET", "/act"), await _request(app, "GET", "/observe"),
                    await _request(app, "GET", "/act/episode_end"), await _request(app, "POST", "/ping"),
                    await _request(app, "GET", "/infer"), await _request(app, "HEAD", "/ping")]
        finally:
            app.close()

    answers = _run(go())
    assert [(status, headers["allow"], body) for status, headers, body in answers[:4]] == [
        (405, "POST", b"POST /act"), (405, "POST", b"POST /observe"), (405, "POST", b"POST /act/episode_end"),
        (405, "GET, HEAD", b"GET /ping")]
    (other, _, body), (head, _, _) = answers[4:]
    assert other == 404 and body == (b"this server answers GET /stream, POST /observe, POST /act, "
                                     b"POST /act/episode_end, GET /ping and GET /example")
    assert head == 200


def test_what_is_not_a_policy_function_is_refused_with_what_to_pass():
    class OpenpiLike:
        def infer(self, observation):
            return observation

    async def policy(observation):
        return 1

    with pytest.raises(TypeError, match="an object with an infer method is served as policy.infer"):
        App(OpenpiLike())
    with pytest.raises(TypeError, match="async function"):
        App(policy)
    with pytest.raises(ValueError, match="must include 1"):
        App(lambda os: os, batch=[2, 4])


# ── the listening sockets ───────────────────────────────────────────────────────────────────────────────────────────
def test_with_no_host_it_listens_on_every_ipv4_address_and_every_ipv6_one_on_the_same_port():
    sockets = serving._listen(None, 0)
    try:
        assert sockets[0].family == socket.AF_INET and sockets[0].getsockname()[0] == "0.0.0.0"
        if socket.has_ipv6 and len(sockets) == 2:
            assert sockets[1].family == socket.AF_INET6 and sockets[1].getsockname()[0] == "::"
            assert sockets[1].getsockname()[1] == sockets[0].getsockname()[1]
            assert sockets[1].getsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY) != 0
    finally:
        for sock in sockets:
            sock.close()


def test_a_robot_reaches_it_over_ipv6():
    httpx = pytest.importorskip("httpx")
    try:
        serving._listen("::1", 0)[0].close()
    except OSError:
        pytest.skip("no IPv6 loopback on this machine")
    app = App(lambda o: {"ok": True})
    with _serving(app, "::1") as url, httpx.Client(base_url=url) as client:
        pinged = client.get("/ping")
        posted = client.post("/observe", content=_observed("e-v6", {}), headers=dict(_OBSERVE))
        acted = client.post("/act", content=_act_call("e-v6"), headers=dict(_ACT))
    assert url.startswith("http://[::1]:") and pinged.status_code == 200 and posted.json() == {"through": 0}
    assert msgpack_io.unpack(acted.content)["outputs"] == {"ok": True}


def test_a_port_held_but_not_answering_is_named_by_the_bind():
    """A port held by a socket that is bound but not listening - a probe to it is refused, so nothing "answers" - is
    still named, by the bind that then fails: the guard behind the answered-at probe."""
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))                         # bound, not listen()ing
        port = taken.getsockname()[1]
        with pytest.raises(OSError, match=f"port {port} on 127.0.0.1 is taken"):
            serving._listen("127.0.0.1", port)


@pytest.mark.parametrize("host", [None, "0.0.0.0", "127.0.0.1"])
def test_a_port_another_server_answers_on_this_machines_loopback_is_taken_for_every_host(host):
    """A server already answering on this machine's loopback takes the port whether serve() is given no host, a wildcard,
    or that loopback itself: macOS lets 0.0.0.0 bind beside another program's 127.0.0.1 at the same port (and SO_REUSEADDR
    lets a second 127.0.0.1 bind over a 0.0.0.0), and the robots on this machine would reach that other program - so
    serve() probes the loopback and refuses the port rather than split this machine's robots between the two."""
    with socket.socket() as other:
        other.bind(("127.0.0.1", 0))
        other.listen()
        port = other.getsockname()[1]
        with pytest.raises(OSError, match=f"port {port} is taken: a server already answers at 127.0.0.1:{port}"):
            serving._listen(host, port)


def test_answered_at_raises_for_a_server_there_and_returns_for_a_free_port():
    """answered_at is what `seq serve` calls before it says it is serving, so a taken port is one worded line, not a bind
    traceback after the "serving ..." line: it raises for a server answering on this machine (no host, wildcard, or the
    loopback itself) and returns for a port nothing listens on."""
    with socket.socket() as other:
        other.bind(("127.0.0.1", 0))
        other.listen()
        port = other.getsockname()[1]
        for host in (None, "0.0.0.0", "127.0.0.1"):
            with pytest.raises(OSError, match=f"port {port} is taken"):
                serving.answered_at(host, port)
    with socket.socket() as free:
        free.bind(("127.0.0.1", 0))
        free_port = free.getsockname()[1]                    # closed on leaving the block: nothing listens there now
    serving.answered_at("127.0.0.1", free_port)              # returns, raises nothing
    serving.answered_at(None, free_port)


# ── on a real port ──────────────────────────────────────────────────────────────────────────────────────────────────
@contextlib.contextmanager
def _serving(app: App, host: str = "127.0.0.1"):
    """`app` on uvicorn as serve() runs it, on `host` and a free port, in a thread; its URL."""
    server = serving._uvicorn(app, "warning")
    sockets = serving._listen(host, 0)
    thread = threading.Thread(target=server.run, kwargs={"sockets": sockets}, daemon=True)
    thread.start()
    try:
        for _ in range(500):
            if server.started:
                break
            time.sleep(0.01)
        yield serving._url(sockets[0])
    finally:
        server.should_exit = True
        thread.join(10)


def test_the_warm_up_failing_stops_a_real_server():
    app = App(lambda o: 1 / 0, example={"x": 1})
    server = serving._uvicorn(app, "critical")
    app.on_failure = lambda: setattr(server, "should_exit", True)
    sockets = serving._listen("127.0.0.1", 0)
    thread = threading.Thread(target=server.run, kwargs={"sockets": sockets}, daemon=True)
    thread.start()
    thread.join(20)
    assert not thread.is_alive() and isinstance(app.failure, ZeroDivisionError)


@pytest.fixture
def serve_log(monkeypatch):
    """The serve module's logger and the uvicorn and root loggers as a run leaves them, put back after the test."""
    import logging

    from sequence_ai.engine import serve as serve_module

    shown = logging.Handler()
    monkeypatch.setattr(logging.getLogger("uvicorn"), "handlers", [shown])
    monkeypatch.setattr(serve_module.log, "handlers", [])
    monkeypatch.setattr(logging.getLogger("sequence_ai"), "handlers", [])
    monkeypatch.setattr(serve_module.log, "propagate", True)
    monkeypatch.setattr(serve_module.log, "level", logging.NOTSET)
    monkeypatch.setattr(logging.getLogger(), "handlers", list(logging.getLogger().handlers))
    monkeypatch.setattr(logging.getLogger(), "level", logging.getLogger().level)
    return serve_module, shown


def test_the_ready_line_shows_when_a_library_set_the_root_logger_up_at_warning(serve_log):
    import logging

    serve_module, shown = serve_log
    root = logging.getLogger()
    root.addHandler(logging.NullHandler())
    root.setLevel(logging.WARNING)                 # as absl leaves it, imported by a JAX policy
    serve_module._log_with("info")
    assert shown in serve_module.log.handlers and serve_module.log.propagate is False   # uvicorn's handler shows them
    assert serve_module.log.isEnabledFor(logging.INFO)


def test_a_program_that_shows_info_on_its_root_logger_keeps_the_lines_there(serve_log):
    import logging

    serve_module, _ = serve_log
    root = logging.getLogger()
    root.addHandler(logging.NullHandler())
    root.setLevel(logging.INFO)
    serve_module._log_with("info")
    assert serve_module.log.handlers == [] and serve_module.log.propagate is True



def test_the_warm_up_tries_the_speed_ups_on_the_policys_thread_on_the_examples_as_robots_send_them(monkeypatch):
    """engine.optimize runs before the warm-up, on the policy's own thread, with the batch sizes and switches serve()
    was given, on each example as a robot's observation reaches the policy; /ping then names what it kept."""
    import sys

    from sequence_ai.engine.optimize import Optimized

    picture, asked = _picture(), []

    def stand_in(policy, examples, *, batch=None, say=print, **switches):
        image = examples[0]["image"]
        asked.append((threading.current_thread().name, len(examples), image.flags.writeable,
                      bool(np.abs(image.astype(int) - picture).mean() < 4), batch, switches))
        return Optimized(policy, {"on": ["cuda_graphs"]})

    monkeypatch.setattr(sys.modules["sequence_ai.engine.optimize"], "optimize", stand_in)
    examples = [{"image": picture, "state": np.zeros(7, dtype=np.float32)} for _ in range(3)]

    async def go():
        app = App(lambda observations: [{"actions": np.zeros(3)} for _ in observations], batch=[1, 2],
                  examples=examples, whole_graph=False)
        try:
            await _request(app, "GET", "/ping")
            for _ in range(200):
                if app.ready:
                    break
                await asyncio.sleep(0.01)
            return await _request(app, "GET", "/ping"), await _request(app, "GET", "/example")
        finally:
            app.close()

    (status, _, body), (example_status, _, example_body) = _run(go())
    assert status == 200 and json.loads(body)["optimized"] == ["cuda_graphs"]
    assert asked == [("sequence_ai-policy_0", 3, True, True, [1, 2],
                      {"cuda_graphs": None, "whole_graph": False, "torch_compile": False, "fused_kernels": False,
                       "tensorrt": False, "precision": None, "tolerance": None})]
    assert example_status == 200 and np.array_equal(msgpack_io.unpack(example_body)["image"], picture)


def test_a_server_with_no_speed_up_on_says_so_in_its_ping():
    async def go():
        app = App(lambda o: {"actions": np.zeros(2)}, example={"x": np.zeros(2)})
        try:
            await _request(app, "GET", "/ping")
            for _ in range(200):
                if app.ready:
                    break
                await asyncio.sleep(0.01)
            return await _request(app, "GET", "/ping")
        finally:
            app.close()

    status, _, body = _run(go())
    assert status == 200 and json.loads(body)["optimized"] == []       # no torch module: nothing to turn on


def test_a_speed_up_that_changes_the_outputs_without_a_tolerance_is_refused_before_the_server_starts():
    with pytest.raises(ValueError, match="torch_compile changes the policy's outputs: give tolerance="):
        App(lambda o: o, example={"x": np.zeros(2)}, torch_compile=True)
    with pytest.raises(ValueError, match="precision='fp16' changes the policy's outputs"):
        serving.serve(lambda o: o, port=0, precision="fp16")
