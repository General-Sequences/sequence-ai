# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
serve() and its app (sequence_ai/engine/serve.py): an observation in and the policy's output out, as msgpack, with what
the robot compressed or sent as JPEG restored first; the policy on one thread of its own, one call at a time, or on a
batch of the observations waiting; /ping answered while the policy runs, and 503 until the warm-up on the example is
done; the token; a body over the limit, one that is not msgpack, a policy that raises or returns what cannot be sent;
torch and JAX outputs as numpy; the listening sockets; and the server on a real port, with connect() on the robot side.

    python -m pytest -q tests/engine/test_serve.py
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import socket
import threading
import time

import numpy as np
import pytest

from sequence_ai.engine import serve as serving
from sequence_ai.engine.serve import App
from sequence_ai.network import compressed, frames, msgpack_io


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


async def _act(app: App, observation, headers=()):
    return await _request(app, "POST", "/act", msgpack_io.pack(observation), headers=headers)


def _picture(seed: int = 0) -> np.ndarray:
    """A 64 x 48 frame with detail in it, so that JPEG changes its values a little."""
    y, x = np.mgrid[0:48, 0:64]
    return np.stack([(x * 4 + seed) % 256, (y * 5 + 2 * seed) % 256, (x + y) * 2 % 256], axis=-1).astype(np.uint8)


# ── an observation in, the output out ──────────────────────────────────────────────────────────────────────────────
def test_an_observation_in_and_the_policys_output_out_with_the_servers_time_and_what_it_reads():
    def policy(observation):
        return {"actions": np.full((8, 7), observation["state"].sum(), dtype=np.float32), "note": "ok"}

    async def go():
        app = App(policy)
        try:
            return await _act(app, {"state": np.arange(4, dtype=np.float32)})
        finally:
            app.close()

    status, headers, body = _run(go())
    assert status == 200
    assert headers["content-type"] == msgpack_io.MEDIA_TYPE
    out = msgpack_io.unpack(body)
    assert out["note"] == "ok" and out["actions"].shape == (8, 7) and float(out["actions"][0, 0]) == 6.0
    # the time from the request's last byte to its response first: what connect() takes off the round trip
    assert re.fullmatch(r"act;dur=\d+\.\d, model;dur=\d+\.\d", headers["server-timing"]), headers["server-timing"]
    assert headers["x-seq-reads"] == "jpeg, zstd, zlib, once, split"


def test_what_the_server_says_it_reads_follows_its_switches():
    async def go(**switches):
        app = App(lambda o: 1, **switches)
        try:
            return (await _request(app, "GET", "/ping"))[1]["x-seq-reads"]
        finally:
            app.close()

    # and, whatever the switches, that it runs an act's copies once (network.lines)
    assert _run(go(jpeg=False)) == "zstd, zlib, once, split"
    assert _run(go(compress=False)) == "jpeg, once, split"
    assert _run(go(jpeg=False, compress=False)) == "once, split"


def test_pictures_sent_as_jpeg_and_arrays_sent_compressed_reach_the_policy_as_the_arrays_they_were():
    got = {}

    def policy(observation):
        got.update(observation)
        return 0

    picture = _picture()
    joints = np.linspace(0, 1, 5000).astype(np.float32)
    sent, count = frames.pack_pictures({"camera": {"wrist": picture}})
    assert count == 1
    sent["joints"] = compressed.compress(joints)

    async def go():
        app = App(policy, jpeg=False, compress=False)     # restored all the same: the switches change what it says
        try:
            return await _act(app, sent)
        finally:
            app.close()

    status, _, _ = _run(go())
    assert status == 200
    wrist = got["camera"]["wrist"]
    assert isinstance(wrist, np.ndarray) and wrist.shape == (48, 64, 3) and wrist.dtype == np.uint8
    assert not np.array_equal(wrist, picture)                                     # JPEG: close, not equal
    assert np.abs(wrist.astype(int) - picture.astype(int)).mean() < 4
    assert np.array_equal(got["joints"], joints)                                  # compressed: byte for byte


def test_the_policys_arrays_are_its_own_to_write_into():
    """msgpack reads arrays in place from the request's bytes, read-only: torch.from_numpy warns on each, and a policy
    that writes into its observation would fail."""
    def policy(observation):
        observation["state"][0] = 9.0
        observation["camera"][0][0, 0, 0] = 1
        return {"state": observation["state"], "writable": observation["camera"][0].flags.writeable}

    sent, _ = frames.pack_pictures({"state": np.zeros(3, dtype=np.float32), "camera": [_picture()]})

    async def go():
        app = App(policy)
        try:
            return await _act(app, sent)
        finally:
            app.close()

    status, _, body = _run(go())
    out = msgpack_io.unpack(body)
    assert status == 200, body
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
    gate = threading.Event()

    def policy(observation):
        gate.wait(5)
        return 1

    async def go():
        app = App(policy)
        try:
            acting = asyncio.ensure_future(_act(app, {"x": 1}))
            await asyncio.sleep(0.05)                      # the act is in the policy's thread, which waits on the gate
            started = time.perf_counter()
            pinged = await _request(app, "GET", "/ping")
            took = time.perf_counter() - started
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
        app = App(policy_on_a_list, batch=[1, 2, 4], wait_ms=1.0)
        try:
            first = asyncio.ensure_future(_act(app, {"i": 0}))
            while not seen:                                # the first observation is in the policy, alone
                await asyncio.sleep(0.005)
            rest = [asyncio.ensure_future(_act(app, {"i": i})) for i in range(1, 6)]
            await asyncio.sleep(0.05)                      # five more wait while the policy runs
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
    sizes = []

    def policy_on_a_list(observations):
        sizes.append(len(observations))
        return returned(observations)

    async def go():
        app = App(policy_on_a_list, batch=[1, 2])
        try:
            return await asyncio.gather(_act(app, {"i": 1}), _act(app, {"i": 2}))
        finally:
            app.close()

    answers = _run(go())
    # the batch of 2 failed, so each observation was run again alone, and failed alone, saying why
    assert sizes == [2, 1, 1]
    for status, _, body in answers:
        assert status == 500 and said in body.decode()


# ── warm-up ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_the_warm_up_runs_the_example_twice_at_each_batch_size_and_ping_says_503_until_it_is_done():
    gate, seen = threading.Event(), []
    example = {"image": _picture(), "state": np.zeros(7, dtype=np.float32)}

    def policy_on_a_list(observations):
        gate.wait(5)
        image = observations[0]["image"]
        seen.append((len(observations), image.flags.writeable, np.array_equal(image, example["image"]),
                     bool(np.abs(image.astype(int) - example["image"]).mean() < 4)))
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
    # twice at each size, on the example as a robot's observation reaches the policy: its own arrays, its picture sent
    # as JPEG and restored (close to the example's, not equal)
    assert seen == [(1, True, False, True), (1, True, False, True), (2, True, False, True), (2, True, False, True)]
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


def test_without_an_example_the_server_is_ready_at_once_and_has_none_to_hand_out():
    async def go():
        app = App(lambda o: 1)
        try:
            return await _request(app, "GET", "/ping"), await _request(app, "GET", "/example")
        finally:
            app.close()

    (status, _, _), (example_status, _, body) = _run(go())
    assert status == 200 and example_status == 404 and b"serve(example=...)" in body


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
                    await _act(app, {"x": 1}),
                    await _request(app, "GET", "/ping", headers=[("Authorization", "Bearer s3cret")]),
                    await _act(app, {"x": 1}, headers=[("Authorization", "bearer s3cret")])]
        finally:
            app.close()

    answers = _run(go())
    assert [status for status, _, _ in answers] == [401, 401, 401, 200, 200]
    assert answers[0][1]["www-authenticate"] == "Bearer" and b"Bearer <token>" in answers[0][2]


def test_a_body_over_the_limit_is_refused_before_it_is_read_and_the_connection_closed():
    async def go():
        app = App(lambda o: 1, limit=1000)
        try:
            declared = await _request(app, "POST", "/act", b"x" * 2000)
            streamed = await _request(app, "POST", "/act", chunks=[b"x" * 600, b"x" * 600])
            return declared, streamed
        finally:
            app.close()

    for status, headers, body in _run(go()):
        assert status == 413 and headers["connection"] == "close" and b"limit of 1000 bytes" in body


def test_a_picture_that_would_decode_past_the_limit_is_refused_before_it_is_decoded():
    sent, _ = frames.pack_pictures({"image": _picture()})            # 64 x 48 x 3 = 9,216 bytes once decoded
    body = msgpack_io.pack(sent)

    async def go():
        app = App(lambda o: 1, limit=9_000)
        try:
            return await _request(app, "POST", "/act", body)
        finally:
            app.close()

    status, _, text = _run(go())
    assert len(body) < 9_000 and status == 400
    assert b"decodes to 9,216 bytes, past the limit of 9,000" in text, text


@pytest.mark.parametrize("body, said", [
    (b"\xc1", "not an observation this server can read"),
    (msgpack_io.pack({"__seq__": "jpeg", "v": 1, "data": b"not a jpeg"}), "not a JPEG or PNG"),
    (msgpack_io.pack({"__seq__": "compressed", "v": 1, "codec": "zstd", "n": 5, "data": b"xx"}), "zstd"),
])
def test_a_body_that_is_not_an_observation_is_a_400_that_says_why(body, said):
    async def go():
        app = App(lambda o: 1)
        try:
            return await _request(app, "POST", "/act", body)
        finally:
            app.close()

    status, _, text = _run(go())
    assert status == 400 and said in text.decode()


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

    (status, _, body), (after, _, fine) = _run(go())
    assert status == 500 and body == b"the policy raised ValueError: no camera named wrist"
    assert after == 200 and msgpack_io.unpack(fine) == "fine"


def test_an_output_that_cannot_be_sent_is_a_500_that_says_so():
    async def go():
        app = App(lambda o: {"actions": {1.5: "a float key"}})
        try:
            return await _act(app, {})
        finally:
            app.close()

    status, _, body = _run(go())
    assert status == 500 and b"cannot be sent as msgpack" in body


def test_other_paths_and_methods():
    async def go():
        app = App(lambda o: 1)
        try:
            return [await _request(app, "GET", "/act"), await _request(app, "POST", "/ping"),
                    await _request(app, "GET", "/infer"), await _request(app, "HEAD", "/ping")]
        finally:
            app.close()

    (get_act, h1, _), (post_ping, h2, _), (other, _, body), (head, _, _) = _run(go())
    assert (get_act, h1["allow"]) == (405, "POST") and (post_ping, h2["allow"]) == (405, "GET, HEAD")
    assert other == 404 and b"POST /act, GET /ping and GET /example" in body
    assert head == 200


def test_what_is_not_a_policy_function_is_refused_with_what_to_pass():
    class OpenpiLike:
        def infer(self, observation):
            return observation

    async def policy(observation):
        return 1

    with pytest.raises(TypeError, match="an openpi Policy is served as policy.infer"):
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
    with _serving(app, "::1") as url:
        with httpx.Client(base_url=url) as client:
            pinged = client.get("/ping")
            acted = client.post("/act", content=msgpack_io.pack({}))
    assert url.startswith("http://[::1]:") and pinged.status_code == 200
    assert msgpack_io.unpack(acted.content) == {"ok": True}


def test_a_port_that_is_taken_is_named():
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = taken.getsockname()[1]
        with pytest.raises(OSError, match=f"port {port} on 127.0.0.1 is taken"):
            serving._listen("127.0.0.1", port)


@pytest.mark.parametrize("host", [None, "0.0.0.0"])
def test_a_port_another_server_answers_on_this_machines_loopback_is_taken_for_every_address(host):
    """macOS binds 0.0.0.0 beside another program's 127.0.0.1 at the same port; the robots on this machine would reach
    that program, so serve() refuses the port on every system."""
    with socket.socket() as other:
        other.bind(("127.0.0.1", 0))
        other.listen()
        port = other.getsockname()[1]
        with pytest.raises(OSError, match=f"port {port} is taken: a server already answers at 127.0.0.1:{port}"):
            serving._listen(host, port)


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


@pytest.mark.parametrize("jpeg", [True, False])
def test_a_robots_connect_gets_its_actions_and_sends_pictures_as_the_server_says(jpeg):
    pytest.importorskip("httpx")
    from sequence_ai.network.link import connect

    got = []

    def policy(observation):
        got.append(observation["image"])
        return {"actions": np.full((8, 7), float(observation["image"].mean()), dtype=np.float32)}

    picture = _picture(3)
    app = App(policy, jpeg=jpeg, example={"image": picture, "state": np.zeros(7, dtype=np.float32)})
    with _serving(app) as url:
        with connect(url, check=False) as link:
            link.wait_until_ready(10)
            for _ in range(3):
                out = link.act({"image": picture, "state": np.zeros(7, dtype=np.float32)})
            told = dict(link.server or {})
    assert out["actions"].shape == (8, 7)
    assert told["server"] == "sequence-ai" and told["pid"] == os.getpid()    # what the robot was told the server is
    received = got[-1]
    assert received.shape == picture.shape and received.dtype == np.uint8
    if jpeg:     # the robot followed the server: sent as JPEG, close to the frame and not equal to it
        assert not np.array_equal(received, picture) and np.abs(received.astype(int) - picture).mean() < 4
    else:        # the server does not read JPEG: sent as the array, exactly
        assert np.array_equal(received, picture)
    assert app._thread._shutdown                           # the server's end stopped the policy's thread


@pytest.mark.parametrize("jpeg", [True, False])
def test_a_frame_fitted_as_the_docs_fit_it_reaches_the_policy_as_the_picture_it_is(jpeg):
    """frames.resize_image returns an RGB PIL image; handed to act() as the docs fit a frame, it is sent as the picture it
    is - as JPEG where the server reads JPEG, as its array where it does not - and the policy reads an HxWx3 uint8 array,
    where it was refused "Image cannot be sent as msgpack"."""
    pytest.importorskip("httpx")
    pytest.importorskip("PIL")
    from sequence_ai.network import frames
    from sequence_ai.network.link import connect

    class _Camera:
        """camera_user of docs/components.md: what reads a frame, here a 64 x 48 one."""

        def read(self):
            return _picture(5)

    camera_user = _Camera()
    got = []

    def policy(observation):
        got.append(observation["image"])
        return {"actions": np.zeros((8, 7), dtype=np.float32)}

    app = App(policy, jpeg=jpeg, example={"image": np.zeros((224, 224, 3), dtype=np.uint8)})
    # docs/components.md, as written there
    frame_user = frames.resize_image(camera_user.read(), 224, 224, mode="pad", filter="bilinear")
    with _serving(app) as url:
        with connect(url, check=False) as link:
            link.wait_until_ready(10)
            link.act({"image": frame_user})
    received, exact = got[-1], np.asarray(frame_user)
    assert received.shape == (224, 224, 3) and received.dtype == np.uint8
    if jpeg:     # as JPEG: close to the frame, not equal to it
        assert not np.array_equal(received, exact) and np.abs(received.astype(int) - exact).mean() < 4
    else:        # as its array, exactly
        assert np.array_equal(received, exact)


def test_a_robot_that_acts_at_once_sends_its_first_picture_as_small_as_the_next_ones():
    """A robot that acts without wait_until_ready() has heard nothing of what the server reads: its first act asks the
    ping first, so its pictures already go as JPEG, and the first upload is the size of the ones after it - where a
    first act of 301,473 bytes, sent as arrays, took 1815 ms and the next ones were about 15 KB."""
    httpx = pytest.importorskip("httpx")
    from sequence_ai.network.link import connect

    picture = np.tile(_picture(7), (5, 4, 1))                     # 240 x 256: a camera-sized picture
    app = App(lambda observation: {"actions": np.zeros((8, 7), dtype=np.float32)}, example={"image": picture})
    with _serving(app) as url:
        for _ in range(200):                                       # ready, as another program sees it: the link hears nothing
            if httpx.get(url + "/ping").status_code == 200:
                break
            time.sleep(0.05)
        with connect(url, check=False) as link:
            assert link.reads is None
            sizes = []
            for _ in range(3):
                link.act({"image": picture, "state": np.zeros(7, dtype=np.float32)})
                sizes.append(link.last_body_bytes)
            pictures = link.stats()["counters"]["pictures"]
    assert sizes[0] == sizes[1] == sizes[2] and sizes[0] < picture.nbytes / 4, sizes
    assert pictures == 3                                           # each act's picture went as JPEG, the first's too


def test_the_top_level_connect_takes_the_switches_the_docs_give_it():
    """`sequence_ai.connect(url, jpeg=False)`, as the docs write it, reaches a real serve() with the switch on the
    connection: the picture goes as its array, exactly; compress= and first_timeout= reach it too. Given with a policy's
    name, each of them is refused, as token= is."""
    pytest.importorskip("httpx")
    import sequence_ai
    from sequence_ai.errors import InvalidRequest

    got = []

    def policy(observation):
        got.append(observation["image"])
        return {"actions": np.zeros((8, 7), dtype=np.float32)}

    picture = _picture(9)
    app = App(policy, example={"image": picture})                  # serve()'s default: it reads JPEG
    with _serving(app) as url:
        with sequence_ai.connect(url, jpeg=False) as link:          # the docs' words
            link.wait_until_ready(10)
            link.act({"image": picture})
            assert not link.sends_jpeg
        with sequence_ai.connect(url, compress=False, first_timeout=5.0) as link:
            assert link._compress is False and link._first_timeout == 5.0
    assert np.array_equal(got[-1], picture)                         # the array, exactly: no JPEG on the way
    for switch in ({"jpeg": False}, {"compress": False}, {"first_timeout": 5.0}, {"token": "t"}):
        with pytest.raises(InvalidRequest, match="given by its address|policy server of your own"):
            sequence_ai.connect("my_policy_user", **switch)


def test_the_warm_up_failing_stops_a_real_server():
    app = App(lambda o: 1 / 0, example={"x": 1})
    server = serving._uvicorn(app, "critical")
    app.on_failure = lambda: setattr(server, "should_exit", True)
    sockets = serving._listen("127.0.0.1", 0)
    thread = threading.Thread(target=server.run, kwargs={"sockets": sockets}, daemon=True)
    thread.start()
    thread.join(20)
    assert not thread.is_alive() and isinstance(app.failure, ZeroDivisionError)


# ── an act's copies run it once (REQUEST_HEADER): a robot's copies of one act over each of its uplinks ──────────────────
def _rid(value: str = "r-0000abcd"):
    return [("X-Seq-Request-Id", value)]


def test_copies_of_one_act_run_the_policy_once_and_each_gets_its_answer():
    """Two copies of one act - one X-Seq-Request-Id, as network.lines sends one over each uplink - reach the policy once:
    the copy that comes while the first is running waits for its answer and gets it, marked a copy, and so does one that
    comes after it, from the answer kept; another id runs the policy again, and a request with no id runs it each time.
    [fails before: the policy ran for each copy, and no answer was marked a copy]"""
    calls: list = []
    release = threading.Event()

    def policy(observation):
        calls.append(float(observation["state"][0]))
        release.wait(5)
        return {"n": len(calls)}

    async def go():
        app = App(policy)
        try:
            first = asyncio.ensure_future(_act(app, {"state": np.ones(1)}, headers=_rid()))
            await asyncio.sleep(0.05)
            second = asyncio.ensure_future(_act(app, {"state": np.ones(1)}, headers=_rid()))
            await asyncio.sleep(0.05)
            release.set()
            got = [await first, await second, await _act(app, {"state": np.ones(1)}, headers=_rid())]
            got.append(await _act(app, {"state": np.ones(1)}, headers=_rid("r-0000abce")))
            got.append(await _act(app, {"state": np.ones(1)}))
            got.append(await _act(app, {"state": np.ones(1)}))
            return got
        finally:
            app.close()

    first, second, late, other, plain, again = _run(go())
    assert [msgpack_io.unpack(r[2])["n"] for r in (first, second, late)] == [1, 1, 1]
    assert [r[1].get("x-seq-request-copy") for r in (first, second, late)] == [None, "1", "1"]
    assert all(r[0] == 200 and re.fullmatch(r"act;dur=\d+\.\d, model;dur=\d+\.\d", r[1]["server-timing"])
               for r in (first, second, late))
    assert [msgpack_io.unpack(r[2])["n"] for r in (other, plain, again)] == [2, 3, 4]
    assert len(calls) == 4 and "once" in first[1]["x-seq-reads"].split(", ")


def test_a_copy_of_an_act_the_policy_raised_on_gets_the_same_500_and_the_policy_is_not_run_again():
    calls: list = []

    def policy(observation):
        calls.append(1)
        raise RuntimeError("no gripper")

    async def go():
        app = App(policy)
        try:
            return [await _act(app, {"state": np.ones(1)}, headers=_rid()) for _ in range(2)]
        finally:
            app.close()

    first, copy = _run(go())
    assert first[0] == copy[0] == 500 and first[2] == copy[2] and b"no gripper" in copy[2]
    assert copy[1].get("x-seq-request-copy") == "1" and len(calls) == 1


def test_an_answer_is_kept_for_its_copies_only_so_long(monkeypatch):
    """Past ONCE_KEEP_S an act's answer is let go: a copy that comes later runs the policy again."""
    calls: list = []
    monkeypatch.setattr(serving, "ONCE_KEEP_S", 0.0)

    async def go():
        app = App(lambda o: calls.append(1) or {"n": len(calls)})
        try:
            first = await _act(app, {"state": np.ones(1)}, headers=_rid())
            await asyncio.sleep(0.01)
            return first, await _act(app, {"state": np.ones(1)}, headers=_rid())
        finally:
            app.close()

    first, later = _run(go())
    assert [msgpack_io.unpack(r[2])["n"] for r in (first, later)] == [1, 2]
    assert later[1].get("x-seq-request-copy") is None


# ── an act's halves (HALF_HEADER): a robot's act in two halves, each on a connection of its own ───────────────────────
def _half(n: int, rid: str = "r-0000abcd"):
    return [("X-Seq-Request-Id", rid), ("X-Seq-Half", f"{n}/2")]


@pytest.mark.parametrize("second_first", [False, True], ids=["first-half-first", "second-half-first"])
def test_an_acts_two_halves_are_joined_in_either_order_and_run_once(second_first):
    """The body's two halves reach the policy as the observation they make together, once, in whichever order they
    come: the first half gets the answer, with the server's time from when the act was whole, and the second 202."""
    calls: list = []

    def policy(observation):
        calls.append(observation)
        return {"sum": int(observation["noise"].astype(np.int64).sum())}

    noise = np.random.default_rng(1).integers(0, 256, 5000, dtype=np.uint8)
    body = msgpack_io.pack({"noise": noise})
    cut = len(body) // 2

    async def go():
        app = App(policy)
        try:
            def first():
                return _request(app, "POST", "/act", body[:cut], headers=_half(1))

            def second():
                return _request(app, "POST", "/act", body[cut:], headers=_half(2))

            if second_first:
                got_second = await second()
                return await first(), got_second
            waiting = asyncio.ensure_future(first())
            await asyncio.sleep(0.05)
            assert not waiting.done()                       # the first half waits for the second
            got_second = await second()
            return await waiting, got_second
        finally:
            app.close()

    first, second = _run(go())
    assert first[0] == 200 and msgpack_io.unpack(first[2])["sum"] == int(noise.astype(np.int64).sum())
    assert second[0] == 202 and second[2] == b""
    assert len(calls) == 1 and np.array_equal(calls[0]["noise"], noise)
    assert "split" in first[1]["x-seq-reads"].split(", ") and first[1].get("x-seq-request-copy") is None
    assert re.fullmatch(r"act;dur=\d+\.\d, model;dur=\d+\.\d", first[1]["server-timing"])


def test_a_first_half_whose_second_does_not_come_is_answered_424_and_the_act_runs_once_sent_whole(monkeypatch):
    """Past HALF_WAIT_S the first half is answered 424 and nothing runs: the robot sends the act whole, and that runs it
    once; a second half that comes after the wait finds nothing to join."""
    monkeypatch.setattr(serving, "HALF_WAIT_S", 0.1)
    calls: list = []
    body = msgpack_io.pack({"state": np.ones(3)})

    async def go():
        app = App(lambda o: calls.append(1) or {"n": len(calls)})
        try:
            first = await _request(app, "POST", "/act", body[:5], headers=_half(1))
            late = await _request(app, "POST", "/act", body[5:], headers=_half(2))
            ran = len(calls)
            return first, late, ran, await _request(app, "POST", "/act", body, headers=_rid())
        finally:
            app.close()

    first, late, ran, whole = _run(go())
    assert first[0] == 424 and b"send the act whole" in first[2] and late[0] == 202 and ran == 0
    assert whole[0] == 200 and msgpack_io.unpack(whole[2])["n"] == 1 and len(calls) == 1


def test_a_half_of_an_act_that_ran_whole_gets_its_answer_and_a_whole_copy_answers_a_first_half_waiting():
    """network.lines may send one act whole over one uplink and in halves over another: whichever is whole first runs
    it, once, and every half of it gets what it got, the first halves marked a copy."""
    calls: list = []
    body = msgpack_io.pack({"state": np.ones(3)})

    async def go():
        app = App(lambda o: calls.append(1) or {"n": len(calls)})
        try:
            whole = await _request(app, "POST", "/act", body, headers=_rid())
            first = await _request(app, "POST", "/act", body[:5], headers=_half(1))
            second = await _request(app, "POST", "/act", body[5:], headers=_half(2))
            waiting = asyncio.ensure_future(_request(app, "POST", "/act", body[:5], headers=_half(1, "r-0000abce")))
            await asyncio.sleep(0.05)
            other = await _request(app, "POST", "/act", body, headers=_rid("r-0000abce"))
            return whole, first, second, await waiting, other
        finally:
            app.close()

    whole, first, second, woken, other = _run(go())
    assert [msgpack_io.unpack(r[2])["n"] for r in (whole, first, woken, other)] == [1, 1, 2, 2] and len(calls) == 2
    assert [r[1].get("x-seq-request-copy") for r in (whole, first, woken, other)] == [None, "1", "1", None]
    assert second[0] == 202


@pytest.mark.parametrize("headers", [[("X-Seq-Half", "1/2")], [("X-Seq-Request-Id", "r-0000abcd"), ("X-Seq-Half", "3/2")]],
                         ids=["no-request-id", "not-a-half"])
def test_a_half_with_no_act_or_no_half_named_is_a_400_that_says_so(headers):
    async def go():
        app = App(lambda o: {"n": 1})
        try:
            return await _request(app, "POST", "/act", b"\x80", headers=headers)
        finally:
            app.close()

    status, _, said = _run(go())
    assert status == 400 and b"X-Seq-Half is 1/2 or 2/2" in said


def test_halves_that_together_are_over_the_limit_are_a_413():
    async def go():
        app = App(lambda o: {"n": 1}, limit=100)
        try:
            waiting = asyncio.ensure_future(_request(app, "POST", "/act", b"\x00" * 60, headers=_half(1)))
            await asyncio.sleep(0.01)
            await _request(app, "POST", "/act", b"\x00" * 60, headers=_half(2))
            return await waiting
        finally:
            app.close()

    status, _, said = _run(go())
    assert status == 413 and b"halves are over this server's limit of 100 bytes" in said


def test_halves_kept_are_let_go_past_their_wait_and_past_how_many_acts_may_wait(monkeypatch):
    """Only the halves of the HALF_KEEP_N latest acts are kept, each HALF_WAIT_S at most: a robot that sends second halves
    alone holds no more than that."""
    monkeypatch.setattr(serving, "HALF_KEEP_N", 3)

    async def go():
        app = App(lambda o: {"n": 1})
        try:
            for i in range(5):
                await _request(app, "POST", "/act", b"\x00", headers=_half(2, f"r-0000000{i}"))
            kept = list(app._halves)
            monkeypatch.setattr(serving, "HALF_WAIT_S", 0.0)
            await asyncio.sleep(0.01)
            await _request(app, "POST", "/act", b"\x00", headers=_half(2, "r-00000009"))
            return kept, list(app._halves)
        finally:
            app.close()

    kept, after = _run(go())
    assert kept == [b"r-00000002", b"r-00000003", b"r-00000004"] and after == [b"r-00000009"]


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

