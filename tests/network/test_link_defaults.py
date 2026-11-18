# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
connect()'s defaults: pictures as JPEG and compression while the server's X-Seq-Reads says it restores them, a longer
wait for the first act, wait_until_ready() until the ping answers 200, a token, and the wait behind a ping in flight
counted - against a server on this machine that says what it reads and records each body it is sent.
"""
import asyncio
import json
import threading
import time
import warnings

import numpy as np
import pytest

httpx = pytest.importorskip("httpx")
pytest.importorskip("uvicorn")
pytest.importorskip("PIL")

from sequence_ai.network import compressed, connect, frames, msgpack_io  # noqa: E402
from tests.network.test_link import frame, serving  # noqa: E402


class Server:
    """A policy server that names what it restores (`reads`, None: no X-Seq-Reads), answers its first pings with
    `pings` (then 200), takes `first_act_s` for its first act and `act_s` for the others, and wants `token`."""

    def __init__(self, reads="jpeg, zstd, zlib", *, pings=(), act_s=0.0, first_act_s=None, ping_s=0.0,
                 token=None, says_who=True) -> None:
        self.reads, self.pings, self.act_s, self.first_act_s = reads, list(pings), act_s, first_act_s
        self.ping_s, self.token, self.says_who = ping_s, token, says_who
        self.bodies, self.paths, self.auth = [], [], []
        self.one_at_a_time = None                   # an asyncio.Lock, made in the server's loop: one act at a time

    async def __call__(self, scope, receive, send) -> None:
        body = b""
        while True:
            message = await receive()
            body += message.get("body", b"")
            if not message.get("more_body"):
                break
        path, asked = scope["path"], dict(scope["headers"])
        self.paths.append(path)
        self.auth.append(asked.get(b"authorization"))
        headers = [(b"x-seq-reads", self.reads.encode())] if self.reads is not None else []
        if self.says_who:
            headers.append((b"x-seq-server", b"sequence-ai/0.63.0"))
        status, content = 200, b""
        if self.token and asked.get(b"authorization") != f"Bearer {self.token}".encode():
            status = 401
        elif path == "/ping":
            await asyncio.sleep(self.ping_s)
            status = self.pings.pop(0) if self.pings else 200
            if status == 503:
                headers.append((b"retry-after", b"0"))
            elif self.says_who:
                headers.append((b"content-type", b"application/json"))
                content = json.dumps({"server": "sequence-ai", "version": "0.63.0", "up_s": 5.0, "pid": 4242,
                                      "policy": "toy"}).encode()
        elif path == "/act":
            self.bodies.append(body)
            first = len(self.bodies) == 1 and self.first_act_s is not None
            if self.one_at_a_time is None:
                self.one_at_a_time = asyncio.Lock()
            async with self.one_at_a_time:              # as a server with one model runs them
                await asyncio.sleep(self.first_act_s if first else self.act_s)
            observation = frames.unpack_pictures(compressed.expand(msgpack_io.unpack(body), limit=64 << 20)[0])
            content = msgpack_io.pack({"image": observation["image"]})
            headers += [(b"content-type", msgpack_io.MEDIA_TYPE.encode()), (b"server-timing", b"act;dur=1.0")]
        await send({"type": "http.response.start", "status": status,
                    "headers": headers + [(b"content-length", str(len(content)).encode())]})
        await send({"type": "http.response.body", "body": content})

    def sent(self, i: int = -1) -> dict:
        """The observation of the i-th act as it was sent: the maps in place of its pictures and compressed items."""
        return msgpack_io.unpack(self.bodies[i])


def observation(step: int = 0) -> dict:
    return {"image": frame(step), "state": np.arange(7, dtype=np.float32)}


def test_after_the_server_says_it_restores_jpeg_each_picture_goes_as_jpeg_and_comes_back_as_its_array():
    server = Server()
    with serving(server) as url, connect(url, ping_path="/ping", check=False) as link:
        link.wait_until_ready()
        assert link.reads == frozenset({"jpeg", "zstd", "zlib"}) and link.sends_jpeg
        answer = link.act(observation())
    sent = server.sent()
    assert frames.is_packed_picture(sent["image"]) and isinstance(sent["state"], np.ndarray)
    assert len(server.bodies[0]) < frame().nbytes / 5                         # 230,400 bytes as an array
    assert answer["image"].shape == (240, 320, 3) and answer["image"].dtype == np.uint8
    assert np.abs(answer["image"].astype(int) - frame().astype(int)).mean() < 3     # JPEG: close, not equal
    assert link.stats()["counters"]["pictures"] == 1


def test_before_any_answer_and_from_a_server_that_names_nothing_pictures_go_as_arrays():
    server = Server(reads=None)
    with serving(server) as url, connect(url, ping_path=None, check=False) as link:
        link.act(observation())
        link.act(observation())
        assert link.reads == frozenset() and not link.sends_jpeg and not link.compresses
    assert all(isinstance(msgpack_io.unpack(body)["image"], np.ndarray) for body in server.bodies)


def test_the_first_act_goes_as_arrays_and_the_next_as_jpeg_once_its_answer_named_jpeg():
    server = Server(reads="jpeg")
    with serving(server) as url, connect(url, ping_path=None, check=False) as link:
        link.act(observation())
        link.act(observation())
    assert isinstance(server.sent(0)["image"], np.ndarray) and frames.is_packed_picture(server.sent(1)["image"])


def test_jpeg_false_sends_arrays_whatever_the_server_says_and_the_latest_x_seq_reads_counts():
    server = Server(reads="jpeg")
    with serving(server) as url:
        with connect(url, ping_path=None, jpeg=False, check=False) as link:
            link.act(observation())
            link.act(observation())
        assert all(isinstance(msgpack_io.unpack(body)["image"], np.ndarray) for body in server.bodies)
        with connect(url, ping_path=None, check=False) as link:
            link.act(observation())
            server.reads = ""                                    # the server stops restoring them
            link.act(observation())
            assert link.sends_jpeg is False
            link.act(observation())
    assert frames.is_packed_picture(server.sent(-2)["image"]) and isinstance(server.sent(-1)["image"], np.ndarray)


def test_compression_follows_the_codecs_the_server_names():
    big = {"image": frame(), "depth": np.zeros(200_000, np.float32)}      # 800 KB that compresses well
    server = Server(reads="zlib")
    with serving(server) as url, connect(url, ping_path=None, check=False) as link:
        link.act(big)
        link.act(big)
        assert link.compresses
    item = server.sent(-1)["depth"]
    assert compressed.is_compressed(item) and item["codec"] == "zlib"
    assert not frames.is_packed_picture(server.sent(-1)["image"])           # "jpeg" not named: no JPEG
    server = Server(reads="")
    with serving(server) as url, connect(url, ping_path=None, check=False) as link:
        link.act(big)
        link.act(big)
        assert not link.compresses
    assert isinstance(server.sent(-1)["depth"], np.ndarray)


def test_the_first_act_waits_first_timeout_and_the_next_ones_timeout():
    server = Server(act_s=0.6, first_act_s=0.6)
    with serving(server) as url, connect(url, ping_path=None, timeout=0.3, first_timeout=3.0, check=False) as link:
        link.act(observation())                                  # 0.6 s: within the first act's wait
        with pytest.raises(httpx.ReadTimeout):
            link.act(observation())                              # 0.6 s: past the next ones'


def test_wait_until_ready_asks_until_the_ping_answers_200_and_warms_up_with_an_example():
    server = Server(pings=[503, 503, 503])
    with serving(server) as url, connect(url, check=False) as link:
        link.wait_until_ready(timeout_s=10.0, poll_s=0.01, example=observation())
    assert server.paths[:4] == ["/ping"] * 4 and server.paths[4] == "/act"


def test_wait_until_ready_gives_up_past_its_time_and_says_the_last_answer():
    server = Server(pings=[503] * 1000)
    with serving(server) as url, connect(url, check=False) as link:
        with pytest.raises(TimeoutError, match="503"):
            link.wait_until_ready(timeout_s=0.3, poll_s=0.05)


def test_a_server_started_with_a_token_refuses_none_and_takes_it_as_a_bearer_token():
    server = Server(token="s3cret")
    with serving(server) as url:
        with connect(url, check=False) as link:
            with pytest.raises(httpx.HTTPStatusError) as refused:
                link.wait_until_ready(timeout_s=5.0)
            assert refused.value.response.status_code == 401
        with connect(url, token="s3cret", check=False) as link:
            link.wait_until_ready(timeout_s=5.0)
            assert link.act(observation())["image"].shape == (240, 320, 3)


def test_an_act_queued_behind_a_ping_in_flight_counts_that_wait():
    server = Server(ping_s=0.3)
    with serving(server) as url, connect(url, check=False) as link:
        link.wait_until_ready()
        ping = threading.Thread(target=link._ping)                 # as the idle pings send one
        ping.start()
        time.sleep(0.05)
        link.act(observation())
        ping.join()
        snap = link.stats()
    assert snap["counters"]["acts_behind_ping"] == 1
    waited, act = snap["latency_us"]["ping_wait_us"]["last"], snap["latency_us"]["act_us"]["last"]
    assert waited >= 150_000 and act >= waited                    # in the act's time too: it counts from the queue


def test_an_act_past_its_wait_says_why_and_what_to_do_and_the_next_waits_as_long_as_a_first():
    server = Server(act_s=0.0, first_act_s=0.5)          # the first inference compiles: 0.5 s, the next ones none
    with serving(server) as url, connect(url, ping_path=None, timeout=0.1, first_timeout=0.3, check=False) as link:
        with pytest.raises(httpx.ReadTimeout) as late:
            link.act(observation())
        said = str(late.value)
        assert "in 0.3 s" in said and "may still be running this act" in said
        assert "wait_until_ready(example=observation)" in said and "first_timeout=" in said
        link.act(observation())                          # behind the first, still running: 0.2 s, past timeout=0.1


def test_a_server_that_does_not_say_it_is_serve_is_told_once_and_one_that_does_is_not(monkeypatch):
    from sequence_ai.network import wifi_power, window

    monkeypatch.setattr(wifi_power, "check_once", lambda: None)    # this machine's own checks: not what is tested
    monkeypatch.setattr(window, "check", lambda: None)
    with serving(Server(says_who=False)) as url, connect(url, ping_path=None) as link:
        with pytest.warns(UserWarning, match="does not say it is sequence-ai's serve"):
            link.act(observation())
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            link.act(observation())                      # told once
    with serving(Server()) as url, connect(url) as link, warnings.catch_warnings():
        warnings.simplefilter("error")
        link.wait_until_ready()
        link.act(observation())
        assert link.server == {"server": "sequence-ai", "version": "0.63.0", "up_s": 5.0, "pid": 4242, "policy": "toy"}


def test_sequence_ai_connect_to_an_address_sends_the_token_given_and_never_an_api_key(monkeypatch):
    import sequence_ai
    from sequence_ai.errors import InvalidRequest

    monkeypatch.setenv("SEQUENCES_API_KEY", "the-platform-key")
    server = Server(token="s3cret")
    with serving(server) as url:
        with sequence_ai.connect(url, api_key="the-account-key") as link:      # an api_key: never sent to an address
            with pytest.raises(httpx.HTTPStatusError):
                link.act(observation())
        with sequence_ai.connect(url, token="s3cret") as link:
            assert link.act(observation())["image"].shape == (240, 320, 3)
    assert server.auth[0] is None and server.auth[-1] == b"Bearer s3cret"
    sent = b"".join(a for a in server.auth if a)
    assert b"the-platform-key" not in sent and b"the-account-key" not in sent
    with pytest.raises(InvalidRequest, match="token= is a policy server of your own's"):
        sequence_ai.connect("my_policy_user", token="s3cret")


def test_pillow_and_its_jpeg_encoder_are_loaded_once_when_the_server_first_says_it_reads_jpeg(monkeypatch):
    calls = []
    monkeypatch.setattr(frames, "warm", lambda: calls.append(1))
    with serving(Server(reads="zlib")) as url, connect(url, check=False) as link:
        link.wait_until_ready()
        assert calls == []                                       # no JPEG: nothing to load
    with serving(Server()) as url, connect(url, check=False) as link:
        link.wait_until_ready()
        link.act(observation())
        link.act(observation())
    deadline = time.monotonic() + 5
    while not calls and time.monotonic() < deadline:
        time.sleep(0.01)
    assert calls == [1]
