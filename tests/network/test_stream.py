# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
A robot's stream to serve() on a real port, through the platform's own client: sequence_ai.connect(url) gives the same
Policy a policy's name gives. Each observation goes as it is made and the act after it names how many came before
it; the policy is handed the newest byte for byte; a Video item comes back close to what was sent; two robots at once
are kept apart; a picture not marked goes as a video keyframe of its own, every byte of it with pictures="exact";
an observation the server lost is sent again, one it refused is left out and the video's next frame is a
keyframe; a stream that broke is opened again with nothing lost; act() takes no observation, as the platform's
workers do; the robots' steps read from GET /record into a file whose video plays, frame for frame; and a robot's
stream over two lines carries each act on both, the first answer used, a line that broke opening again while the other
carries the acts.

    python -m pytest -q tests/network/test_stream.py
"""
from __future__ import annotations

import contextlib
import hashlib
import sys
import threading
import time

import numpy as np
import pytest

pytest.importorskip("uvicorn")
pytest.importorskip("httpx")
pytest.importorskip("sequence_base")

import sequence_ai  # noqa: E402
from sequence_ai.engine import serve as serving  # noqa: E402
from sequence_ai.engine import stream as stream_module  # noqa: E402
from sequence_ai.network import record  # noqa: E402


def _policy(observation):
    """What it was handed, by key: each item's SHA-256, and the frames themselves where they are pictures."""
    out = {}
    for key, value in observation.items():
        a = np.asarray(value)
        out[key] = {"sha": hashlib.sha256(a.tobytes()).hexdigest(), "shape": list(a.shape)}
        if a.ndim == 3:
            out[key]["frame"] = a
    return out


@contextlib.contextmanager
def _serving(app):
    server = serving._uvicorn(app, "warning")
    sockets = serving._listen("127.0.0.1", 0)
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
        app.close()


def _sha(a) -> str:
    return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()


def _scene(k: int, h: int = 96, w: int = 128) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    r = 60 + 0.8 * xx + 2 * k
    g = 128 + 60 * np.sin((xx + 2 * k) / 9.0) * np.cos(yy / 13.0)
    b = 40 + 190 * np.clip((350 - (xx - 40 - k) ** 2 - (yy - 0.5 * h) ** 2) / 200.0, 0, 1)
    return np.clip(np.stack([r, g, b], -1), 0, 255).astype(np.uint8)


def _psnr(a, b) -> float:
    mse = float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def test_connect_gives_the_platforms_policy_on_the_servers_address():
    with _serving(serving.App(_policy)) as url:
        policy = sequence_ai.connect(url)
        assert isinstance(policy, sequence_ai.Policy)
        state = np.arange(3, dtype=np.float32)
        policy.observe({"state": state})
        assert policy.act()["state"]["sha"] == _sha(state)
        policy.close()


def test_observations_stream_byte_for_byte_and_the_policy_reads_the_newest():
    with _serving(serving.App(_policy)) as url:
        policy = sequence_ai.connect(url)
        rng = np.random.default_rng(0)
        for _ in range(6):
            state = rng.standard_normal(7).astype(np.float32)
            image = rng.integers(0, 255, (24, 32, 3), dtype=np.uint8)
            policy.observe({"state": state, "image": image})
            got = policy.act()
            assert got["state"]["sha"] == _sha(state) and got["image"]["sha"] == _sha(image)
        policy.close()


def test_a_video_item_comes_back_close_to_what_was_sent():
    pytest.importorskip("av")
    with _serving(serving.App(_policy)) as url:
        policy = sequence_ai.connect(url)
        worst = 99.0
        for k in range(10):
            policy.observe({"cam": sequence_ai.Video(_scene(k)), "step": np.int64(k)})
            worst = min(worst, _psnr(policy.act()["cam"]["frame"], _scene(k)))
        policy.close()
    assert worst > 30.0


def test_an_unmarked_picture_goes_as_a_keyframe_of_its_own_and_comes_back_close():
    pytest.importorskip("av")
    with _serving(serving.App(_policy)) as url:
        policy = sequence_ai.connect(url)
        worst, exact = 99.0, 0
        for k in range(5):
            picture = _scene(k)
            policy.observe({"cam": picture})
            got = policy.act()["cam"]
            worst, exact = min(worst, _psnr(got["frame"], picture)), exact + (got["sha"] == _sha(picture))
        policy.close()
    assert worst > 25.0 and exact == 0


def test_pictures_exact_sends_each_picture_byte_for_byte():
    with _serving(serving.App(_policy)) as url:
        policy = sequence_ai.connect(url, pictures="exact")
        for k in range(3):
            picture = _scene(k)
            policy.observe({"cam": picture})
            assert policy.act()["cam"]["sha"] == _sha(picture)
        policy.close()


def test_a_policy_that_raises_is_said_with_its_type_and_its_words_and_the_next_act_goes_on():
    def policy(observation):
        if int(observation["k"]) == 1:
            raise ValueError("joint 3 is out of range")
        return int(observation["k"])

    with _serving(serving.App(policy)) as url:
        policy_user = sequence_ai.connect(url)
        policy_user.observe({"k": np.int64(0)})
        assert policy_user.act() == 0
        policy_user.observe({"k": np.int64(1)})
        with pytest.raises(ValueError) as raised:                      # what the policy raised, as itself
            policy_user.act()
        policy_user.observe({"k": np.int64(2)})
        assert policy_user.act() == 2
        policy_user.close()
    assert str(raised.value) == ("the policy raised ValueError: joint 3 is out of range (the server's log has the "
                                 "traceback)")
    assert getattr(raised.value, "remote_traceback", None) is None   # the traceback stays in the server's log


def test_two_robots_at_once_are_each_answered_from_their_own():
    app = serving.App(lambda o: {"who": str(o["who"]), "k": int(o["k"])})
    wrong = []
    with _serving(app) as url:
        policies = [sequence_ai.connect(url) for _ in range(2)]

        def robot(i):
            for k in range(20):
                policies[i].observe({"who": f"robot{i}", "k": np.int64(k)})
                got = policies[i].act()
                if got != {"who": f"robot{i}", "k": k}:
                    wrong.append((i, k, got))

        threads = [threading.Thread(target=robot, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)
        for policy in policies:
            policy.close()
    assert not wrong


def test_an_observation_the_server_lost_is_sent_again_and_the_act_reads_the_newest():
    app = serving.App(lambda o: int(o["k"]))
    app._plane.store = stream_module.StreamStore(refuse_gap_early=True)
    put, lost = app._plane.store.put, []

    def losing(episode_id, frames, skipped, size, **kw):
        if not lost and any(s == 3 for s, _ in frames):
            lost.append([s for s, _ in frames])
            return 2                                       # answered as taken, kept nowhere
        return put(episode_id, frames, skipped, size, **kw)

    app._plane.store.put = losing
    with _serving(app) as url:
        policy = sequence_ai.connect(url)
        answers = []
        for k in range(6):
            policy.observe({"k": np.int64(k)})
            answers.append(policy.act())
        policy.close()
    assert lost and answers == list(range(6))


def test_a_refused_observation_is_left_out_and_the_videos_next_frame_is_a_keyframe():
    pytest.importorskip("av")
    app = serving.App(lambda o: int(o["k"]))
    put, refused, keyframes = app._plane.store.put, [], []

    def refusing(episode_id, frames, skipped, size, **kw):
        for s, inputs in frames:
            cam = inputs.get("cam") if isinstance(inputs, dict) else None
            if cam is not None and cam.get("kf"):
                keyframes.append(s)
        if not refused and any(s == 4 for s, _ in frames):
            refused.append(4)
            raise stream_module.StreamError("not kept")
        return put(episode_id, frames, skipped, size, **kw)

    app._plane.store.put = refusing
    with _serving(app) as url:
        policy = sequence_ai.connect(url)
        got = []
        for k in range(10):
            policy.observe({"cam": sequence_ai.Video(_scene(k)), "k": np.int64(k)})
            got.append(policy.act())
        policy.close()
    assert refused and keyframes[0] == 0 and any(5 <= s <= 6 for s in keyframes[1:])
    assert got[-1] == 9 and 4 not in got


def test_a_stream_that_broke_is_opened_again_and_nothing_is_lost():
    with _serving(serving.App(lambda o: int(o["k"]))) as url:
        policy = sequence_ai.connect(url)
        answers = []
        for k in range(8):
            policy.observe({"k": np.int64(k)})
            if k == 3:
                policy._t._worker._stream._sock.shutdown(2)          # the connection gone under the robot's feet
            answers.append(policy.act())
        policy.close()
    assert answers == list(range(8))


def test_act_takes_no_observation_and_observe_then_act_answers():
    with _serving(serving.App(_policy)) as url:
        policy = sequence_ai.connect(url)
        with pytest.raises(TypeError):
            policy.act({"state": np.zeros(4, dtype=np.float32)})
        state = np.arange(4, dtype=np.float32)
        policy.observe({"state": state})
        assert policy.act()["state"]["sha"] == _sha(state)
        policy.close()


def test_the_robots_steps_are_read_from_the_record_port_and_the_video_plays_frame_for_frame(tmp_path):
    av = pytest.importorskip("av")
    path = str(tmp_path / "run.rec")
    with _serving(serving.App(lambda o: int(o["k"]))) as url:
        saved = {}
        recorder = threading.Thread(target=lambda: saved.setdefault("n", record.save(url, path, seconds=5.0)))
        recorder.start()
        time.sleep(0.3)
        policy = sequence_ai.connect(url)
        for k in range(12):
            policy.observe({"cam": sequence_ai.Video(_scene(k)), "k": np.int64(k)})
            policy.act()
        policy.close()
        recorder.join(15)
    kinds = [item["t"] for item in record.items(path)]
    assert kinds.count("observe") == 12 and kinds.count("act") == 12
    mp4 = str(tmp_path / "cam.mp4")
    assert record.video(path, "cam", mp4) == 12
    with av.open(mp4) as played:
        assert sum(1 for _ in played.decode(video=0)) == 12


# a robot's lines, bound to this machine's loopback interface: two lines to a server on this machine
_LOOPBACK = "lo0" if sys.platform == "darwin" else "lo"
_bound = pytest.mark.skipif(sys.platform != "darwin" and not sys.platform.startswith("linux"),
                            reason="a line is bound to its interface on Linux or macOS")


@_bound
def test_two_lines_each_carry_the_stream_and_each_act_takes_the_first_answer():
    with _serving(serving.App(lambda o: int(o["k"]))) as url:
        policy = sequence_ai.connect(url, lines=[_LOOPBACK, _LOOPBACK])
        answers = []
        for k in range(10):
            policy.observe({"k": np.int64(k)})
            answers.append(policy.act())
        each = policy.stats()["lines"]
        policy.close()
    assert answers == list(range(10))
    assert [line["line"] for line in each] == [_LOOPBACK, _LOOPBACK] and all(line["open"] for line in each)
    assert sum(line["first"] for line in each) == 10 and all(line["acts"] >= 1 for line in each)


@_bound
def test_an_act_both_lines_carry_runs_once_on_a_policy_that_keeps_state():
    """Each act goes on every open line under its one request id: a policy that counts its calls runs each act once,
    and the robot is handed that one answer, in order."""
    calls = []

    def counting(observation):
        calls.append(int(observation["k"]))
        return len(calls)

    with _serving(serving.App(counting)) as url:
        policy = sequence_ai.connect(url, lines=[_LOOPBACK, _LOOPBACK])
        answers = []
        for k in range(10):
            policy.observe({"k": np.int64(k)})
            answers.append(policy.act())
        policy.close()
    assert calls == list(range(10)) and answers == list(range(1, 11))


@_bound
def test_a_line_that_broke_leaves_the_acts_to_the_other_and_opens_again():
    with _serving(serving.App(lambda o: int(o["k"]))) as url:
        policy = sequence_ai.connect(url, lines=[_LOOPBACK, _LOOPBACK])
        answers, broken = [], None
        for k in range(8):
            policy.observe({"k": np.int64(k)})
            if k == 3:
                line = policy._t._worker._stream._lines[0]
                broken = line.stream
                broken._sock.shutdown(2)                            # one line's connection gone under the robot's feet
            answers.append(policy.act())
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and (line.stream is broken or not line.alive):
            time.sleep(0.05)
        reopened = line.stream is not broken and line.alive
        policy.close()
    assert answers == list(range(8)) and reopened

