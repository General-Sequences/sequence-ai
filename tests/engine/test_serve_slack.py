# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
serve() holding and seating each robot by the slack its chunk leaves (sequence_ai/engine/serve.py with engine/slack.py):
how a robot plays an answer, given or refused, and an answer with fewer steps than that warned of; the warm-up timing
each batch size as slack.measure does, and /ping saying how robots play, the robots seated, the seats and the batch
times; a robot seated at its first act - each act the platform's, its observation posted first - one that would be late
beside the seated refused with why at each act, and none seated by putting a seated one out; robots that report no round
trip seated while seats are left; an act that names no robot served, unseated; a robot gone giving its seat up; an act
held for a robot due before it must go, and the two run as one batch; the platform's own Policy (connect(url)) naming
its robot on every act, saying its round trip and what is left of the answer it plays (none of a pause), and raising
serve()'s words for a robot with no seat; and four robots in front of serve() (examples/fleet_on_serve.py): three of
them seated and answered, the fourth refused with why.

    python -m pytest -q tests/engine/test_serve_slack.py
"""
from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
import pathlib
import threading
import time

import pytest

from sequence_ai.engine import serve as serving
from sequence_ai.engine import slack
from sequence_ai.engine.serve import NOT_SEATED, App
from sequence_ai.network import msgpack_io

pytest.importorskip("sequence_base")         # serve() takes each robot's stream, which needs sequence-base

TIMES = {1: 0.080, 2: 0.110}                     # the stand-in's batches, in seconds
PLAY = dict(hz=20, action_chunk_valid_len=15, action_chunk_replan_at=8)  # a 350 ms buffer, an act every 400 ms
OBSERVATION = {"state": [0.0] * 7}


def stand_in(observations: list) -> list:
    time.sleep(TIMES[len(observations)])
    return [{"actions": [[0.0] * 7] * 15} for _ in observations]


def _app(**options) -> App:
    return App(stand_in, batch=sorted(TIMES), example=OBSERVATION, **{**PLAY, **options})


def _run(coroutine, timeout: float = 30.0):
    return asyncio.run(asyncio.wait_for(coroutine, timeout))


async def _request(app: App, method: str, path: str, body: bytes = b"", *, headers=()):
    """One request through the ASGI app: (status, headers, body)."""
    parts, sent = [body], []

    async def receive():
        if parts:
            return {"type": "http.request", "body": parts.pop(0), "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    given = [(k.lower().encode(), v.encode()) for k, v in headers] + [(b"content-length", str(len(body)).encode())]
    await app({"type": "http", "method": method, "path": path, "headers": given}, receive, send)
    start, *rest = sent
    return (start["status"], {k.decode(): v.decode() for k, v in start["headers"]},
            b"".join(m.get("body", b"") for m in rest))


_SENT: dict = {}                                 # (app, episode): the observations posted, each next act's seq


async def _act(app: App, robot=None, outside=None, left=None):
    """An act as the platform's client sends it to serve(): its episode's next observation posted (POST /observe),
    then the act on it (POST /act), naming its robot, its round trip and what is left of its chunk, each if given -
    the act's (status, headers, body). A robot keeps one episode; an act that names none, an episode of its own."""
    pytest.importorskip("sequence_base")
    headers = [(name, str(value)) for name, value in (("X-Seq-Session", robot), ("X-Seq-Outside-Ms", outside),
                                                      ("X-Seq-Chunk-Left-Ms", left)) if value is not None]
    episode = f"episode-{robot}" if robot is not None else f"episode-{len(_SENT)}-{time.perf_counter_ns()}"
    seq = _SENT.get((id(app), episode), 0)
    _SENT[(id(app), episode)] = seq + 1
    post = json.dumps({"episode_id": episode, "frames": [{"seq": seq, "inputs": OBSERVATION}]}).encode()
    status, _, said = await _request(app, "POST", "/observe", post, headers=[("Content-Type", "application/json")])
    assert status == 200, said
    call = json.dumps({"call": "act", "episode_id": episode, "seq": seq + 1}).encode()
    return await _request(app, "POST", "/act", call, headers=headers + [("Content-Type", "application/json"),
                                                                      ("Accept", msgpack_io.MEDIA_TYPE)])


def _actions(body: bytes) -> list:
    """The actions of the platform's answer to an act: {"outputs": what the policy returned, "perf"}."""
    return msgpack_io.unpack(body)["outputs"]["actions"]


def _refusal(answer: tuple) -> str:
    """Why an act was refused, as the platform's refusal (ErrorBody) says it: unavailable, its reason not_seated."""
    status, headers, body = answer
    said = json.loads(body)
    assert status == NOT_SEATED and headers["content-type"].startswith("application/json")
    assert said["code"] == "unavailable" and said["details"] == {"reason": "not_seated"}
    return said["detail"]


async def _ready(app: App) -> dict:
    """The warm-up started and waited for; /ping's JSON then."""
    await _request(app, "GET", "/ping")
    for _ in range(1000):
        if app.ready:
            break
        await asyncio.sleep(0.01)
    return json.loads((await _request(app, "GET", "/ping"))[2])


# ── how a robot plays an answer ───────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("given", [dict(hz=0, action_chunk_valid_len=15), dict(action_chunk_valid_len=15), dict(hz=20),
                                   dict(hz=20, action_chunk_valid_len=0), dict(hz=20, action_chunk_replan_at=1.5),
                                   dict(hz=20, action_chunk_valid_len=15, max_wait_for_batch=120),
                                   dict(hz=2000, action_chunk_valid_len=15)])
def test_numbers_that_do_not_say_how_a_robot_plays_an_answer_are_refused_with_what_each_is(given):
    with pytest.raises(ValueError, match="hz, the steps it plays a second"):
        App(stand_in, batch=[1, 2], example=OBSERVATION, **given)


def test_how_a_robot_plays_an_answer_needs_an_example_to_time_the_batches_by():
    with pytest.raises(ValueError, match="pass example="):
        App(stand_in, batch=[1, 2], **PLAY)


def test_an_answer_with_fewer_steps_than_a_robot_plays_of_each_is_warned_of_at_the_warm_up(caplog):
    np = pytest.importorskip("numpy")

    async def warmed(answer, **play):
        app = App(lambda observation: answer, example=OBSERVATION, **play)
        try:
            await _ready(app)
            return app.ready
        finally:
            app.close()

    said = []
    for answer, play in (({"actions": [[0.0] * 7] * 5}, dict(hz=20, action_chunk_valid_len=15)),
                         (np.zeros((1, 7)), dict(hz=20, action_chunk_replan_at=8)),
                         ({"actions": np.zeros((15, 7)), "value": 0.5}, dict(hz=20, action_chunk_valid_len=15))):
        caplog.clear()
        with caplog.at_level("WARNING", logger="sequence_ai.serve"):
            assert _run(warmed(answer, **play))                      # ready all the same
        said.append([r.getMessage() for r in caplog.records if r.name == "sequence_ai.serve"])
    assert said == [["the policy answers 5 steps, fewer than the 15 it declares valid (action_chunk_valid_len): its "
                     "robots run out of steps before the next answer comes"],
                    ["the policy answers 1 step, fewer than the 8 it plays before asking for the next "
                     "(action_chunk_replan_at): its robots run out of steps before the next answer comes"],
                    []]


def test_the_warm_up_times_each_size_as_slack_measure_does_and_ping_says_how_robots_play_and_the_seats():
    async def go():
        app = _app()
        try:
            return await _ready(app), app._fleet.table
        finally:
            app.close()

    said, table = _run(go())
    assert said["play"] == {"hz": 20.0, "action_chunk_valid_len": 15, "action_chunk_replan_at": 8}
    assert said["robots"] == 0 and said["seats"] == 2                 # two robots 100 ms away, at most its largest size
    assert table.sizes() == [1, 2] and sorted(table.first_ms) == [1, 2]
    assert table.as_dict()["1"]["n"] == slack.RUNS == table.as_dict()["2"]["n"]
    assert 80 <= said["batch_ms"]["1"] < 120 and 110 <= said["batch_ms"]["2"] < 150


# ── seats ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_a_robot_is_seated_at_its_first_act_and_one_that_would_be_late_beside_it_is_refused_with_why_at_each_act():
    async def go():
        app = _app()
        try:
            await _ready(app)
            return (await _act(app, "near", 20), await _act(app, "far", 200), await _act(app, "far", 200),
                    json.loads((await _request(app, "GET", "/ping"))[2]))
        finally:
            app.close()

    near, far, again, said = _run(go())
    assert near[0] == 200 and _actions(near[2])
    words = _refusal(far)
    assert _refusal(again) == words
    assert words.startswith("this server cannot answer this robot within the 350 ms left of each answer as a robot "
                            "asks for the next (7 of 15 steps at 20 Hz) beside the 1 robot it serves, however their "
                            "requests fall, with this robot at the round trip it reports (a batch takes 1: ")
    assert " ms away" not in words                                    # no robot's round trip as a number
    assert words.endswith("Serve it from a GPU nearer it, or from this one once the others have stopped: a robot gives "
                          "its place up 3 s after its last act")
    assert said["robots"] == 1


def test_no_robot_is_seated_by_putting_a_seated_one_out():
    async def go():
        app = _app()
        try:
            await _ready(app)
            return await _act(app, "far", 200), await _act(app, "near", 20)
        finally:
            app.close()

    far, near = _run(go())
    assert far[0] == 200                                              # an empty server seats a robot, however far
    assert ("seated beside the 1 robot it serves, this robot, at the round trip it reports, would leave one of them "
            "without its answers within the 350 ms left") in _refusal(near)
    assert " ms away" not in _refusal(near)


def test_robots_that_report_no_round_trip_are_seated_as_robots_at_the_default_and_no_more_than_a_batch_holds():
    def quick(observations: list) -> list:
        time.sleep(0.01 * len(observations))
        return [{"actions": [[0.0] * 7] * 15} for _ in observations]

    async def go():
        app = App(quick, batch=[1, 2], example=OBSERVATION, **PLAY)
        try:
            await _ready(app)
            return [await _act(app, robot) for robot in ("a", "b", "c")]
        finally:
            app.close()

    a, b, c = _run(go())
    assert a[0] == 200 == b[0]
    assert _refusal(c).startswith("this server already serves 2 robots, as many as its largest batch holds (2)")


def test_the_seats_and_a_robot_that_reports_no_round_trip_are_said_at_a_default_round_trip_with_no_number(caplog):
    """The default round trip is what robots are counted at until they report their own: the seats line at the warm-up
    and a refusal of a robot that reported none say it as a default, never as slack.ROUND_TRIP_MS's number. One act
    at a time takes 150 ms here: a robot at the default is answered in time alone, and a second, asking as the first
    one's act runs, is not."""
    def slow(observations: list) -> list:
        time.sleep({1: 0.150, 2: 0.250}[len(observations)])
        return [{"actions": [[0.0] * 7] * 15} for _ in observations]

    async def go():
        app = App(slow, batch=[1, 2], example=OBSERVATION, **PLAY)
        try:
            await _ready(app)
            return await _act(app, "a"), await _act(app, "b")
        finally:
            app.close()

    with caplog.at_level("INFO", logger="sequence_ai.serve"):
        a, b = _run(go())
    said = [r.getMessage() for r in caplog.records if r.name == "sequence_ai.serve"]
    seats = [line for line in said if line.startswith("seats ")]
    assert seats == ["seats 1 robot at a default round trip, then each robot by the round trip it reports"]
    assert a[0] == 200
    words = _refusal(b)
    assert ("however their requests fall, with this robot at a default round trip, having reported none (a batch "
            "takes 1: ") in words
    assert " ms away" not in words and f"{slack.ROUND_TRIP_MS:.0f} ms" not in words


def test_an_act_that_names_no_robot_is_served_and_seats_nobody():
    async def go():
        app = _app()
        try:
            await _ready(app)
            return await _act(app), json.loads((await _request(app, "GET", "/ping"))[2])
        finally:
            app.close()

    (status, _, body), said = _run(go())
    assert status == 200 and _actions(body) and said["robots"] == 0


def test_a_robot_that_stopped_acting_gives_its_seat_up():
    async def go():
        app = _app()
        try:
            await _ready(app)
            first = await _act(app, "near", 20)
            app._fleet.gone_s = 0.05                                  # in place of its 4 chunks' 3 s
            await asyncio.sleep(0.1)
            far = await _act(app, "far", 200)
            app._fleet.gone_s = 60.0                                  # the far one stays, however slow the machine
            return first, far, await _act(app, "near", 20)
        finally:
            app.close()

    first, far, near_again = _run(go())
    assert first[0] == 200 and far[0] == 200                          # the near robot's seat went to the far one
    assert _refusal(near_again)                                       # ... which it would now put out


def test_without_a_robots_time_a_request_waits_for_no_other_and_those_that_waited_for_the_policy_go_together():
    """Told nothing of how a robot plays an answer, serve() holds no request for a batch: the first, with the policy
    free, runs alone at once; the two that come while it runs wait for the policy, and go together after it."""
    seen: list = []

    def policy(observations):
        seen.append(len(observations))
        time.sleep(0.05)
        return [{"actions": [[0.0] * 7]} for _ in observations]

    async def go():
        app = App(policy, batch=[1, 2, 4])
        try:
            first = asyncio.ensure_future(_act(app))
            await asyncio.sleep(0.001)
            rest = [asyncio.ensure_future(_act(app)) for _ in range(2)]
            return [status for status, _, _ in await asyncio.gather(first, *rest)]
        finally:
            app.close()

    assert _run(go()) == [200, 200, 200] and seen == [1, 2]


# ── an act that is not the call alone ─────────────────────────────────────────────────────────────────────────────────
def test_an_act_that_is_not_the_call_alone_is_refused_with_the_reason_the_platforms_worker_gives():
    """sequence_base.contract.dataplane's reasons, as the platform's worker answers them: a body that is no object
    "body"; a call other than "act", or an act that carries data, "call"."""
    async def go():
        app = _app()
        try:
            await _ready(app)
            said = []
            for body in (b"[1, 2]", json.dumps({"call": "jump"}).encode(),
                         json.dumps({"call": "act", "episode_id": "e", "seq": 1, "inputs": OBSERVATION}).encode()):
                status, _, answer = await _request(app, "POST", "/act", body,
                                                   headers=[("Content-Type", "application/json")])
                said.append((status, json.loads(answer)["details"]["reason"]))
            return said
        finally:
            app.close()

    assert _run(go()) == [(400, "body"), (400, "call"), (400, "call")]


# ── each robot's time ─────────────────────────────────────────────────────────────────────────────────────────────────
def test_an_act_waits_for_a_robot_due_before_it_must_go_and_the_two_run_as_one_batch():
    async def go():
        # a 1.6 s buffer (32 steps at 20 Hz): room for a's act to wait for b's however a loaded machine stretched the
        # batches the warm-up timed - with 350 ms, a batch timed past 300 ms leaves a nothing to wait with
        app = _app(action_chunk_valid_len=40)
        try:
            await _ready(app)
            await asyncio.gather(_act(app, "a", 20), _act(app, "b", 20))     # first acts: nothing playing, not held
            sent = dict(app._batcher.sent)
            await asyncio.sleep(0.4 - 0.2)                                    # a round later: a's act, b's 30 ms on
            began = time.perf_counter()

            async def later():
                await asyncio.sleep(0.03)
                return await _act(app, "b", 20)

            a_answer = asyncio.ensure_future(_act(app, "a", 20))
            await later()
            await a_answer
            return sent, dict(app._batcher.sent), time.perf_counter() - began
        finally:
            app.close()

    before, after, took = _run(go())
    assert after.get(2, 0) == before.get(2, 0) + 1                     # a's act waited for b's: one batch of two
    assert took >= 0.03 + TIMES[2]


# ── the robot's side: the platform's own Policy ─────────────────────────────────────────────────────────────────────
@contextlib.contextmanager
def _served(app: App):
    """`app` on uvicorn as serve() runs it - a robot's stream on its WebSocket - on a free port of this machine; its
    URL. The app is closed after."""
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


def _reads(app: App) -> list:
    """What `app` reads of each act's robot from now on, in order: (its name, its round trip in ms, what is left of
    the answer it plays in ms), as App._said reads them."""
    said, read = [], app._said
    app._said = lambda scope: said.append(read(scope)) or said[-1]
    return said


def _observation():
    np = pytest.importorskip("numpy")
    pytest.importorskip("sequence_base")
    return {"state": np.zeros(7, np.float32)}


def test_the_policy_names_its_robot_on_each_act_and_says_its_round_trip_and_what_is_left_of_its_answer():
    """connect(url) - the platform's own Policy - on serve(hz=...): every act names the same robot (X-Seq-Session), says
    its round trip once it has one (X-Seq-Outside-Ms) and, from its second act on, what is left of the answer it plays
    (X-Seq-Chunk-Left-Ms), by the hz and action_chunk_valid_len serve()'s /ping says, which its connect hands it."""
    import sequence_ai

    observation, app = _observation(), _app()
    said = _reads(app)
    with _served(app) as url, sequence_ai.connect(url) as policy:
        policy.wait_until_ready(30)
        first = time.perf_counter()
        policy.observe(observation)
        policy.act()
        time.sleep(0.2)
        policy.observe(observation)
        policy.act()
        since = time.perf_counter() - first
    (robot, _, left), (again, outside, then) = said
    assert robot and robot == again and left is None
    assert 750 - since * 1000 - 50 <= then <= 750 - 200       # its 750 ms, from when its first act was sent
    assert outside is None or 0 <= outside <= 5000


def test_a_robot_back_from_a_pause_says_none_of_the_pause_is_left():
    """A robot that stood past the end of its answer before it asked again (a pause between episodes): the act that
    ends the pause says 0 is left, and the act after it its 750 ms less the time since that act was sent - nothing of
    the pause on top, which would have its server hold its act past what the robot has."""
    import sequence_ai

    observation, app = _observation(), _app()
    said = _reads(app)
    with _served(app) as url, sequence_ai.connect(url) as policy:
        policy.wait_until_ready(30)
        policy.observe(observation)
        policy.act()
        time.sleep(1.0)                                       # past its 750 ms of steps: the robot stands
        sent = time.perf_counter()
        policy.observe(observation)
        policy.act()
        time.sleep(0.2)
        policy.observe(observation)
        policy.act()
        since = time.perf_counter() - sent
    assert said[1][2] == 0
    assert 750 - since * 1000 - 50 <= said[2][2] <= 750 - 200


def test_a_robot_with_no_seat_is_told_why_by_its_policys_act():
    """A robot serve() cannot seat has its act refused unavailable, reason not_seated, with why - which its Policy's
    act() raises as it is, trying no other worker: here a batch holds one robot, and the second robot is told so."""
    import sequence_ai
    from sequence_ai.errors import Unavailable

    observation, app = _observation(), App(stand_in, batch=[1], example=OBSERVATION, **PLAY)
    with _served(app) as url, sequence_ai.connect(url) as first, sequence_ai.connect(url) as second:
        first.wait_until_ready(30)
        first.observe(observation)
        first.act()
        with pytest.raises(Unavailable, match=r"this server already serves 1 robot, as many as its largest batch "
                                              r"holds \(1\)"):
            second.observe(observation)
            second.act()


# ── four robots in front of serve() ───────────────────────────────────────────────────────────────────────────────────
def test_four_robots_three_seated_and_answered_and_the_far_one_refused_with_why():
    """Four robots through proxies on this machine in front of serve(hz=...): those 20, 20 and 100 ms away seated and
    answered, the one 300 ms away refused with why. Whether their answers come in time as slack's clock says is what
    examples/fleet_on_serve.py measures, on a machine quiet enough to tell: each act waits for others up to MARGIN_MS
    before its deadline, so on a machine loaded past its cores (a load of 29 on 10) its threads and timers run late and
    so do its answers, whatever the scheduling (14 of 48 late there, 0 on the clock)."""
    pytest.importorskip("httpx")
    path = pathlib.Path(__file__).resolve().parents[2] / "examples" / "fleet_on_serve.py"
    spec = importlib.util.spec_from_file_location("fleet_on_serve", path)
    fleet = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fleet)

    # quick batches beside the 350 ms buffer, and the far robot past it even alone (40 + 300 + 50 ms): what is seated
    # does not move with the load of the machine the test runs on
    robots_ms = [("near-a", 20.0), ("near-b", 20.0), ("mid", 100.0), ("far", 300.0)]
    out = fleet.run(seconds=10.0, robots=robots_ms, times={1: 40.0, 2: 50.0, 4: 60.0})
    robots, clock = out["robots"], out["clock"]
    # refused with why: late itself, or - where the machine's load stretched the batches - putting a seated robot out
    refused = robots["far"]["refused"] or ""
    assert ("cannot answer this robot within the 350 ms left" in refused
            or "would leave one of them without its answers within the 350 ms left" in refused)
    seated = [robot for name, robot in robots.items() if name != "far"]
    batch_ms = out["server"]["batch_ms"]
    if any(robot["refused"] for robot in seated) and max(batch_ms.values(), default=0.0) > 100.0:
        # the 40 to 60 ms batches timed past 100 ms: a load that leaves even a robot 20 ms away no answer within its
        # 350 ms at its worst start, refused rightly - what is seated is then the load's, not the scheduling's
        pytest.skip(f"this machine's load stretched the batches serve() timed to {batch_ms} ms")
    assert all(robot["refused"] is None and robot["answers"] >= 10 for robot in seated), robots
    assert clock["late"] == 0 and clock["answers"] > 0              # the clock has them in time, from these starts
    # the three seated; the seats - robots at the default round trip, further than two of these - at least one, however
    # a loaded machine stretched the batches it timed
    assert out["server"]["robots"] == 3 and out["server"]["seats"] >= 1
