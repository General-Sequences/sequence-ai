# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
serve() as several processes, one model across them (sequence_ai/engine/ranks.py): the group a launcher's environment
names (WORLD_SIZE, RANK, MASTER_ADDR, MASTER_PORT), or none; each call reaching every rank before rank 0 runs its own,
and rank 0 answering with its own; a rank that raised where rank 0 did not, or left, halting the group, where a call
that raised on every rank is the policy's own error; and, in processes of their own as torchrun starts them, a
classifier-free-guidance policy split across two ranks over gloo answering a robot bit for bit as one process does,
every rank running every call of a policy of numpy alone, and an act after a rank died refused with why.

    python -m pytest -q tests/engine/test_serve_ranks.py
"""
from __future__ import annotations

import os
import pathlib
import signal
import socket
import subprocess
import sys
import threading
import time

import pytest

from sequence_ai.engine import ranks

HERE = pathlib.Path(__file__).resolve().parent
SERVER = HERE / "_ranks_server.py"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# ── the group a launcher names ────────────────────────────────────────────────────────────────────────────────────────
def test_a_process_alone_is_no_group_and_a_launchers_names_its_rank_and_rank_0():
    assert ranks.from_env({}) is None and ranks.from_env({"WORLD_SIZE": "1", "RANK": "0"}) is None
    group = ranks.from_env({"WORLD_SIZE": "2", "RANK": "1", "MASTER_ADDR": "10.0.0.7", "MASTER_PORT": "29500"})
    assert (group.rank, group.size, group.host, group.port, group.leads) == (1, 2, "10.0.0.7", 29501, False)
    assert ranks.from_env({"WORLD_SIZE": "2", "RANK": "0", "MASTER_PORT": "29500",
                           ranks.RANKS_PORT_ENV: "31000"}).port == 31000
    for bad in ({"WORLD_SIZE": "2", "MASTER_PORT": "29500"}, {"WORLD_SIZE": "2", "RANK": "0"},
                {"WORLD_SIZE": "2", "RANK": "2", "MASTER_PORT": "29500"}, {"WORLD_SIZE": "two", "RANK": "0"}):
        with pytest.raises(ValueError):
            ranks.from_env(bad)


# ── a group in this process, its ranks on threads ─────────────────────────────────────────────────────────────────────
def _pair(run_one):
    """Rank 0 of two, its rank 1 following on a thread with `run_one`: (rank 0, rank 1's thread)."""
    port = _free_port()
    leader, follower = ranks.Group(0, 2, "127.0.0.1", port, join_s=10), ranks.Group(1, 2, "127.0.0.1", port, join_s=10)
    leader.listen()
    thread = threading.Thread(target=follower.follow, args=(run_one,), daemon=True)
    thread.start()
    leader.wait_joined()
    return leader, thread


def test_each_call_reaches_every_rank_before_rank_0_runs_it_and_rank_0_answers_with_its_own():
    """Rank 0's call waits for rank 1's, as a collective does: it returns only because rank 1 had the call first."""
    seen, entered = [], threading.Event()

    def run_one(op, args):
        seen.append((op, args))
        entered.set()

    leader, thread = _pair(run_one)

    def here():
        assert entered.wait(5), "rank 1 never had the call: rank 0 ran it alone"
        return "rank 0's"

    assert leader.run("call", [[{"state": [1.0, 2.0]}]], here) == "rank 0's"
    assert seen == [("call", [[{"state": [1.0, 2.0]}]])] and leader.halted is None
    leader.close()
    thread.join(5)
    assert not thread.is_alive()                                     # told the server stopped, rank 1 returned


def test_a_rank_that_raised_where_rank_0_did_not_halts_the_group_and_a_call_every_rank_raised_on_is_the_policys():
    def run_one(op, args):
        if args[0] == "both":
            raise ValueError("the policy refused it")
        if args[0] == "only rank 1":
            raise RuntimeError("CUDA error: out of memory")

    leader, _ = _pair(run_one)

    def raises():
        raise ValueError("the policy refused it")

    with pytest.raises(ValueError, match="the policy refused it"):
        leader.run("call", ["both"], raises)
    assert leader.halted is None                                     # the policy's own error: the group serves on
    with pytest.raises(ranks.GroupHalted, match="rank 1 raised where rank 0 did not: RuntimeError: CUDA error"):
        leader.run("call", ["only rank 1"], lambda: "fine")
    with pytest.raises(ranks.GroupHalted, match="rank 1 raised where rank 0 did not"):
        leader.run("call", ["after"], lambda: "fine")                # halted: no call after it
    assert isinstance(ranks.GroupHalted("x"), ranks.PoolHalted)       # as a pool with no replica left


def test_a_rank_that_left_halts_the_group_saying_which():
    port = _free_port()
    leader = ranks.Group(0, 2, "127.0.0.1", port, join_s=10)
    leader.listen()
    gone = socket.create_connection(("127.0.0.1", port))
    ranks._send(gone, ranks.msgpack_io.pack({"rank": 1}))           # rank 1 joins, then goes
    leader.wait_joined()
    gone.close()
    with pytest.raises(ranks.GroupHalted, match="rank 1 left"):
        leader.run("call", [[1]], lambda: "fine")


def test_a_rank_that_never_joined_is_said_once_the_wait_is_over():
    leader = ranks.Group(0, 3, "127.0.0.1", _free_port(), join_s=0.3)
    leader.listen()
    with pytest.raises(ranks.GroupHalted, match="ranks 1, 2 of 3 did not join rank 0"):
        leader.wait_joined()


# ── processes of their own, as torchrun starts them ───────────────────────────────────────────────────────────────────
class _Servers:
    """The processes of one server, each one's output to a file of its own (never a pipe no one reads)."""

    def __init__(self, tmp: pathlib.Path, mode: str, world: int, **env) -> None:
        self.port, master, rankport = _free_port(), _free_port(), _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.logs, self.procs = [], []
        base = {k: v for k, v in os.environ.items() if k not in ("WORLD_SIZE", "RANK", "MASTER_ADDR", "MASTER_PORT",
                                                                  ranks.RANKS_PORT_ENV, "LOCAL_RANK")}
        for rank in range(world):
            ranked = {} if world == 1 else {"WORLD_SIZE": str(world), "RANK": str(rank), "MASTER_ADDR": "127.0.0.1",
                                            "MASTER_PORT": str(master), ranks.RANKS_PORT_ENV: str(rankport)}
            log = tmp / f"{mode}-{world}-{rank}.log"
            self.logs.append(log)
            with open(log, "w") as out:
                self.procs.append(subprocess.Popen([sys.executable, str(SERVER), mode, str(self.port)],
                                                   env={**base, **ranked, **env}, stdout=out, stderr=subprocess.STDOUT))

    def ready(self, timeout: float = 120.0) -> None:
        import httpx

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for proc, log in zip(self.procs, self.logs):
                assert proc.poll() is None, f"a rank exited: {log.read_text()[-2000:]}"
            try:
                if httpx.get(self.url + "/ping", timeout=2).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
        raise AssertionError("no rank 0 ready: " + " | ".join(log.read_text()[-500:] for log in self.logs))

    def stop(self) -> None:
        """Rank 0 first, as Ctrl-C at its launcher would end it: it tells each rank it stopped after every call it sent,
        and each returns once it ran them; any rank still up after that is stopped too."""
        for proc in self.procs:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
            try:
                proc.wait(20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(5)


def _acts(url: str, states: list) -> list:
    """A robot's answers on the platform's own Policy, one act after each observation."""
    np = pytest.importorskip("numpy")
    import sequence_ai

    out = []
    with sequence_ai.connect(url) as policy:
        policy.wait_until_ready(60)
        for state in states:
            policy.observe({"state": np.asarray(state, np.float32)})
            out.append(np.asarray(policy.act()["actions"]))
    return out


def _states():
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(7)
    return [rng.standard_normal(7).astype(np.float32) for _ in range(5)]


def test_a_policy_split_across_two_ranks_answers_bit_for_bit_as_one_process_does(tmp_path):
    """Classifier-free guidance, its conditional branch on rank 0 and its unconditional one on rank 1, each sent to the
    other over gloo - every act answered as the same policy alone answers it, byte for byte."""
    pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    pytest.importorskip("sequence_base")
    states = _states()
    answers = {}
    for world in (1, 2):
        servers = _Servers(tmp_path, "cfg", world)
        try:
            servers.ready()
            answers[world] = _acts(servers.url, states)
        finally:
            servers.stop()
    assert len(answers[2]) == len(states)
    assert all(np.array_equal(a, b) and a.dtype == b.dtype for a, b in zip(answers[1], answers[2]))


def test_every_rank_runs_every_call_of_a_policy_of_numpy_alone(tmp_path):
    """No torch in the policy, none in the engine's dispatch: rank 1 runs each call rank 0 runs - the warm-up's, each
    act's - and the answers are what one process gives."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("sequence_base")
    calls = tmp_path / "calls"
    states = _states()
    servers = _Servers(tmp_path, "plain", 2, SEQ_TEST_CALLS=str(calls))
    try:
        servers.ready()
        got = _acts(servers.url, states)
    finally:
        servers.stop()
    rank0, rank1 = (pathlib.Path(f"{calls}.{rank}").read_text().splitlines() for rank in (0, 1))
    assert rank0 == rank1 and len(rank0) == 2 * 2 + len(states)       # two warm-up runs at each size, then the acts
    assert all(np.array_equal(a, np.tanh(s * 3.0)) for a, s in zip(got, states))


def test_an_act_after_a_rank_died_is_refused_saying_which_rank(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("sequence_base")
    import sequence_ai
    from sequence_ai.errors import Unavailable

    servers = _Servers(tmp_path, "plain", 2)
    try:
        servers.ready()
        with sequence_ai.connect(servers.url) as policy:
            policy.wait_until_ready(60)
            policy.observe({"state": np.zeros(7, np.float32)})
            policy.act()
            servers.procs[1].kill()
            servers.procs[1].wait(10)
            policy.observe({"state": np.zeros(7, np.float32)})
            with pytest.raises(Unavailable, match="this server's ranks serve no more: rank 1 left"):
                policy.act()
    finally:
        servers.stop()


def test_a_policy_class_under_seq_serve_runs_its_observe_function_acts_and_episodes_on_every_rank(tmp_path):
    """`seq serve` of a @seq.policy class as two ranks: each episode's frames reach every rank's observe function, and
    each act and each episode's end every rank, so every rank's copy holds what rank 0's does - and the robot's answers
    are what one process gives."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("sequence_base")
    states = _states()
    answers, calls = {}, {}
    for world in (1, 2):
        written = tmp_path / f"class-calls-{world}"
        servers = _Servers(tmp_path, "class", world, SEQ_TEST_CALLS=str(written),
                           SEQ_TEST_FOLDER=str(tmp_path / f"folders-{world}"))
        try:
            servers.ready()
            answers[world] = _acts(servers.url, states)
        finally:
            servers.stop()
        calls[world] = [pathlib.Path(f"{written}.{rank}").read_text().splitlines() for rank in range(world)]
    assert all(np.array_equal(a, b) for a, b in zip(answers[1], answers[2]))
    rank0, rank1 = calls[2]
    assert rank0 == rank1                                            # every observe, act and episode's end, on both
    acts = [line for line in rank0 if line.startswith("act ") and not line.startswith("act seq-")]
    assert len(acts) == len(states) and [line for line in rank0 if line.startswith("observe ")]


def test_each_rank_of_a_launcher_sees_its_own_card_alone():
    """own_card(): the LOCAL_RANK-th card where CUDA_VISIBLE_DEVICES named some, LOCAL_RANK where none; nothing for one
    process or with no LOCAL_RANK; a rank past the cards named is refused, saying so."""
    from sequence_ai.engine import ranks

    env = {"WORLD_SIZE": "2", "LOCAL_RANK": "1"}
    assert ranks.own_card(env) == "1" and env["CUDA_VISIBLE_DEVICES"] == "1"
    env = {"WORLD_SIZE": "2", "LOCAL_RANK": "1", "CUDA_VISIBLE_DEVICES": "2,3"}
    assert ranks.own_card(env) == "3" and env["CUDA_VISIBLE_DEVICES"] == "3"
    for alone in ({"WORLD_SIZE": "1", "LOCAL_RANK": "0"}, {"WORLD_SIZE": "2", "RANK": "1"}, {}):
        assert ranks.own_card(alone) is None and "CUDA_VISIBLE_DEVICES" not in alone
    with pytest.raises(ValueError, match="LOCAL_RANK 1 has no card of its own"):
        ranks.own_card({"WORLD_SIZE": "2", "LOCAL_RANK": "1", "CUDA_VISIBLE_DEVICES": "0"})
