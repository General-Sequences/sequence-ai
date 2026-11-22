# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
A model's classifier-free guidance split between two processes (sequence_ai/engine/cfg_parallel.py): a call is cut
along the even batch its tensors lead with, and one with none runs whole; in two processes as torchrun starts them,
over gloo, a toy sampler whose denoiser is split gives one process's actions, each process's denoiser called with half
the doubled batch, both processes' actions the same, its dict of outputs gathered back whole, a path that names nothing
said and left, and the model's own forward back once the split is taken back.

    python -m pytest -q tests/engine/test_cfg_parallel.py
"""
from __future__ import annotations

import json
import os
import pathlib
import socket
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from sequence_ai.engine import cfg_parallel  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
RANK = HERE / "_cfg_parallel_rank.py"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_a_call_is_cut_along_the_even_batch_its_tensors_lead_with_and_one_with_none_runs_whole():
    """batch_of: the first dimension of a call's first tensor that has one, where it is even and of two or more."""
    rows = torch.zeros(4, 3)
    assert cfg_parallel.batch_of((rows, torch.zeros(4)), {}) == 4
    assert cfg_parallel.batch_of((torch.tensor(0.5),), {"x": rows}) == 4          # a 0-d tensor leads with nothing
    assert cfg_parallel.batch_of((torch.zeros(3, 2),), {}) is None                 # an odd batch: whole on both
    assert cfg_parallel.batch_of((torch.zeros(1, 2),), {}) is None
    assert cfg_parallel.batch_of(("prompt", 7), {"steps": 4}) is None             # no tensor at all


def test_two_processes_split_a_samplers_denoiser_and_give_one_processs_actions(tmp_path):
    """Each rank runs the sampler whole, split and whole again (tests/engine/_cfg_parallel_rank.py): split, its actions
    are one process's within a millionth, the same on both ranks, from the denoiser called with 3 of the doubled 6 on
    each; the dict it returns is gathered whole (its step too); "model.nothing" is said and not split; uninstall puts
    the model's own forward back. [fails before cfg_parallel.py: no module]"""
    port = _free_port()
    base = {k: v for k, v in os.environ.items() if k not in ("WORLD_SIZE", "RANK", "MASTER_ADDR", "MASTER_PORT")}
    procs = [subprocess.Popen([sys.executable, str(RANK), str(tmp_path)],
                              env={**base, "WORLD_SIZE": "2", "RANK": str(rank), "MASTER_ADDR": "127.0.0.1",
                                   "MASTER_PORT": str(port), "OMP_NUM_THREADS": "1"},
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
             for rank in range(2)]
    try:
        outs = [p.communicate(timeout=120)[0] for p in procs]
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
    assert [p.returncode for p in procs] == [0, 0], outs
    got = [json.loads((tmp_path / f"{rank}.json").read_text()) for rank in range(2)]
    for one in got:
        assert one["seen_alone"] == [6, 6, 6, 6] and one["seen_split"] == [3, 3, 3, 3] and one["seen_again"] == [6] * 4
        assert one["max_abs"] <= 1e-6, one["max_abs"]
        assert one["again"] == one["alone"] and one["restored"] and one["patched"] == ["model"]
        assert one["said"] == ["cfg_parallel: model.nothing is not in this model: not split"]
    assert got[0]["split"] == got[1]["split"] and got[0]["alone"] == got[1]["alone"]


def test_a_group_of_any_size_but_two_is_refused_saying_how_to_start_it(monkeypatch):
    """group(): two processes, or a ValueError naming torchrun's --nproc-per-node 2."""
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 4)
    with pytest.raises(ValueError, match=r"has 4: start the server as 2 \(torchrun --nproc-per-node 2"):
        cfg_parallel.group()


class _Toy(torch.nn.Module):
    """A denoiser called with the doubled batch, and inside it a layer; beside it, a module called with an odd batch."""

    def __init__(self) -> None:
        super().__init__()
        self.inner = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.inner(x)


def test_census_finds_the_outermost_module_called_with_the_doubled_batch_and_none_called_with_an_odd_one():
    """census: of the modules every call of which led with one even batch and answered for it, the outermost that took
    the most time - here the denoiser, not the layer in it, nor the module called with three rows."""
    denoiser, odd = _Toy(), _Toy()

    def act():
        denoiser(torch.zeros(2, 4))
        odd(torch.zeros(3, 4))

    assert cfg_parallel.census({"denoiser": denoiser, "odd": odd}, act) == ["denoiser"]
    assert cfg_parallel.census({"odd": odd}, lambda: odd(torch.zeros(3, 4))) == []


def test_a_trial_keeps_the_split_it_times_faster_and_takes_out_one_without_a_tolerance():
    """trial, its calls and toggles given as serve() gives them on rank 0: the module found split, every run held to
    eager mode's, the act timed both ways and the split kept (min_speedup 0 here); with no tolerance, nothing tried."""
    model = _Toy()

    def policy(observation):
        return {"out": model(torch.cat([torch.as_tensor(observation["x"])] * 2)).detach().numpy()}

    toggled: list = []
    examples = [{"x": [[float(i), 1.0, 2.0, 3.0]]} for i in range(4)]
    report = cfg_parallel.trial(policy, policy, lambda paths, on: toggled.append((paths, on)), examples,
                                limits={"max_abs": 1e-6}, min_speedup=0.0, rounds=1, say=lambda line: None)
    assert report["on"] == ["cfg_parallel"] and report["switches"]["cfg_parallel"]["modules"] == ["model"]
    assert toggled[0] == (["model"], True) and toggled[-1] == (["model"], True)       # left in place: kept
    said: list = []
    off = cfg_parallel.trial(policy, policy, lambda paths, on: None, examples, limits=None, say=said.append)
    assert off["on"] == [] and said == ["optimize: cfg_parallel off - it changes the outputs: give tolerance=, the "
                                        "largest difference you accept from eager mode's"]


SERVER = HERE / "_cfg_parallel_server.py"


class _Servers:
    """The processes of one server (tests/engine/_cfg_parallel_server.py, or `server`), each one's output to a file of
    its own."""

    def __init__(self, tmp: pathlib.Path, world: int, *flags: str, server: pathlib.Path = SERVER) -> None:
        self.port, master, rankport = _free_port(), _free_port(), _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.logs, self.procs = [], []
        base = {k: v for k, v in os.environ.items() if k not in ("WORLD_SIZE", "RANK", "MASTER_ADDR", "MASTER_PORT",
                                                                  "SEQ_SERVE_RANKS_PORT", "LOCAL_RANK")}
        for rank in range(world):
            ranked = {} if world == 1 else {"WORLD_SIZE": str(world), "RANK": str(rank), "MASTER_ADDR": "127.0.0.1",
                                            "MASTER_PORT": str(master), "SEQ_SERVE_RANKS_PORT": str(rankport)}
            log = tmp / f"server-{world}-{rank}.log"
            self.logs.append(log)
            with open(log, "w") as out:
                self.procs.append(subprocess.Popen([sys.executable, str(server), str(self.port), *flags],
                                                   env={**base, **ranked, "OMP_NUM_THREADS": "1"}, stdout=out,
                                                   stderr=subprocess.STDOUT))

    def ready(self, timeout: float = 120.0) -> dict:
        import time

        import httpx

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for proc, log in zip(self.procs, self.logs):
                assert proc.poll() is None, f"a rank exited: {log.read_text()[-3000:]}"
            try:
                got = httpx.get(self.url + "/ping", timeout=2)
                if got.status_code == 200:
                    return got.json()
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
        raise AssertionError("no rank 0 ready: " + " | ".join(log.read_text()[-1500:] for log in self.logs))

    def stop(self) -> None:
        import signal

        for proc in self.procs:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
            try:
                proc.wait(20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(5)


def _answers(url: str, states: list) -> list:
    import numpy as np

    import sequence_ai

    out = []
    with sequence_ai.connect(url) as policy:
        policy.wait_until_ready(60)
        for state in states:
            policy.observe({"state": state})
            out.append(np.asarray(policy.act()["actions"]))
    return out


def test_a_server_of_two_processes_keeps_the_split_and_answers_as_one_process_does(tmp_path):
    """serve(cfg_parallel=True) as two ranks over gloo (tests/engine/_cfg_parallel_server.py, kept: any speed passes):
    its warm-up finds the denoiser, splits it on both ranks and keeps it - /ping says so - and every act is answered
    within a millionth of one process's answer, rank 1's half computed on rank 0's noise; one process alone says it is
    one, and tries nothing. [fails before serve() took cfg_parallel]"""
    pytest.importorskip("sequence_base")
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(3)
    states = [rng.standard_normal(7).astype(np.float32) for _ in range(4)]
    alone = _Servers(tmp_path, 1)
    try:
        ping_alone = alone.ready()
        answers_alone = _answers(alone.url, states)
    finally:
        alone.stop()
    two = _Servers(tmp_path, 2, "kept")
    try:
        ping_two = two.ready()
        answers_two = _answers(two.url, states)
    finally:
        two.stop()
    said_alone, said_two = alone.logs[0].read_text(), two.logs[0].read_text()
    assert ping_alone["optimized"] == [] and "optimize: cfg_parallel off - it cuts a batch between 2 processes, and " \
                                             "this server is 1" in said_alone
    assert ping_two["optimized"] == ["cfg_parallel"], said_two[-3000:]
    assert "optimize: cfg_parallel on - policy.model, its batch cut between two processes" in said_two or \
           "optimize: cfg_parallel on - model, its batch cut between two processes" in said_two, said_two[-3000:]
    for one, split in zip(answers_alone, answers_two):
        assert float(np.max(np.abs(one - split))) <= 1e-6
