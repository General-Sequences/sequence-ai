# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
A model's run of blocks split into one stage a process (sequence_ai/engine/pipeline_parallel.py): each process's
share as many blocks as its GPU runs in the time the others run theirs; the run found where its parent calls its
blocks one after another, each on the one before's output; in two processes as torchrun starts them, over gloo, a toy
diffusion transformer's actions in parts of its batch as one process's, in patches of its tokens near them - and the
same input's patches as one process's - its blocks' own forwards back after, a block called out of turn said, and a
stage that raises raised in both processes; a trial keeps the fastest form within the tolerance; a server of two
processes keeps the split and answers near one process's.

    python -m pytest -q tests/engine/test_pipeline_parallel.py
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch.distributed.pipelining")

from sequence_ai.engine import pipeline_parallel  # noqa: E402

from .test_cfg_parallel import _answers, _free_port, _Servers  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
RANK = HERE / "_pipeline_parallel_rank.py"
SERVER = HERE / "_pipeline_parallel_server.py"


def test_each_process_runs_as_many_blocks_as_its_gpu_runs_in_the_time_the_others_run_theirs():
    """stages_for: alike GPUs share the blocks evenly; a GPU twice as fast takes the blocks the slower leaves it; one
    block a stage at least, and more stages than blocks refused."""
    assert pipeline_parallel.stages_for([[1.0] * 4, [1.0] * 4]) == [2, 2]
    assert pipeline_parallel.stages_for([[1.0] * 4, [0.5] * 4]) == [1, 3]
    assert pipeline_parallel.stages_for([[1.0] * 6] * 3) == [2, 2, 2]
    assert pipeline_parallel.stages_for([[5.0, 1.0, 1.0], [1.0, 1.0, 5.0]]) == [1, 2]
    with pytest.raises(ValueError, match="3 blocks cannot be cut into 4 stages"):
        pipeline_parallel.stages_for([[1.0] * 3] * 4)


class _Chain(torch.nn.Module):
    """Blocks its forward calls one after another, each on the one before's output - given back as a tensor, or first
    in a tuple, or passed by keyword; or each added to its input outside the block, which is no chain."""

    def __init__(self, how: str = "tensor") -> None:
        super().__init__()
        self.how = how
        self.blocks = torch.nn.ModuleList([_Block(how) for _ in range(4)])

    def forward(self, x):
        for block in self.blocks:
            if self.how == "tuple":
                x = block(x)[0]
            elif self.how == "keyword":
                x = block(hidden_states=x)
            elif self.how == "residual":
                x = x + block(x)
            else:
                x = block(x)
        return x


class _Block(torch.nn.Module):
    def __init__(self, how: str) -> None:
        super().__init__()
        self.how, self.linear = how, torch.nn.Linear(6, 6)

    def forward(self, x=None, hidden_states=None):
        x = x if hidden_states is None else hidden_states
        q = x.view(x.shape[0], x.shape[1], 2, 3).transpose(1, 2)
        y = self.linear(x) + torch.nn.functional.scaled_dot_product_attention(q, q, q).transpose(1, 2).reshape(x.shape)
        return (y, None) if self.how == "tuple" else y


def test_census_finds_the_blocks_their_parent_calls_one_after_another_and_how_each_is_called():
    """census: the ModuleList whose blocks chain - the argument they chain by, how they give back their output, the
    batch and tokens of their input, the self-attentions a pass made over all its tokens; none for blocks each added
    to their input outside the block, nor for fewer blocks than processes."""
    x = torch.randn(3, 5, 6)
    chain = _Chain()
    got = pipeline_parallel.census({"model": chain}, lambda: chain(x), 2)
    assert got == {"path": "model.blocks", "blocks": 4, "slot": 0, "out": None, "batch": 3, "tokens": 5,
                   "attention": 4}
    tupled = _Chain("tuple")
    assert pipeline_parallel.census({"model": tupled}, lambda: tupled(x), 2)["out"] == ["tuple", 2]
    keyword = _Chain("keyword")
    assert pipeline_parallel.census({"model": keyword}, lambda: keyword(x), 2)["slot"] == "hidden_states"
    residual = _Chain("residual")
    assert pipeline_parallel.census({"model": residual}, lambda: residual(x), 2) is None
    assert pipeline_parallel.census({"model": chain}, lambda: chain(x), 8) is None


def test_two_processes_pipeline_a_diffusion_transformers_blocks_in_parts_and_in_patches(tmp_path):
    """Each rank runs a toy diffusion transformer's sampler whole, then split (tests/engine/_pipeline_parallel_rank.py):
    in two parts of its batch its actions are one process's within a hundred-thousandth, every call in parts; the
    shares timed on each process are the same on both; in two patches of its tokens they are near one process's, each
    act's first call whole and its others in patches, and a call of the input the patches' keys and values were kept
    for is one process's within a hundred-thousandth; uninstall puts the blocks' own forwards back; a block called out
    of turn is said, and a stage that raises on rank 1 is raised on both. [fails before pipeline_parallel.py]"""
    port = _free_port()
    base = {k: v for k, v in os.environ.items() if k not in ("WORLD_SIZE", "RANK", "MASTER_ADDR", "MASTER_PORT")}
    procs = [subprocess.Popen([sys.executable, str(RANK), str(tmp_path)],
                              env={**base, "WORLD_SIZE": "2", "RANK": str(rank), "MASTER_ADDR": "127.0.0.1",
                                   "MASTER_PORT": str(port), "OMP_NUM_THREADS": "1"},
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
             for rank in range(2)]
    try:
        outs = [p.communicate(timeout=180)[0] for p in procs]
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
    assert [p.returncode for p in procs] == [0, 0], outs
    got = [json.loads((tmp_path / f"{rank}.json").read_text()) for rank in range(2)]
    first = got[0]
    assert first["layers_max_abs"] <= 1e-5, got
    assert 0.0 < first["patches_max_abs"] <= 0.2, got
    assert first["once_max_abs"] <= 1e-5 and first["again_max_abs"] <= 1e-5, got
    for one in got:
        assert one["layers_ran"] == {"whole": 0, "parts": 4}
        assert one["patches_ran"] == {"whole": 1, "parts": 3}
        assert sum(one["stages"]) == 4 and min(one["stages"]) >= 1
        assert one["restored"] and one["after_same"]
        assert one["order"] == ("pipeline_parallel: model.blocks.1 was called where model.blocks.0 was next: its "
                                "parent does not call its blocks one after another, each once a pass")
        assert one["failed"] == ("pipeline_parallel: rank 1's stage of model.blocks raised: ValueError: this block "
                                 "fails on purpose")
    assert got[0]["stages"] == got[1]["stages"]


def test_a_group_of_one_is_refused_saying_how_to_start_it(monkeypatch):
    """group(): two processes or more, or a ValueError naming torchrun's --nproc-per-node 2."""
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    with pytest.raises(ValueError, match=r"has 1: start the server as 2 or more \(torchrun --nproc-per-node 2"):
        pipeline_parallel.group()


def test_a_trial_keeps_the_fastest_form_within_the_tolerance_and_none_without_one():
    """trial, its calls and toggles given as serve() gives them on rank 0: the blocks found, each form they can take
    tried - two and four patches of their tokens - every run held to eager mode's, the act timed both ways and the
    fastest kept (min_speedup 0 here); none faster by 10%, each form's reason said; with no tolerance, nothing
    tried."""
    chain = _Chain()

    def policy(observation):
        return {"out": chain(torch.as_tensor(observation["x"])).detach().numpy()}

    examples = [{"x": torch.randn(1, 4, 6).numpy()} for _ in range(3)]
    toggled: list = []
    with torch.inference_mode():
        report = pipeline_parallel.trial(policy, policy, lambda spec, on: toggled.append((dict(spec), on)), examples,
                                         size=2, limits={"max_abs": 1e-6}, min_speedup=0.0, rounds=1,
                                         say=lambda line: None)
    assert report["on"] == ["pipeline_parallel"], report
    kept = report["switches"]["pipeline_parallel"]
    assert kept["path"] == "chain.blocks" and kept["form"] == "patches" and kept["parts"] in (2, 4)
    assert toggled[-1][1] is True and toggled[-1][0]["parts"] == kept["parts"]
    assert {spec["parts"] for spec, on in toggled} == {2, 4}
    said: list = []
    with torch.inference_mode():
        slow = pipeline_parallel.trial(policy, policy, lambda spec, on: None, examples, size=2,
                                       limits={"max_abs": 1e-6}, min_speedup=10.0, rounds=1, say=said.append)
    assert slow["on"] == [] and said[0].startswith("optimize: pipeline_parallel off - each call's tokens in 2 "
                                                   "patches, ") and "each call's tokens in 4 patches, " in said[0]
    said = []
    off = pipeline_parallel.trial(policy, policy, lambda spec, on: None, examples, size=2, limits=None,
                                  say=said.append)
    assert off["on"] == [] and said == ["optimize: pipeline_parallel off - it changes the outputs: give tolerance=, "
                                        "the largest difference you accept from eager mode's"]


def test_without_a_tolerance_it_is_refused_before_the_server_starts():
    """check_switches: pipeline_parallel changes the outputs, so it needs tolerance=."""
    from sequence_ai.engine.optimize import check_switches

    with pytest.raises(ValueError, match="pipeline_parallel changes the policy's outputs: give tolerance="):
        check_switches(pipeline_parallel=True)
    assert check_switches(pipeline_parallel=True, tolerance=0.1)["pipeline_parallel"] is True


def test_a_server_of_two_processes_keeps_the_split_and_answers_near_one_process(tmp_path):
    """serve(pipeline_parallel=True) as two ranks over gloo (tests/engine/_pipeline_parallel_server.py, kept: any
    speed passes): its warm-up finds the diffusion transformer's blocks, splits them on both ranks and keeps them -
    /ping says so - and every act is answered near one process's answer, within 0.2; one process alone says it
    is one, and tries nothing. [fails before serve() took pipeline_parallel]"""
    pytest.importorskip("sequence_base")
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(3)
    states = [rng.standard_normal(7).astype(np.float32) for _ in range(4)]
    alone = _Servers(tmp_path, 1, server=SERVER)
    try:
        ping_alone = alone.ready()
        answers_alone = _answers(alone.url, states)
    finally:
        alone.stop()
    two = _Servers(tmp_path, 2, "kept", server=SERVER)
    try:
        ping_two = two.ready(timeout=180.0)
        answers_two = _answers(two.url, states)
    finally:
        two.stop()
    said_alone, said_two = alone.logs[0].read_text(), two.logs[0].read_text()
    assert ping_alone["optimized"] == [] and "optimize: pipeline_parallel off - it splits a model's blocks between " \
                                             "processes, and this server is 1" in said_alone
    assert ping_two["optimized"] == ["pipeline_parallel"], said_two[-3000:]
    assert "optimize: pipeline_parallel on - " in said_two and "its 4 blocks in 2 stages of " in said_two, \
        said_two[-3000:]
    for one, split in zip(answers_alone, answers_two):
        assert float(np.max(np.abs(one - split))) <= 0.2
