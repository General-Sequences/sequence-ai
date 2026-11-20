# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
A policy's one hook for both ends of an episode, @seq.on_episode, as the server runs it (sequence_ai/engine/adapter.py,
the class a deployment's worker and `seq serve` share): handed the episode's seq.Episode as it begins on a copy - before
its first frame or act there, episode.phase "start" - and as it ends, "end", the same Episode between; one that raises
as the episode begins fails that episode alone, said by its frame and every act of it until it ends, and one that raises
at its end is said on the server's log; an episode that never began on a copy has no end there; episode.local is the
episode's and the server's alone; and a policy that marks @seq.reset is refused where it is loaded.

    python -m pytest -q tests/engine/test_adapter_episodes.py
"""
from __future__ import annotations

import json
import types

import pytest

pytest.importorskip("sequence_base")

from sequence_base import SeqFlags  # noqa: E402
from sequence_base.by_name import RESET_GONE  # noqa: E402
from sequence_base.contract.policy import ImageSpec, PolicyRegistration, PolicySpec  # noqa: E402

from sequence_ai.engine import adapter, record  # noqa: E402

HOOKED = {SeqFlags.INFER: "act", SeqFlags.OBSERVE: "update", SeqFlags.ON_EPISODE: "on_episode"}


class _Hooked:
    """A class that keeps each episode in its seq.Episode: what it saw in episode.state, a history of its server's alone
    in episode.local, both begun as the episode starts; `broken`, the episode ids its hook raises on as they begin, and
    `end_broken` as they end."""

    broken: set = set()
    end_broken: set = set()

    def __init__(self):
        self.calls = []

    def on_episode(self, episode):
        own = {k: v for k, v in episode.state.items() if not k.startswith("seq.")}   # less the server's record
        self.calls.append((episode.phase, episode.id, own, dict(episode.local)))
        if episode.phase == "start":
            if episode.id in self.broken:
                raise ValueError("no gripper")
            episode.state["seen"], episode.local["history"] = [], []
        elif episode.id in self.end_broken:
            raise ValueError("the log is full")

    def update(self, obs, episode):
        episode.state["seen"].append(obs["x"])
        episode.local.setdefault("history", []).append(f"frame {obs['x']}")   # none after a switch: begun again

    def act(self, episode):
        return {"seen": list(episode.state["seen"]), "phase": episode.phase}


class _Plain(_Hooked):
    """With no observe function: its act function handed each act's inputs, and the episode."""

    def act(self, obs, episode):
        return {"x": obs["x"], "begun": episode.state.get("seen") == []}


def _adapter(bindings=None, *, episodes="own", cls=_Hooked) -> adapter.RegisteredAdapter:
    spec = PolicySpec(name="h", gpu="cpu", image=ImageSpec(base="python:3.11-slim"), episodes=episodes)
    reg = PolicyRegistration(spec=spec, handler_ref=f"{__name__}:{cls.__name__}", bindings=bindings or HOOKED)
    a = adapter.RegisteredAdapter(cls, types.SimpleNamespace(registration=reg, cls=cls))
    a.load(None)
    return a


@pytest.fixture(autouse=True)
def _none_broken(monkeypatch):
    monkeypatch.setattr(_Hooked, "broken", set())
    monkeypatch.setattr(_Hooked, "end_broken", set())


def test_its_hook_runs_as_an_episode_begins_and_as_it_ends_with_the_same_episode_its_phase_said_then_alone():
    a = _adapter()
    assert a.observe({"x": 1}, episode_id="e1", seq=0) == {"ok": True}       # its first frame begins it
    assert a.observe({"x": 2}, episode_id="e1", seq=1) == {"ok": True}
    assert a.infer(episode_id="e1") == {"seen": [1, 2], "phase": None}
    begun = a._episodes["e1"]
    a.episode_reset("e1")
    assert a._inst.calls == [("start", "e1", {}, {}),
                             ("end", "e1", {"seen": [1, 2]}, {"history": ["frame 1", "frame 2"]})]
    assert begun.phase is None                                                 # said while its hook runs alone
    b = _adapter({SeqFlags.INFER: "act", SeqFlags.ON_EPISODE: "on_episode"}, cls=_Plain)   # its first frame begins it
    assert b.observe({"x": 0}, episode_id="e2", seq=0) == {"ok": True}
    assert b.infer(episode_id="e2") == {"x": 0, "begun": True}
    assert b._inst.calls == [("start", "e2", {}, {})]


def test_an_episode_whose_hook_raises_as_it_begins_fails_alone_said_by_its_every_act_and_begins_again_once_ended():
    _Hooked.broken = {"bad"}
    a = _adapter()
    said = "@seq.on_episode raised as the episode began: ValueError: no gripper"
    assert a.observe({"x": 1}, episode_id="bad", seq=0) == {"ok": False, "error": said}
    for _ in range(2):
        with pytest.raises(RuntimeError, match=f"^{said}$") as raised:
            a.infer(episode_id="bad")
        assert isinstance(raised.value.__cause__, ValueError)                  # its own error under it
    assert [c[:2] for c in a._inst.calls] == [("start", "bad")]                # not run again for the same episode
    assert "bad" not in a._episodes
    assert a.observe({"x": 5}, episode_id="good", seq=0) == {"ok": True}      # another episode begins as ever
    assert a.infer(episode_id="good")["seen"] == [5]
    assert a.episode_reset("bad") is None                                      # it never began: no end of its own
    assert [c[:2] for c in a._inst.calls] == [("start", "bad"), ("start", "good")]
    _Hooked.broken = set()
    assert a.observe({"x": 7}, episode_id="bad", seq=0) == {"ok": True}       # ended, it begins anew
    assert a.infer(episode_id="bad")["seen"] == [7]


def test_a_hook_that_raises_as_an_episode_ends_is_said_on_the_servers_log_and_the_episode_ends(capsys):
    _Hooked.end_broken = {"e1"}
    a = _adapter()
    a.observe({"x": 1}, episode_id="e1", seq=0)
    assert a.episode_reset("e1") is None
    assert "e1" not in a._episodes
    assert "@seq.on_episode for episode e1 raised ValueError: the log is full" in capsys.readouterr().out


def test_an_episode_that_never_began_on_a_copy_or_ended_there_already_has_no_end_there():
    """Its hook runs once as each episode begins and once as it ends: an episode placed on a copy that never saw a frame
    or act of it has no end there, nor one pushed out to make room, whose end ran then."""
    a = _adapter(episodes="shared")
    assert a.episode_reset("placed") is None
    a.observe({"x": 1}, episode_id="e1", seq=0)
    a.observe({"x": 2}, episode_id="e2", seq=0)
    a.observe({"x": 3}, episode_id="e3", seq=0)
    a.infer(episode_id="e3", keep=2)               # the copy keeps two: e1 pushed out
    assert [c[:2] for c in a._inst.calls] == [("start", "e1"), ("start", "e2"), ("start", "e3"), ("end", "e1")]
    a.episode_reset("e1")
    assert [c[:2] for c in a._inst.calls].count(("end", "e1")) == 1


def test_an_episode_taken_on_from_its_saved_state_goes_on_without_a_start_its_local_empty_and_ends_as_any():
    a = _adapter()
    assert a.restore_episode("e1", state={"seen": [1, 2]}, steps=1, through=1) is True
    assert a._episodes["e1"].local == {}                                       # never saved: begun on the first server
    assert a._inst.calls == []                                                 # it goes on rather than begins
    assert a.observe({"x": 3}, episode_id="e1", seq=2) == {"ok": True}
    assert a.infer(episode_id="e1")["seen"] == [1, 2, 3]
    a.episode_reset("e1")
    assert [c[:2] for c in a._inst.calls] == [("end", "e1")]


def test_what_a_checkpoint_takes_of_an_episode_is_its_state_never_its_local():
    a = _adapter()
    a.observe({"x": 1}, episode_id="e1", seq=0)
    kept = record.state_of(a._episodes["e1"])
    assert kept["state"]["seen"] == [1] and "frame 1" not in json.dumps(kept, default=str)
    assert a.episode_reset("e1") is None


def test_a_policy_that_marks_seq_reset_is_refused_where_it_is_loaded_in_the_sdks_words():
    with pytest.raises(TypeError, match=r"^_Hooked\.on_episode: @seq\.reset is gone: use @seq\.on_episode") as refused:
        _adapter({SeqFlags.INFER: "act", SeqFlags.RESET: "on_episode"})
    assert str(refused.value) == f"_Hooked.on_episode: {RESET_GONE}"
