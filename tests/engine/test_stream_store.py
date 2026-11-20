# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
engine.stream: an episode's observations kept in order and forwarded to the copy of its policy in order; an act waits
for the ones before it and is refused, from the first missing, when they do not come; a robot that moved here is whole
from the first frame it sends; numbers the robot left out count as had; an episode or a server past what it keeps
refuses a post - or, for a post that may shed, lets its oldest unread go; an episode read back from its saved state
begins after it. The same module the hosted platform's workers run.

    python -m pytest -q tests/engine/test_stream_store.py
"""
from __future__ import annotations

import asyncio

import pytest

from sequence_ai.engine.stream import StreamError, StreamStore


def frames_through(observe):
    async def observe_frames(episode_id, frames, handed=None):
        answers = []
        for seq, inputs in frames:
            answers.append(await observe(episode_id, seq, inputs))
            if not (isinstance(answers[-1], dict) and answers[-1].get("ok")):
                break
        return answers
    return observe_frames


def _frames(*seqs):
    return [(s, {"frame": s}) for s in seqs]


def _observed(store, episode):
    """The frames of `episode` kept in `store` handed to its replica, as (seq, frame) in order."""
    seen = []

    async def observe(eid, seq, inputs):
        seen.append((seq, inputs["frame"]))
        return {"ok": True}

    asyncio.run(store.forward(episode, frames_through(observe)))
    return seen


def test_frames_go_to_the_replica_in_order_then_they_are_forgotten():
    store = StreamStore()
    assert store.put("e1", _frames(0, 1, 2), [], 300) == 2
    assert _observed(store, "e1") == [(0, 0), (1, 1), (2, 2)]
    assert store.bytes == 0                                      # gone: forgotten
    assert store.put("e1", _frames(3, 4), [], 200) == 4
    assert _observed(store, "e1") == [(3, 3), (4, 4)]
    assert _observed(store, "e1") == []                          # nothing since


def test_a_frame_sent_again_is_answered_never_kept_twice_and_gaps_wait_for_their_frames():
    store = StreamStore()
    assert store.put("e1", _frames(0, 2), [], 200) == 0          # 1 missing: whole through 0
    assert store.put("e1", _frames(0, 1), [], 200) == 2          # 0 again, 1 fills the gap
    assert store.bytes == 300                                    # 0, 1 and 2 once each
    assert _observed(store, "e1") == [(0, 0), (1, 1), (2, 2)]
    assert store.put("e1", _frames(1, 2), [], 200) == 2          # already gone: not kept
    assert store.bytes == 0


def test_skipped_numbers_are_not_waited_for():
    store = StreamStore()
    assert store.put("e1", _frames(0, 2), [1], 200) == 2
    assert _observed(store, "e1") == [(0, 0), (2, 2)]
    asyncio.run(asyncio.wait_for(store.before("e1", 3), timeout=1.0))


def test_an_act_waits_for_frames_on_their_way_and_one_that_never_comes_is_named():
    async def main():
        store = StreamStore(wait_s=1.0)

        async def observe(eid, seq, inputs):
            return {"ok": True}

        store.put("e1", _frames(0), [], 100)
        await store.forward("e1", frames_through(observe))
        waiting = asyncio.ensure_future(store.before("e1", 3))
        await asyncio.sleep(0.05)
        assert not waiting.done()
        store.put("e1", _frames(1, 2), [], 200)
        await store.forward("e1", frames_through(observe))
        await waiting
        quick = StreamStore(wait_s=0.1)
        quick.put("e2", _frames(0, 1), [], 200)
        await quick.forward("e2", frames_through(observe))
        with pytest.raises(StreamError) as missing:
            await quick.before("e2", 5)
        assert missing.value.missing and missing.value.first == 2
        await quick.before("e2", 2)                              # nothing waited for: its 0 and 1 went

    asyncio.run(main())


def test_an_episode_is_whole_from_the_first_frame_a_worker_is_sent():
    """A robot moved to this server sends what it kept since its last served act: frames 25 on, not 0 on."""
    async def main():
        store = StreamStore(wait_s=0.1)
        seen = []

        async def observe(eid, seq, inputs):
            seen.append(seq)
            return {"ok": True}

        with pytest.raises(StreamError) as missing:
            await store.before("e1", 30)                         # the act first, on a worker that has nothing
        assert missing.value.first == 0
        assert store.put("e1", _frames(25, 26, 27, 28, 29), [], 500) == 29
        await store.forward("e1", frames_through(observe))
        assert seen == list(range(25, 30))
        await store.before("e1", 30)

    asyncio.run(main())


@pytest.mark.parametrize("limits, size", [({"max_frames": 2}, 30), ({"max_bytes": 100}, 300),
                                          ({"max_worker_bytes": 100}, 300)])
def test_past_a_limit_a_post_is_refused_whole(limits, size):
    store = StreamStore(**limits)
    with pytest.raises(StreamError) as over:
        store.put("e1", _frames(0, 1, 2), [], size)
    assert not over.value.missing and store.bytes == 0
    assert store.put("e1", _frames(0), [], 10) == 0              # a smaller one is kept


def test_a_number_a_full_episode_refused_is_taken_as_left_out_and_no_act_waits_for_it():
    """The robot's SDK sends the numbers of frames this refused as `skipped`: taken even when the episode is full, as
    they hold nothing, and the act after them is served with the frames that were kept."""
    store = StreamStore(max_frames=2, wait_s=0.1)
    assert store.put("e1", _frames(0, 1), [], 200) == 1
    with pytest.raises(StreamError) as over:
        store.put("e1", _frames(2), [], 100)                     # past its 2: refused whole
    assert not over.value.missing
    assert store.put("e1", [], [2], 0) == 2                      # left out: whole through it
    assert _observed(store, "e1") == [(0, 0), (1, 1)]
    asyncio.run(asyncio.wait_for(store.before("e1", 3), timeout=1.0))


def test_an_ended_episode_goes_and_an_idle_one_expires():
    now = [0.0]
    store = StreamStore(ttl_s=10.0, clock=lambda: now[0])
    store.put("e1", _frames(0, 1), [], 200)
    store.put("e2", _frames(0), [], 100)
    store.end("e1")
    assert len(store) == 1 and store.bytes == 100
    now[0] = 11.0
    store.put("e3", _frames(0), [], 100)                         # any post expires the idle ones
    assert len(store) == 1 and store.bytes == 100


def test_a_policy_that_observes_is_sent_each_frame_in_order_and_its_act_waits_for_them_to_go():
    async def main():
        store = StreamStore(wait_s=1.0)
        seen = []

        async def observe(eid, seq, inputs):
            await asyncio.sleep(0.01)
            seen.append((eid, seq, inputs["frame"]))
            return {"ok": True}

        store.put("e1", _frames(0, 1), [], 200)
        first = asyncio.ensure_future(store.forward("e1", frames_through(observe)))
        store.put("e1", _frames(2), [], 100)
        second = asyncio.ensure_future(store.forward("e1", frames_through(observe)))     # taken up by the one running
        await store.before("e1", 3)                                      # once 0..2 went
        await asyncio.gather(first, second)
        assert seen == [("e1", 0, 0), ("e1", 1, 1), ("e1", 2, 2)] and store.bytes == 0

    asyncio.run(main())


def test_posts_after_an_episode_s_first_come_in_any_order_and_its_replica_is_handed_each_frame_in_order():
    """Once an episode's first post is answered, a robot has several on their way at once (the robot's
    observe_stream.IN_FLIGHT): a later one that comes first is kept, its frames waiting for the
    ones before them, and the replica is handed every frame in order, each once. The first goes alone, for the episode
    is whole from the first frame the server is sent (the test above)."""
    async def main():
        store = StreamStore(wait_s=1.0)
        seen = []

        async def observe(eid, seq, inputs):
            seen.append(seq)
            return {"ok": True}

        assert store.put("e1", _frames(0), [], 100) == 0              # its first, alone: the episode begins at 0
        await store.forward("e1", frames_through(observe))
        assert store.put("e1", _frames(3, 4), [], 200) == 0           # a later post first: kept, whole through 0
        await store.forward("e1", frames_through(observe))
        assert seen == [0]                                            # nothing goes past the frames not here
        assert store.put("e1", _frames(1, 2), [], 200) == 4
        await store.forward("e1", frames_through(observe))
        assert seen == [0, 1, 2, 3, 4] and store.bytes == 0
        await store.before("e1", 5)

    asyncio.run(main())


def test_the_frame_a_replica_refuses_ends_its_episodes_forwarding_and_its_act_goes_on():
    async def main():
        store = StreamStore(wait_s=1.0)
        seen = []

        async def observe(eid, seq, inputs):
            seen.append(seq)
            return {"ok": seq != 1, **({} if seq != 1 else {"error": "observe raised at frame 1"})}

        store.put("e1", _frames(0, 1, 2), [], 300)
        await store.forward("e1", frames_through(observe))
        assert seen == [0, 1]                                    # 2 never went
        await store.before("e1", 3)                              # a refused frame: its act waits for nothing, and says
        assert store.bytes == 0

    asyncio.run(main())



def test_an_episode_read_back_on_this_copy_begins_after_what_its_state_holds():
    """An episode read back from its saved state on a copy it was not on (a switch to its
    standby): its observations up to `through` went to its policy on the copy it left, so here it begins at through + 1
    - its robot, sending again from its last answered act, has the frames at or before it answered and not kept, never
    observed twice, and an act waits on those after it alone."""
    async def main():
        store = StreamStore(wait_s=0.5)
        seen = []

        async def observe(eid, seq, inputs):
            seen.append(seq)
            return {"ok": True}

        store.resume("e1", 4)
        assert store.put("e1", _frames(3, 4, 5, 6, 7), [], 500) == 7          # 3 and 4 answered, not kept
        await store.forward("e1", frames_through(observe))
        assert seen == [5, 6, 7]
        await store.before("e1", 8)                                            # nothing before 5 waited for
        assert store.bytes == 0

    asyncio.run(main())


def test_a_post_that_may_shed_lets_the_oldest_unread_go_never_itself_and_no_act_waits_for_them():
    """A robot a person drives: its policy reads none of its observations, and past what a server keeps a post is not
    refused - the episode's oldest unread let go (counted, its bytes given back) - so its console's picture goes on; an
    act after them waits for none of those let go, and what comes after still goes to its replica in order. A post that
    may not shed is refused as ever."""
    store = StreamStore(max_frames=4, max_bytes=1000)
    for n in range(4):
        assert store.put("e1", _frames(n), [], 100, shed=True) == n
    assert store.put("e1", _frames(4), [], 100, shed=True) == 4       # past the limit: the oldest let go
    assert store.put("e1", _frames(5, 6), [], 200, shed=True) == 6
    assert store.shed("e1") == 3 and store.bytes == 400               # 3, 4, 5 and 6 kept
    with pytest.raises(StreamError, match="more than the 4"):
        store.put("e1", _frames(7), [], 100)                          # a post that may not shed: refused as ever
    assert _observed(store, "e1") == [(3, 3), (4, 4), (5, 5), (6, 6)]  # the kept ones go on, in order
    asyncio.run(asyncio.wait_for(store.before("e1", 7), 1.0))          # an act after them waits for none let go
    assert store.bytes == 0


def test_a_post_that_may_shed_makes_room_by_bytes_and_in_the_whole_worker_too():
    """By an episode's bytes and by the server's: the same - the post's own observations kept, its episode's oldest
    let go."""
    store = StreamStore(max_frames=100, max_bytes=300)
    store.put("e1", _frames(0, 1, 2), [], 300, shed=True)
    assert store.put("e1", _frames(3), [], 100, shed=True) == 3
    assert store.bytes == 300 and store.shed("e1") == 1
    whole = StreamStore(max_frames=100, max_bytes=10_000, max_worker_bytes=300)
    whole.put("e1", _frames(0, 1, 2), [], 300, shed=True)
    assert whole.put("e1", _frames(3), [], 100, shed=True) == 3
    assert whole.bytes == 300 and whole.shed("e1") == 1
