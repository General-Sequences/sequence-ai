# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
An observation's large arrays and bytes, compressed on the robot where that saves time
(sequence_ai/network/compress.py): data that does not compress, or that is sent over a link too fast for compression to
save time, is sent uncompressed and tried again after 64 observations; the uplink's rate is what large uploads take
beyond the RTT; small values and maps that already carry the reserved key are left unchanged.

    python -m pytest -q tests/network/test_compress.py
"""
from __future__ import annotations

import ast
import os
import pathlib

import pytest

np = pytest.importorskip("numpy")

from sequence_ai.network import compress  # noqa: E402


def _frame(i: int = 0, h: int = 120, w: int = 160) -> "np.ndarray":
    """A camera-like frame: flat regions and edges, as a camera sees a table, which compresses about as well as a real
    frame (real DROID frames: about 1.9 times with zstd level 1)."""
    yy, xx = np.mgrid[0:h, 0:w]
    return np.stack([(xx // 8) * 9 + i, (yy // 6) * 7, ((xx + yy) // 16) * 11], -1).astype(np.uint8)


def test_data_that_does_not_compress_is_sent_uncompressed_for_its_next_64_observations(monkeypatch):
    squeeze = compress.Squeeze()
    tries = []
    pack = squeeze._pack
    monkeypatch.setattr(squeeze, "_pack", lambda b: tries.append(len(b)) or pack(b))
    noise = {"jpeg_user": os.urandom(50_000)}
    assert squeeze.apply(noise) is noise and len(tries) == 1          # compressed once, sent uncompressed
    for _ in range(compress.PAUSE - 1):
        assert squeeze.apply(noise) is noise
    assert len(tries) == 1                                              # not compressed while paused
    squeeze.apply(noise)
    assert len(tries) == 2                                              # then tried again


def test_compression_stops_while_the_link_is_too_fast_for_it_to_save_time():
    frame = {"head_user": _frame()}
    slow, fast = compress.Squeeze(), compress.Squeeze()
    fast.uplink.note(1_000, 0.0)                                        # an RTT of zero
    fast.uplink.note(10_000_000, 0.002)                                 # 5 GB per second: a byte saved is worth nothing
    for squeeze in (slow, fast):
        assert squeeze.apply(frame)["head_user"]["__seq__"] == "compressed"      # first: compressed to learn its ratio
    assert slow.apply(frame)["head_user"]["__seq__"] == "compressed"             # 6 MB per second: it saves time
    assert fast.apply(frame) is frame                                             # it does not: uncompressed, for 64


def test_the_uplink_rate_is_what_large_posts_take_beyond_the_rtt():
    up = compress.Uplink()
    assert up.rate() == compress.RATE_S                                 # nothing measured yet: the default rate
    up.note(2_000, 0.25)                                                 # the RTT alone
    up.note(600_000, 0.35)                                               # 0.1 s more for 600 KB
    assert up.rate() == pytest.approx(6e6)
    up.note(600_000, 0.45)                                               # slower now: 3 MB per second
    assert up.rate() == pytest.approx(0.7 * 6e6 + 0.3 * 3e6)


def test_marked_items_and_small_values_are_left_unchanged():
    video = {"__seq__": "video", "v": 1, "codec": "h264", "data": b"\0" * 50_000}
    value = {"head_user": video, "joints_user": np.zeros(7), "small_user": b"\0" * 1000}
    assert compress.Squeeze().apply(value) is value


def test_the_compressing_parses_as_python_39():
    source = pathlib.Path(compress.__file__).read_text()
    ast.parse(source, feature_version=(3, 9))
