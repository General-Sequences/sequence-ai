# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
Compressed items, decompressed (sequence_ai/network/compressed.py): every array and bytes value is restored byte for
byte, with its dtype and shape; a value that holds nothing compressed is returned as the same object; a map whose `n` is
smaller than what its data holds is refused, and nothing is decompressed past `n`; a map that carries the reserved key
with another kind is never looked inside.

    python -m pytest -q tests/network/test_compressed.py
"""
from __future__ import annotations

import zlib

import numpy as np
import pytest

pytest.importorskip("zstandard")

from sequence_ai.network import (
    compressed,  # noqa: E402
    msgpack_io,  # noqa: E402
)
from sequence_ai.network.compressed import CompressedError, compress, expand  # noqa: E402

LIMIT = 64 << 20


def _frame(seed: int = 0) -> np.ndarray:
    """A camera-like uint8 frame: smooth, with a little noise, so it compresses as a real one does."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:96, 0:128]
    base = np.stack([xx * 2, yy * 2, (xx + yy)], -1).astype(np.int16)
    return np.clip(base + rng.normal(0, 3, base.shape), 0, 255).astype(np.uint8)


@pytest.mark.parametrize("codec", compressed.CODECS)
@pytest.mark.parametrize("value", [
    _frame(),
    np.linspace(-1, 1, 6000, dtype=np.float32).reshape(20, 300),
    np.arange(5000, dtype=">i8"),
    np.zeros((0, 3), dtype=np.uint8),
    np.array([True, False] * 9000),
], ids=["uint8-hwc", "float32", "int64-big-endian", "empty", "bool"])
def test_an_array_is_restored_byte_for_byte(codec, value):
    item = compress(value, codec)
    body = msgpack_io.unpack(msgpack_io.pack({"frames": [{"seq": 0, "inputs": {"cam": item}}]}))
    back, grew = expand(body["frames"][0]["inputs"], limit=LIMIT)
    out = back["cam"]
    assert out.dtype == value.dtype and out.shape == value.shape
    assert out.tobytes() == value.tobytes()
    assert not out.flags.writeable
    assert grew == max(0, value.nbytes - len(item["data"]))


@pytest.mark.parametrize("codec", compressed.CODECS)
def test_bytes_are_restored_as_bytes(codec):
    raw = bytes(range(256)) * 200
    back, _ = expand([compress(raw, codec), "kept"], limit=LIMIT)
    assert back == [raw, "kept"] and isinstance(back[0], bytes)


def test_nothing_compressed_is_the_same_object():
    value = {"joints": np.zeros(7), "say": "pick it up", "rows": [1, 2, 3]}
    back, grew = expand(value, limit=LIMIT)
    assert back is value and grew == 0


def test_a_marked_item_is_never_looked_inside():
    """A map that carries the reserved key with another kind (here a tactile reading, sent raw, that holds its value)
    belongs to whatever wrote it: expand returns it unchanged."""
    tactile = {"__seq__": "tactile", "v": 1, "codec": "raw", "value": compress(b"x" * 20000)}
    back, grew = expand({"finger": tactile}, limit=LIMIT)
    assert back["finger"] is tactile and grew == 0


@pytest.mark.parametrize("codec", compressed.CODECS)
def test_an_item_that_declares_less_than_it_holds_is_refused(codec):
    item = compress(b"y" * 50000, codec)
    with pytest.raises(CompressedError):
        expand({**item, "n": 1000}, limit=LIMIT)


def test_a_zstd_bomb_is_refused_by_what_its_map_declares():
    """A frame that decompresses to far more than its map declares, or than one item may hold, is refused before it is
    decompressed past either limit."""
    import zstandard

    bomb = zstandard.ZstdCompressor(level=1).compress(b"\0" * (8 << 20))
    with pytest.raises(CompressedError):
        expand({"__seq__": "compressed", "v": 1, "codec": "zstd", "n": 100, "data": bomb}, limit=LIMIT)
    with pytest.raises(CompressedError):
        expand({"__seq__": "compressed", "v": 1, "codec": "zstd", "n": 8 << 20, "data": bomb}, limit=1 << 20)


def test_a_zlib_stream_longer_than_its_map_is_refused():
    data = zlib.compress(b"\0" * 100000, 1)
    with pytest.raises(CompressedError):
        expand({"__seq__": "compressed", "v": 1, "codec": "zlib", "n": 1000, "data": data}, limit=LIMIT)


@pytest.mark.parametrize("item", [
    {"codec": "lz4"},
    {"n": -1},
    {"n": True},
    {"data": "not bytes"},
    {"type": "|O"},
    {"type": "<f4", "shape": [7, 7]},
    {"type": "<u1", "shape": "16"},
], ids=["codec", "negative", "bool-n", "str-data", "object-dtype", "wrong-size", "shape"])
def test_a_map_that_does_not_read_is_refused(item):
    good = compress(np.zeros((40, 50), dtype=np.uint8))
    with pytest.raises(CompressedError):
        expand({**good, **item}, limit=LIMIT)
