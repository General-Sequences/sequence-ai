# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
Values as msgpack (sequence_ai/network/msgpack_io.py): every value round-trips unchanged, with numpy arrays written byte
for byte as msgpack-numpy writes them; openpi's layout is read as the same arrays; an array that holds objects (which
msgpack-numpy pickles) and every extension type are refused; without numpy an array is read as plain lists.

    python -m pytest -q tests/network/test_msgpack_io.py
"""
from __future__ import annotations

import base64
import json
import math
import pickle
import sys

import pytest

from sequence_ai.network import msgpack_io
from sequence_ai.network.msgpack_io import MsgpackError, pack, unpack

np = pytest.importorskip("numpy")

INTS = [0, 1, 127, 128, 255, 256, 65535, 65536, 2**32 - 1, 2**32, 2**64 - 1, -1, -32, -33, -128, -129, -32768, -32769,
        -2**31, -2**31 - 1, -2**63]
STRS = ["", "a", "é\u4e2d", "x" * 31, "x" * 32, "x" * 255, "x" * 256, "x" * 65535, "x" * 65536]
BLOBS = [b"", b"\x00", b"\xff" * 255, b"\x01" * 256, b"\x02" * 65536]


def test_every_plain_value_round_trips_unchanged():
    envelope = {"inputs": {"ints": INTS, "strs": STRS, "blobs": BLOBS, "floats": [0.0, -1.5, 1e308, float("inf")],
                           "flags": [True, False, None], "nested": {"a": [[1, [2, {"b": b"c"}]]]}, 7: "int key",
                           b"k": "bytes key"},
                "seed": 3, "episode_id": "ep-1", "params": None, "lists": [list(range(15)), list(range(16)),
                                                                          list(range(70000))]}
    assert unpack(pack(envelope)) == envelope
    assert unpack(pack((1, 2))) == [1, 2]                                         # a tuple is packed as a list
    assert math.isnan(unpack(pack(float("nan"))))


def test_ints_strings_and_containers_take_the_smallest_encoding_msgpack_does():
    msgpack = pytest.importorskip("msgpack")
    for value in [*INTS, *STRS, *BLOBS, [1] * 15, [1] * 16, {str(i): i for i in range(15)},
                  {str(i): i for i in range(16)}, 1.5, None, True]:
        assert pack(value) == msgpack.packb(value, use_bin_type=True), repr(value)[:60]
        assert unpack(msgpack.packb(value, use_bin_type=True)) == value


def test_numpy_arrays_are_written_byte_for_byte_as_msgpack_numpy_writes_them_and_it_reads_ours():
    msgpack = pytest.importorskip("msgpack")
    msgpack_numpy = pytest.importorskip("msgpack_numpy")
    frame = np.arange(2 * 4 * 3, dtype=np.uint8).reshape(2, 4, 3)
    values = [frame, np.zeros((0, 3), np.float32), np.array(3.5, np.float64), np.array([True, False]),
              np.arange(6, dtype=">i4").reshape(2, 3), np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2],
              np.float32(1.25), np.int64(-7), np.bool_(True), np.array(["ab", "c"])]
    envelope = {"inputs": {"vision": {"cam_head": {"color": frame}}, "state": np.ones(7)}, "seed": 1}
    for value in [*values, envelope]:
        theirs = msgpack.packb(value, default=msgpack_numpy.encode, use_bin_type=True)
        assert pack(value) == theirs, repr(value)[:60]
        back = msgpack.unpackb(pack(value), object_hook=msgpack_numpy.decode, raw=False)
        _same(back, value)
        _same(unpack(theirs), value)


def test_openpis_layout_is_read_as_the_same_arrays():
    """openpi-client's own msgpack_numpy (copied here as it is in openpi): {b"__ndarray__", b"data", b"dtype",
    b"shape"} and {b"__npgeneric__", b"data", b"dtype"}."""
    msgpack = pytest.importorskip("msgpack")

    def pack_array(obj):
        if isinstance(obj, np.ndarray):
            return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
        if isinstance(obj, np.generic):
            return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
        return obj

    obs = {"observation/image": np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3),
           "observation/state": np.linspace(0, 1, 8, dtype=np.float32), "prompt": "pick", "step": np.int32(4)}
    _same(unpack(msgpack.packb(obs, default=pack_array)), obs)


def test_arrays_are_returned_read_only_with_their_dtype_and_shape():
    arr = np.arange(24, dtype=">f8").reshape(2, 3, 4)[:, 1:, ::2]                  # big-endian, not contiguous
    back = unpack(pack(arr))
    assert back.dtype == arr.dtype and back.shape == arr.shape and np.array_equal(back, arr)
    assert not back.flags.writeable                                                 # as msgpack-numpy's are


def test_an_array_that_holds_objects_is_refused_and_nothing_is_unpickled(monkeypatch):
    """msgpack-numpy writes an array of objects with pickle; reading it would run what the sender chose."""
    msgpack = pytest.importorskip("msgpack")
    msgpack_numpy = pytest.importorskip("msgpack_numpy")
    hostile = msgpack.packb(np.array([{"a": 1}, None], dtype=object), default=msgpack_numpy.encode, use_bin_type=True)
    monkeypatch.setattr(pickle, "loads", lambda *a, **k: pytest.fail("pickle.loads ran on a body"))
    with pytest.raises(MsgpackError, match="only plain numeric"):
        unpack(hostile)
    with pytest.raises(MsgpackError, match="only plain numeric"):
        pack(np.array([object()], dtype=object))
    structured = np.zeros(2, dtype=[("x", "<f4"), ("y", "<i4")])
    with pytest.raises(MsgpackError, match="only plain numeric"):
        pack(structured)
    with pytest.raises(MsgpackError, match="only plain numeric"):
        unpack(msgpack.packb(structured, default=msgpack_numpy.encode, use_bin_type=True))


@pytest.mark.parametrize("body, why", [
    (b"\xd4\x01\x00", "extension"),                     # fixext 1
    (b"\xc7\x01\x05\x00", "extension"),                 # ext 8
    (b"\xc1", "unused"),
    (b"\xda\x00\x05ab", "ends inside"),                 # str16 of 5, 2 bytes
    (b"\x01\x02", "follow"),                            # two values
    (b"\xdd\xff\xff\xff\xff", "list of"),               # a list of 4 billion items, nothing behind it
    (b"\xdf\x00\x01\x00\x00", "map of"),                # a map of 65536 entries in 2 bytes
    (b"\x81\x91\x01\x02", "map key"),                   # a list as a key
    (b"\x91" * 70 + b"\x01", "deeper"),
    (b"\xa1\xff", "UTF-8"),
])
def test_a_body_that_is_not_one_supported_value_is_refused_with_the_reason(body, why):
    with pytest.raises(MsgpackError, match=why):
        unpack(body)


def test_a_map_that_only_looks_like_an_array_is_the_users_dict():
    for value in ({"nd": True, "x": 1}, {"nd": 3}, {"__ndarray__": True}, {b"nd": False}):
        assert unpack(pack(value)) == value


def test_an_array_whose_data_disagrees_with_its_shape_is_refused():
    body = bytearray(pack(np.zeros((2, 2), np.float32)))
    with pytest.raises(MsgpackError, match="its data is"):
        unpack(bytes(body).replace(b"\x92\x02\x02", b"\x92\x02\x03"))              # shape [2, 3], 16 bytes of data


def test_without_numpy_an_array_is_read_as_the_lists_its_tolist_returns(monkeypatch):
    """A client without numpy reads a response that holds arrays (a policy may return a dict of them) as the nested
    lists of plain values a JSON response holds (ndarray.tolist()), in both layouts; with numpy it gets arrays."""
    msgpack = pytest.importorskip("msgpack")
    arrays = {"f4": np.linspace(-1, 1, 6, dtype=np.float32).reshape(2, 3), "f8": np.arange(4, dtype=">f8"),
              "f2": np.array([0.5, -2.0], np.float16), "i8": np.array([-2**63, 2**63 - 1]),
              "u1": np.arange(250, 255, dtype=np.uint8), "i2": np.array([[-3, 4]], dtype=">i2"),
              "b1": np.array([True, False]), "c8": np.array([1 + 2j, -0.5j], np.complex64),
              "U": np.array(["a", "b\u00e9\u4e2d"]), "S": np.array([b"x", b"yz"]), "0-d": np.array(7.25),
              "empty": np.zeros((2, 0, 3), np.float32), "none": np.zeros((0, 4), np.int32)}
    scalars = {"i4": np.int32(-7), "f8": np.float64(1.5), "b": np.bool_(True)}
    body = pack({"outputs": arrays, "scalars": scalars, "note": "x", "frame": b"\xff\xd8"})
    openpi = msgpack.packb({"a": np.arange(6, dtype=np.int16).reshape(3, 2), "s": np.float32(2.5)},
                           default=_openpi_array)
    want = {k: v.tolist() for k, v in arrays.items()}
    monkeypatch.setitem(sys.modules, "numpy", None)            # `import numpy` fails, as on a robot that has none
    got = unpack(body)
    assert got == {"outputs": want, "scalars": {"i4": -7, "f8": 1.5, "b": True}, "note": "x", "frame": b"\xff\xd8"}
    _plain_values_only(got)
    assert unpack(openpi) == {"a": [[0, 1], [2, 3], [4, 5]], "s": 2.5}
    monkeypatch.undo()
    _same(unpack(body)["outputs"], arrays)                      # with numpy: the arrays, as before


@pytest.mark.parametrize("type_, shape, data, why", [
    (b"|O", [1], b"\x00" * 8, "only plain numeric"),          # what msgpack-numpy pickles
    (b"|V8", [1], b"\x00" * 8, "only plain numeric"),         # a structured array
    ([["x", "<f4"]], [1], b"\x00" * 4, "only plain numeric"),
    (b"|S0", [10**12], b"", "0 bytes"),                       # a trillion empty strings in no bytes at all
    (b"<f16", [1], b"\x00" * 16, "only numpy can read"),          # a long double
    (b"<M8[ns]", [1], b"\x00" * 8, "only numpy can read"),
    (b"<f4", [1] * 65, b"\x00" * 4, "dimensions"),
])
def test_without_numpy_an_unreadable_type_is_refused_with_the_reason(monkeypatch, type_, shape, data, why):
    body = pack({b"nd": True, b"type": type_, b"kind": b"", b"shape": shape, b"data": data})
    monkeypatch.setitem(sys.modules, "numpy", None)
    with pytest.raises(MsgpackError, match=why):
        unpack(body)


def test_what_msgpack_cannot_carry_is_refused_when_packed():
    for value, why in ((2**64, "64-bit"), (-2**63 - 1, "64-bit"), ({1.5: 1}, "key"), ({True: 1}, "key"),
                       ({1, 2}, "cannot be sent"), ("\ud800", "UTF-8"), (object(), "cannot be sent")):
        with pytest.raises(MsgpackError, match=why):
            pack(value)


def test_three_jpeg_frames_take_a_quarter_less_than_base64_json():
    """What msgpack saves on a request that carries encoded frames: their bytes are sent as they are, where JSON carries
    them as base64."""
    frames = {f"cam_{i}": bytes(np.random.default_rng(i).integers(0, 256, 80_000, dtype=np.uint8)) for i in range(3)}
    as_msgpack = len(pack({"inputs": frames, "seed": 1}))
    as_json = len(json.dumps({"inputs": {k: base64.b64encode(v).decode() for k, v in frames.items()}, "seed": 1}))
    assert as_msgpack < 0.76 * as_json and as_msgpack - sum(map(len, frames.values())) < 64


def test_the_media_type_is_its_own():
    assert msgpack_io.MEDIA_TYPE == "application/x-seq-msgpack"


def _openpi_array(obj):
    """openpi-client's msgpack_numpy pack_array, as it is in openpi."""
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _plain_values_only(value):
    if isinstance(value, dict):
        for v in value.values():
            _plain_values_only(v)
    elif isinstance(value, list):
        for v in value:
            _plain_values_only(v)
    else:
        assert value is None or type(value) in (bool, int, float, complex, str, bytes), type(value)


def _same(got, want):
    if isinstance(want, np.ndarray):
        assert isinstance(got, np.ndarray) and got.dtype == want.dtype and got.shape == want.shape
        assert np.array_equal(got, want)
    elif isinstance(want, np.generic):
        assert isinstance(got, np.generic) and got.dtype == want.dtype and got == want
    elif isinstance(want, dict):
        assert set(got) == set(want)
        for k in want:
            _same(got[k], want[k])
    elif isinstance(want, (list, tuple)):
        assert len(got) == len(want)
        for g, w in zip(got, want):
            _same(g, w)
    else:
        assert got == want


def test_a_pil_image_is_sent_as_its_array():
    """A PIL image - frames.resize_image's output - is packed as the numpy array its pixels are, and read back as that
    array; without this a robot's fitted frame was refused "Image cannot be sent as msgpack"."""
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    pixels = (np.arange(40 * 50 * 3) % 251).astype(np.uint8).reshape(40, 50, 3)
    got = unpack(pack({"image": Image.fromarray(pixels), "gray": Image.fromarray(pixels[..., 0])}))
    assert np.array_equal(got["image"], pixels) and got["image"].dtype == np.uint8
    assert np.array_equal(got["gray"], pixels[..., 0])


def test_a_pil_module_still_being_imported_leaves_arrays_packed_as_arrays(monkeypatch):
    """frames.warm() imports Pillow on a thread of its own as the server first says it reads JPEG, and the robot's
    first act packs meanwhile: PIL.Image is in sys.modules with no Image class yet, and every array is packed as it
    is, where reading that class raised AttributeError."""
    np = pytest.importorskip("numpy")
    import types

    monkeypatch.setitem(sys.modules, "PIL.Image", types.ModuleType("PIL.Image"))     # its Image not defined yet
    state = np.arange(6, dtype=np.float32)
    assert np.array_equal(unpack(pack({"state": state}))["state"], state)
