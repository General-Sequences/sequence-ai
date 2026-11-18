# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
frames.pack_pictures and unpack_pictures: every picture of an observation (an HxWx3 uint8 array, under any key, at any
depth) sent as JPEG and restored as its array, and nothing else touched.
"""
import numpy as np
import pytest

pytest.importorskip("PIL")

from sequence_ai.network import compressed, frames, msgpack_io  # noqa: E402


def picture(h: int = 120, w: int = 160) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    return np.stack([xx * 255 // w, yy * 255 // h, (xx + yy) * 127 // (h + w)], -1).astype(np.uint8)


def test_every_picture_at_any_depth_goes_as_jpeg_and_comes_back_close_to_itself():
    observation = {"cameras": {"left": picture(), "right": picture(96, 128)}, "wrist": [picture(64, 64)],
                   "state": np.arange(7, dtype=np.float32), "task": "pick up the cup"}
    packed, n = frames.pack_pictures(observation)
    assert n == 3
    assert frames.is_packed_picture(packed["cameras"]["left"]) and frames.is_packed_picture(packed["wrist"][0])
    assert packed["state"] is observation["state"] and packed["task"] == "pick up the cup"
    wire = msgpack_io.unpack(msgpack_io.pack(packed))
    assert len(msgpack_io.pack(packed)) < len(msgpack_io.pack(observation)) / 4
    restored = frames.unpack_pictures(wire)
    for got, sent in ((restored["cameras"]["left"], observation["cameras"]["left"]),
                      (restored["wrist"][0], observation["wrist"][0])):
        assert got.shape == sent.shape and got.dtype == np.uint8 and not got.flags.writeable
        assert np.abs(got.astype(int) - sent.astype(int)).mean() < 3
    assert np.array_equal(restored["state"], observation["state"]) and restored["task"] == "pick up the cup"


def test_only_hxwx3_uint8_arrays_of_32_pixels_a_side_are_pictures_and_the_caller_s_value_is_not_changed():
    others = {"chw": np.zeros((3, 64, 64), np.uint8), "gray": np.zeros((64, 64), np.uint8),
              "rgba": np.zeros((64, 64, 4), np.uint8), "float": np.zeros((64, 64, 3), np.float32),
              "small": np.zeros((31, 64, 3), np.uint8), "list": [[1, 2, 3]] * 40}
    observation = {**others, "image": picture()}
    packed, n = frames.pack_pictures(observation)
    assert n == 1 and all(packed[key] is value for key, value in others.items() if key != "list")
    assert packed["list"] == others["list"]                             # a container is a new one, its items kept
    assert isinstance(observation["image"], np.ndarray)                 # the caller's dict is not changed
    assert frames.is_picture(picture()) and not frames.is_picture(others["chw"]) and not frames.is_picture("x")


def test_unpack_leaves_other_kinds_alone_and_says_where_a_bad_picture_is():
    squeezed = compressed.compress(np.zeros(4096, np.float32), "zlib")
    assert frames.unpack_pictures({"depth": squeezed})["depth"] is squeezed       # compressed.expand's to restore
    with pytest.raises(ValueError, match="cameras/left: not a JPEG"):
        frames.unpack_pictures({"cameras": {"left": {"__seq__": "jpeg", "v": 1, "data": b"not a jpeg"}}})
    with pytest.raises(ValueError, match=r"wrist\[0\]: a picture of version 2"):
        frames.unpack_pictures({"wrist": [{"__seq__": "jpeg", "v": 2, "data": b""}]})


def test_a_picture_past_the_limit_is_refused_from_its_header_before_it_is_decoded():
    packed, _ = frames.pack_pictures({"image": picture(120, 160)})      # 57,600 bytes as an array
    assert frames.unpack_pictures(packed, limit=57_600)["image"].shape == (120, 160, 3)
    with pytest.raises(ValueError, match=r"image: a picture of 160 x 120 decodes to 57,600 bytes, past the limit "
                                         r"of 57,599"):
        frames.unpack_pictures(packed, limit=57_599)
    with pytest.raises(ValueError, match="image: not a JPEG or PNG image"):
        frames.unpack_pictures({"image": {"__seq__": "jpeg", "v": 1, "data": b"no header"}}, limit=1 << 20)


def test_a_pil_module_still_being_imported_makes_nothing_a_pil_image(monkeypatch):
    """warm() imports Pillow on a thread of its own as the server first says it reads JPEG; the robot's first act packs
    meanwhile and finds PIL.Image in sys.modules with no Image class yet: no picture of its is a PIL image, and an
    array is still one."""
    import sys
    import types

    monkeypatch.setitem(sys.modules, "PIL.Image", types.ModuleType("PIL.Image"))     # its Image not defined yet
    assert not frames.is_pil_image(picture()) and frames.is_picture(picture())
