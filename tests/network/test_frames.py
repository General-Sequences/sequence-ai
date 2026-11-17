# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
A camera frame resized on the robot (sequence_ai/network/frames.py): `encode()` sends a frame at its own size and
returns already-encoded input without re-encoding it; `resize_image` resizes one the way a policy was trained, with its
mode and filter passed explicitly; "pad" matches openpi's resize_with_pad byte for byte; `image_size` reads a frame's
size from its header.

    python -m pytest -q tests/network/test_frames.py
"""

import base64
import io
import struct

import pytest

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

from sequence_ai.network.frames import (  # noqa: E402
    DEFAULT_JPEG_QUALITY,
    FILTERS,
    MODES,
    encode,
    image_size,
    resize_image,
)


def _decoded(data: str):
    return Image.open(io.BytesIO(base64.b64decode(data)))


def _frame(h: int, w: int, seed: int = 0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _template_encode(frame, width: int, height: int) -> str:
    """openpi's resize_with_pad, then JPEG at 95: the reference "pad" must match, byte for byte."""
    img = Image.fromarray(frame).convert("RGB")
    cur_width, cur_height = img.size
    if (cur_width, cur_height) != (width, height):
        ratio = max(cur_width / width, cur_height / height)
        resized_width, resized_height = int(cur_width / ratio), int(cur_height / ratio)
        resized = img.resize((resized_width, resized_height), resample=Image.BILINEAR)
        img = Image.new(resized.mode, (width, height), 0)
        img.paste(resized, (max(0, int((width - resized_width) / 2)), max(0, int((height - resized_height) / 2))))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def test_a_str_is_returned_unchanged_and_bytes_are_only_base64_encoded():
    assert encode("aGVsbG8=") == "aGVsbG8="
    assert encode(b"hello") == base64.b64encode(b"hello").decode()


def test_a_frame_is_encoded_at_its_own_size():
    """A frame is encoded at its own size: resizing it is the caller's decision."""
    assert _decoded(encode(_frame(1080, 1920))).size == (1920, 1080)
    assert _decoded(encode(_frame(126, 224))).size == (224, 126)


def test_the_codec_is_always_jpeg():
    assert _decoded(encode(_frame(64, 64))).format == "JPEG"
    assert DEFAULT_JPEG_QUALITY == 95


@pytest.mark.parametrize("size", [(224, 224), (320, 180), (640, 360), (320, 240), (256, 256)])
@pytest.mark.parametrize("shape", [(720, 1280), (480, 640), (1280, 720), (256, 256), (100, 1000)])
def test_pad_matches_openpis_resize_with_pad_byte_for_byte(size, shape):
    """Common policy input sizes, from landscape, portrait, square and extreme-aspect frames."""
    frame = _frame(*shape)
    assert encode(resize_image(frame, *size, mode="pad", filter="bilinear")) == _template_encode(frame, *size)


def test_stretch_fills_the_box_and_a_frame_already_that_size_is_not_resampled():
    stretched = resize_image(_frame(480, 640), 320, 180, mode="stretch", filter="lanczos")
    assert stretched.size == (320, 180) and np.asarray(stretched)[0].any()      # no black band: filled, not padded
    same = Image.fromarray(_frame(224, 224, seed=2))
    assert resize_image(same, 224, 224, mode="pad", filter="bilinear").tobytes() == same.tobytes()


def test_mode_and_filter_have_no_default():
    """With a default filter, LANCZOS frames once reached a policy trained on BILINEAR ones."""
    with pytest.raises(TypeError):
        resize_image(_frame(64, 64), 32, 32)                                     # type: ignore[call-arg]
    with pytest.raises(ValueError, match="mode must be one of"):
        resize_image(_frame(64, 64), 32, 32, mode="crop", filter="bilinear")
    with pytest.raises(ValueError, match="filter must be one of"):
        resize_image(_frame(64, 64), 32, 32, mode="pad", filter="BILINEAR")
    assert MODES == ("pad", "stretch") and FILTERS == ("nearest", "box", "bilinear", "hamming", "bicubic", "lanczos")


def test_every_kind_of_frame_is_accepted():
    """A PIL image, encoded bytes, an array, and base64 text."""
    frame = _frame(480, 640, seed=1)
    pil = Image.fromarray(frame)
    png = io.BytesIO()
    pil.save(png, format="PNG")
    want = resize_image(frame, 224, 224, mode="pad", filter="bilinear").tobytes()
    for kind in (pil, pil.convert("RGBA"), png.getvalue(), base64.b64encode(png.getvalue()).decode(),
                 "data:image/png;base64," + base64.b64encode(png.getvalue()).decode()):
        out = resize_image(kind, 224, 224, mode="pad", filter="bilinear")
        assert out.mode == "RGB" and out.tobytes() == want


def test_image_size_reads_a_jpeg_or_png_header():
    img = Image.fromarray(_frame(360, 640))
    exif = Image.Exif()
    exif[0x010E] = "x" * 20000                      # a metadata segment longer than the 8 KB prefix decoded first
    shots = []
    for save in ({}, {"progressive": True}, {"exif": exif.tobytes()}):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95, **save)
        shots.append(buf.getvalue())
    png = io.BytesIO()
    img.save(png, format="PNG")
    for raw in (*shots, png.getvalue()):
        assert image_size(base64.b64encode(raw).decode()) == (640, 360)
    assert image_size("data:image/jpeg;base64," + base64.b64encode(shots[0]).decode()) == (640, 360)


def test_image_size_refuses_what_is_not_an_image():
    """It also returns the size a header declares even when huge, which Pillow would refuse as a decompression bomb."""
    huge = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", 60000, 60000) + b"\x08\x02\0\0\0"
    assert image_size(base64.b64encode(huge).decode()) == (60000, 60000)
    gif = io.BytesIO()
    Image.fromarray(_frame(8, 8)).save(gif, format="GIF")
    jpeg = io.BytesIO()
    Image.fromarray(_frame(8, 8)).save(jpeg, format="JPEG")
    for data in ("aGVsbG8gd29ybGQ=", base64.b64encode(jpeg.getvalue()[:4]).decode(),
                 base64.b64encode(gif.getvalue()).decode(), ""):
        with pytest.raises(ValueError):
            image_size(data)
