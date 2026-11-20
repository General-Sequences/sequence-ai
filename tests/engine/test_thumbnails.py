# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
What a server shows of a robot's episode now (sequence_ai/engine/thumbnails.py): each image of its newest observation a
small JPEG, a robot-arm Observation's own JPEGs as they came, eight at most.

    python -m pytest -q tests/engine/test_thumbnails.py
"""
from __future__ import annotations

import base64
import io

import numpy as np
import pytest

from sequence_ai.engine import thumbnails

Image = pytest.importorskip("PIL.Image")


def _size(jpeg: str) -> tuple[int, int]:
    return Image.open(io.BytesIO(base64.b64decode(jpeg))).size


def test_each_image_of_an_observation_is_a_small_jpeg_and_a_robot_arm_ones_go_as_they_came():
    got = thumbnails.frames({"head": np.zeros((480, 640, 3), np.uint8), "depth_mm": np.zeros((480, 640), np.uint16),
                       "wrist": {"gray": np.full((96, 96), 128, np.uint8)}, "state": np.zeros(7, np.float32)})
    assert [(f["name"], f["width"], f["height"]) for f in got] == [("head", 320, 240), ("wrist.gray", 96, 96)]
    assert _size(got[0]["jpeg"]) == (320, 240)
    preset = thumbnails.frames({"images": [{"view": "exterior", "data": "/9j/AAAA", "width": 224, "height": 224}],
                          "proprio": {"joint_positions": [0.0] * 7}})
    assert preset == [{"name": "exterior", "jpeg": "/9j/AAAA", "width": 224, "height": 224}]
    many = thumbnails.frames({f"c{i}": np.zeros((8, 8, 3), np.uint8) for i in range(12)})
    assert len(many) == 8
