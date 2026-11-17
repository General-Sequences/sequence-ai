# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
msgpack_io, alone: observations and responses as msgpack, arrays as their raw bytes.

    python examples/components/msgpack_io.py
"""
import numpy as np

from sequence_ai.network import msgpack_io

observation = {"image": np.zeros((224, 224, 3), np.uint8), "state": np.arange(7, dtype=np.float32), "prompt": "pick"}
body = msgpack_io.pack(observation)
back = msgpack_io.unpack(body)
print(f"{len(body):,} bytes for an image of {observation['image'].nbytes:,}; content type {msgpack_io.MEDIA_TYPE}")
assert np.array_equal(back["state"], observation["state"]) and back["image"].dtype == np.uint8
assert back["prompt"] == "pick"
print("arrays returned as arrays:", back["image"].shape, back["state"].dtype)
