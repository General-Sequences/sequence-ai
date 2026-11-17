# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
compress and compressed, alone: an observation's frame compressed where that saves time, and decompressed byte for byte.

    python examples/components/compress.py
"""
import numpy as np

from sequence_ai.network import compress, compressed, msgpack_io

yy, xx = np.mgrid[0:480, 0:640]
frame = np.stack([(xx // 8) * 9, (yy // 6) * 7, ((xx + yy) // 16) * 11], -1).astype(np.uint8)
observation = {"image": frame, "state": np.zeros(7, dtype=np.float32)}

squeeze = compress.Squeeze()                                   # client side: one per connection
body = msgpack_io.pack(squeeze.apply(observation))
print(f"a frame of {frame.nbytes:,} bytes was uploaded as {len(body):,}")

back, grew = compressed.expand(msgpack_io.unpack(body), limit=64 << 20)     # server side
assert np.array_equal(back["image"], frame)
print(f"decompressed byte for byte: {grew:,} bytes larger than what arrived")

fast = compress.Squeeze()                                      # the same frame on a very fast link
fast.uplink.note(1_000, 0.0)                                   # a small POST: the RTT with almost no payload
fast.uplink.note(10_000_000, 0.002)                            # a large one: 5 GB per second
fast.apply(observation)                                        # the first is compressed, to measure its ratio
print("on that link the next frame is sent uncompressed:", fast.apply(observation) is observation)
