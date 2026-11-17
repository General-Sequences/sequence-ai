# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
frames, alone: a camera frame resized to the size its policy takes, then encoded.

    python examples/components/frames.py            # needs Pillow
"""
import numpy as np

from sequence_ai.network import frames

yy, xx = np.mgrid[0:480, 0:640]
frame = np.stack([xx % 256, yy % 256, (xx + yy) % 256], -1).astype(np.uint8)

padded = frames.resize_image(frame, 224, 224, mode="pad", filter="bilinear")          # openpi's resize_with_pad
stretched = frames.resize_image(frame, 224, 224, mode="stretch", filter="bilinear")
print("pad:", np.asarray(padded).shape, "with black rows at the top:", bool((np.asarray(padded)[:20] == 0).all()))
print("stretch:", np.asarray(stretched).shape)

text = frames.encode(padded)                                   # JPEG, base64: what is uploaded
print(f"encoded: {len(text):,} characters for {frame.nbytes:,} bytes; size from its header: {frames.image_size(text)}")
