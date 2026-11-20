# Third-party notices

SequenceLink (the `sequence-ai` package) is licensed under the Apache License, Version 2.0. Parts of four of its components, and one of its examples,
are adapted from two other open-source projects, and the page `seq eval replay` serves carries other projects' files unchanged. Each is named here
with its license.

## openpi

- **Where:** `sequence_ai/network/frames.py`, `resize_image` with `mode="pad"`, and the reference it is compared with in
  `tests/network/test_frames.py`; `sequence_ai/network/openpi.py`, `image_tools.convert_to_uint8` for an array.
- **From:** `_resize_with_pad_pil` and `convert_to_uint8` in
  `packages/openpi-client/src/openpi_client/image_tools.py` of
  [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi).
- **Why:** so that a frame resized here is, byte for byte, the frame openpi's own client sends.
- **Where:** `examples/serve_openpi_batched.py`, `batched`.
- **From:** `Policy.infer` in `src/openpi/policies/policy.py` of
  [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi).
- **Why:** so that each observation of a batch passes through the same transforms, and its output is the dict, that
  openpi's own `infer()` gives it.
- **License:** Apache License, Version 2.0, the same as this project's [LICENSE](LICENSE).

## InstinctFlash

- **Where:** `sequence_ai/engine/fused.py`, the norm, gated-activation and rotary-embedding kernels.
- **From:** `instinctflash/kernels/norm.py`, `instinctflash/kernels/activation.py` and
  `instinctflash/kernels/rope.py` of [General-Instinct/InstinctFlash](https://github.com/General-Instinct/InstinctFlash),
  at commit 34e1c60.
- **Why:** so that a norm, a gated activation and a rotary embedding each run as one kernel. Changed as the file's
  docstring says: the norm without the residual update, quantization and routing fused with it there; the gated
  activation reading two tensors and writing its result, with exact GELU added; the rotary embedding taking a model's
  own angle tables.
- **Where:** `sequence_ai/engine/fp8.py`, the E4M3 packing kernels and the recipe of `matmul`.
- **From:** `instinctflash/runtime/fp8_pack.py` and `instinctflash/runtime/torch_fp8_linear.py` of
  [General-Instinct/InstinctFlash](https://github.com/General-Instinct/InstinctFlash), at commit 34e1c60.
- **Why:** so that an input is packed to E4M3 with one scale per tensor and rounded as torch's own conversion rounds
  it. Changed as the file's docstring says: any plain linear layer of any float dtype, its forward replaced and the
  layer kept, with a calibration of which layers to run in E4M3.
- **License:** Apache License, Version 2.0, the same as this project's [LICENSE](LICENSE). The repository carries no
  NOTICE file.

## flashpack

- **Where:** `sequence_ai/engine/load.py`, its file of tensors grouped by dtype in aligned blocks, and its reads of
  that file in parallel into pinned buffers copied to the GPU on streams of their own.
- **From:** `src/flashpack/parallel_read.py`, `src/flashpack/serialization.py` and `src/flashpack/constants.py` of
  [fal-ai/flashpack](https://github.com/fal-ai/flashpack), at commit d1db2a8f.
- **Why:** so that a model's weights are read in a few long reads of one contiguous file, in parallel. Changed as the
  file's docstring says: its own header and layout; whether to read through the page cache decided for each chunk, from
  the pages the kernel holds, in place of once for the whole file; JAX arrays, put on the device from memory; checkpoints read
  from safetensors, .pth and orbax.
- **License:** the MIT License, whose text and copyright notice follow.

```text
MIT License

Copyright (c) 2025 fal - Features and Labels, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Files the replay page carries unchanged

`seq eval replay` serves a page from `seq/_replay_player/player/`. Beside this project's own files it holds:

| Files | From | License |
|---|---|---|
| `assets/vendor/three/` | [three.js](https://github.com/mrdoob/three.js), Copyright © 2010-2026 three.js authors | MIT, in `assets/vendor/three/LICENSE.txt` |
| `assets/vendor/fzstd/` | [fzstd](https://github.com/101arrowz/fzstd), Copyright (c) 2020 Arjun Barrett | MIT, in `assets/vendor/fzstd/LICENSE.txt` |
| `assets/fonts/commissioner-*.woff2` | Copyright 2019 The Commissioner Project Authors (https://github.com/kosbarts/Commissioner) | SIL Open Font License 1.1, in `seq/_replay_player/licenses/OFL-commissioner.txt` |
| `assets/fonts/geist-*.woff2` | Copyright 2024 The Geist Project Authors (https://github.com/vercel/geist-font.git) | SIL Open Font License 1.1, in `seq/_replay_player/licenses/OFL-geist.txt` |
| `assets/fonts/jetbrains-mono-*.woff2` | Copyright 2020 The JetBrains Mono Project Authors (https://github.com/JetBrains/JetBrainsMono) | SIL Open Font License 1.1, in `seq/_replay_player/licenses/OFL-jetbrains-mono.txt` |

Each font's copyright notice and the full text of the SIL Open Font License, Version 1.1, travel with the fonts in
`seq/_replay_player/licenses/`, as the license asks; it is also at <https://openfontlicense.org>.
