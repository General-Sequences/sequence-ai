# Third-party notices

`sequence-ai` is licensed under the Apache License, Version 2.0. One of its components is adapted from another
open-source project, and the page `seq eval replay` serves carries other projects' files unchanged. Each is named here
with its license.

## openpi

- **Where:** `sequence_ai/network/frames.py`, `resize_image` with `mode="pad"`, and the reference it is compared with in
  `tests/network/test_frames.py`.
- **From:** `_resize_with_pad_pil` in `packages/openpi-client/src/openpi_client/image_tools.py` of
  [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi).
- **Why:** so that a frame resized here is, byte for byte, the frame openpi's own client sends.
- **License:** Apache License, Version 2.0, the same as this project's [LICENSE](LICENSE).

## Files the replay page carries unchanged

`seq eval replay` serves a page from `seq/_replay_player/player/`. Beside this project's own files it holds:

| Files | From | License |
|---|---|---|
| `assets/vendor/three/` | [three.js](https://github.com/mrdoob/three.js), Copyright © 2010-2026 three.js authors | MIT, in `assets/vendor/three/LICENSE.txt` |
| `assets/vendor/fzstd/` | [fzstd](https://github.com/101arrowz/fzstd), Copyright (c) 2020 Arjun Barrett | MIT, in `assets/vendor/fzstd/LICENSE.txt` |
| `assets/fonts/commissioner-*.woff2` | Copyright 2019 The Commissioner Project Authors (https://github.com/kosbarts/Commissioner) | SIL Open Font License 1.1 |
| `assets/fonts/geist-*.woff2` | Copyright 2024 The Geist Project Authors (https://github.com/vercel/geist-font.git) | SIL Open Font License 1.1 |
| `assets/fonts/jetbrains-mono-*.woff2` | Copyright 2020 The JetBrains Mono Project Authors (https://github.com/JetBrains/JetBrainsMono) | SIL Open Font License 1.1 |

The SIL Open Font License, Version 1.1, is at <https://openfontlicense.org>.

## Acknowledgements

No code is taken from these; a design or a format is followed.

- **LiveKit Portal** (Apache-2.0): `latency_stats` follows the design of its metrics: microseconds, a rolling window of
  256 samples, one snapshot.
- **msgpack-numpy** and **openpi**: `msgpack_io` writes numpy arrays in msgpack-numpy's layout and reads openpi's as
  well.
