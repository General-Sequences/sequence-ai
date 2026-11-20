# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
Linear layers in E4M3 (fp8.py), on the CPU's reference: the same E4M3 values and scales the GPU uses, multiplied in
float32. Each test pins one guarantee:
  an input is packed with one scale, its largest magnitude / 448, and its values come back within E4M3's precision;
  a layer's fp8 output is close to its own; one whose input holds an outlier is measured as far, and keeps its own;
  the layers kept are written to the cache, and a second start with the same weights and key reads them without
  calibrating; uninstall puts every layer's own forward back;
  only plain linear layers whose sides are multiples of 16 are eligible; a machine with no GPU says why it has none.

    python -m pytest -q tests/engine/test_fp8.py
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
if not hasattr(torch, "float8_e4m3fn"):
    pytest.skip("this torch has no float8_e4m3fn", allow_module_level=True)

from sequence_ai.engine import fp8  # noqa: E402


def test_an_input_is_packed_with_one_scale_and_comes_back_within_e4m3s_precision():
    rows = torch.randn(8, 32).to(torch.bfloat16)
    packed, scale = fp8.pack(rows)
    assert packed.dtype == torch.float8_e4m3fn and scale.shape == (1,)
    assert torch.allclose(scale, rows.float().abs().amax().reshape(1) / 448.0)
    back = packed.float() * scale
    assert float(((back - rows.float()).abs() / rows.float().abs().clamp_min(1e-3)).median()) < 0.07


def test_a_layers_fp8_output_is_close_to_its_own():
    torch.manual_seed(0)
    layer = torch.nn.Linear(32, 16)
    x = torch.randn(5, 32)
    with torch.no_grad():
        got, want = fp8.matmul(fp8.Packed(layer), x), layer(x)
    assert got.dtype == torch.float32 and float((got - want).norm() / want.norm()) < 0.05


class Two(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.plain, self.spiky, self.odd = torch.nn.Linear(32, 32), torch.nn.Linear(32, 32), torch.nn.Linear(30, 32)
        with torch.no_grad():
            self.spiky.weight[:, 0] = 0.0

    def forward(self, x):
        y = self.plain(x)
        spikes = torch.zeros_like(y)
        spikes[:, 0] = 1e6                         # one channel far above the rest: one scale loses the others,
        return self.spiky(y + spikes)              # and the layer reads only the others (its column 0 is zero)


def test_a_layer_that_moves_in_fp8_keeps_its_own_and_the_calibration_is_read_back(tmp_path):
    model = Two().eval()
    x = torch.randn(6, 32)
    runs = []

    def calibrate():
        runs.append(1)
        with torch.no_grad():
            model(x)

    assert [path for path, _ in fp8.eligible(model)] == ["plain", "spiky"]      # 30 is no multiple of 16
    patches = fp8.install(model, calibrate=calibrate, key="k", cache=tmp_path, say=lambda line: None)
    assert [p.what for p in patches] == ["plain"] and runs == [1]
    assert "forward" in vars(model.plain) and "forward" not in vars(model.spiky)
    fp8.uninstall(patches)
    assert "forward" not in vars(model.plain)
    again = fp8.install(model, calibrate=calibrate, key="k", cache=tmp_path, say=lambda line: None)
    assert [p.what for p in again] == ["plain"] and runs == [1]                # read from the cache
    fp8.install(model, calibrate=calibrate, key="other examples", cache=tmp_path, say=lambda line: None)
    assert runs == [1, 1]                                                       # other examples: calibrated again


def test_with_no_gpu_fp8_says_why():
    if torch.cuda.is_available():
        pytest.skip("this machine has a GPU")
    assert fp8.unsupported() == "fp8 needs a CUDA GPU"
