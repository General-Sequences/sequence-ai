# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
A model's norms, gated activations and rotary embeddings, each one kernel (fused.py). On the CPU the reference runs -
the arithmetic the kernels do, in torch - so each test pins what is replaced and that it computes what the original
did:
  a LayerNorm, an RMSNorm that scales by its weight and one that scales by 1 + its weight are each told apart by a
  probe and replaced; a norm of another kind is left alone;
  a gated MLP with SiLU or the tanh GELU is replaced, one with another activation is not;
  a code module's rotary function is replaced where the probe agrees, and the original runs the calls it does not cover;
  every replacement is taken back by uninstall, and a module whose forward is replaced already is left alone.

    python -m pytest -q tests/engine/test_fused.py
"""
from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from sequence_ai.engine import fused  # noqa: E402


class LlamaRMSNorm(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight, self.variance_epsilon = torch.nn.Parameter(torch.rand(width) + 0.5), 1e-6

    def forward(self, x):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        return self.weight * (x.float() * torch.rsqrt(variance + self.variance_epsilon)).to(x.dtype)


class GemmaRMSNorm(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.eps, self.weight = 1e-6, torch.nn.Parameter(torch.rand(width) - 0.5)

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * (1.0 + self.weight.float())).type_as(x)


class OddRMSNorm(LlamaRMSNorm):
    """Named like one, computing something else: the probe leaves it alone."""

    def forward(self, x):
        return super().forward(x) * 2 + 1


class GatedMLP(torch.nn.Module):
    def __init__(self, act) -> None:
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = (torch.nn.Linear(8, 16), torch.nn.Linear(8, 16),
                                                        torch.nn.Linear(16, 8))
        self.act_fn = act

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.ln, self.llama, self.gemma, self.odd = (torch.nn.LayerNorm(8), LlamaRMSNorm(8), GemmaRMSNorm(8),
                                                     OddRMSNorm(8))
        self.silu, self.gelu = GatedMLP(torch.nn.SiLU()), GatedMLP(torch.nn.GELU(approximate="tanh"))
        self.relu = GatedMLP(torch.nn.ReLU())

    def forward(self, x):
        for module in (self.ln, self.llama, self.gemma, self.odd, self.silu, self.gelu, self.relu):
            x = module(x)
        return x


def test_each_norm_and_gated_mlp_is_told_apart_by_its_probe_replaced_and_taken_back():
    model = Model().eval()
    x = torch.randn(4, 8)
    with torch.no_grad():
        want = model(x)
    patches = fused.install(model, say=lambda line: None)
    assert [p.what for p in patches] == ["ln: LayerNorm", "llama: RMSNorm", "gemma: RMSNorm (1 + weight)",
                                         "silu: gated MLP (silu)", "gelu: gated MLP (gelu_tanh)"]
    assert "forward" not in vars(model.odd) and "forward" not in vars(model.relu)
    with torch.no_grad():
        assert torch.allclose(model(x), want, atol=1e-5)
    fused.uninstall(patches)
    assert not any("forward" in vars(m) for m in model.modules())
    with torch.no_grad():
        assert torch.equal(model(x), want)


def test_a_module_whose_forward_is_replaced_already_is_left_alone():
    model = Model().eval()
    model.ln.forward = lambda x: x
    assert "ln: LayerNorm" not in [p.what for p in fused.install(model, say=lambda line: None)]


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def hf_rotary(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """A language model's apply_rotary_pos_emb, as its code module holds it (named apart here, so that this test file,
    the code module of the models above, holds none)."""
    cos, sin = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def test_a_code_modules_rotary_function_is_replaced_where_its_probe_agrees_and_covers_what_it_can():
    code = types.ModuleType("modeling_toy")
    code.apply_rotary_pos_emb = hf_rotary
    patch = fused._rotary_patch(code, torch.nn.Linear(2, 2))
    assert patch is not None and patch.before is hf_rotary
    q, k = torch.randn(2, 3, 5, 16), torch.randn(2, 3, 5, 16)
    cos, sin = torch.randn(2, 5, 16), torch.randn(2, 5, 16)
    for got, want in zip(patch.value(q, k, cos, sin), hf_rotary(q, k, cos, sin)):
        assert torch.allclose(got, want, atol=1e-6)
    q2, k2 = q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous()      # [B, T, H, D]: unsqueeze_dim=2
    for got, want in zip(patch.value(q2, k2, cos, sin, unsqueeze_dim=2),
                         hf_rotary(q2, k2, cos, sin, unsqueeze_dim=2)):
        assert torch.allclose(got, want, atol=1e-6)
    # position ids, which an older model indexes its angles by, go to the original
    calls = []
    code.apply_rotary_pos_emb = lambda q, k, cos, sin, **kw: calls.append(kw) or hf_rotary(q, k, cos, sin, **kw)
    patch = fused._rotary_patch(code, torch.nn.Linear(2, 2))
    patch.value(q, k, cos, sin, position_ids=torch.arange(5))
    assert calls and calls[-1].get("position_ids") is not None


def test_a_function_that_rotates_otherwise_is_left_alone():
    code = types.ModuleType("modeling_other")

    def apply_rotary_pos_emb(q, k, cos, sin):                 # pairs neighbours, not halves
        return q * cos, k * sin

    code.apply_rotary_pos_emb = apply_rotary_pos_emb
    assert fused._rotary_patch(code, torch.nn.Linear(2, 2)) is None


def test_the_references_compute_the_steps():
    x = torch.randn(3, 8)
    ln = torch.nn.LayerNorm(8)
    assert torch.allclose(fused.norm(x, ln.weight, ln.bias, eps=ln.eps, layernorm=True), ln(x), atol=1e-6)
    g, u = torch.randn(3, 8), torch.randn(3, 8)
    assert torch.allclose(fused.gated(g, u, fused.SILU), torch.nn.functional.silu(g) * u, atol=1e-6)
    assert torch.allclose(fused.gated(g, u, fused.GELU), torch.nn.functional.gelu(g) * u, atol=1e-6)
