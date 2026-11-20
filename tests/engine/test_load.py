# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
Weights packed into one contiguous file and read back in parallel (load.py). On the CPU, each test pins one guarantee:
  every tensor comes back byte for byte - its name, dtype and shape - as torch tensors, numpy arrays or JAX arrays;
  floats packed in a dtype given (bfloat16) come back as what that cast gives;
  blocks start 4096-aligned, and every chunk after a block's head starts aligned for O_DIRECT;
  a chunk the page cache holds is read through it and one it does not hold is not, each decided apart;
  a torch checkpoint and a safetensors one are read with their names and packed.

    python -m pytest -q tests/engine/test_load.py
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from sequence_ai.engine import load  # noqa: E402


def _state():
    torch.manual_seed(0)
    return {"encoder.weight": torch.randn(64, 33), "encoder.bias": torch.randn(33).to(torch.bfloat16),
            "steps": torch.arange(7, dtype=torch.int64), "mask": torch.tensor([True, False, True]),
            "scalar": torch.tensor(1.5, dtype=torch.float16), "empty": torch.zeros(0, 4)}


def _same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.reshape(-1).view(torch.uint8),
                                                                     b.reshape(-1).view(torch.uint8))


def test_every_tensor_comes_back_byte_for_byte(tmp_path):
    state = _state()
    path = tmp_path / "model.load"
    head = load.pack(state, path)
    assert all(block["offset"] % load.ALIGN == 0 for block in head["blocks"])
    got = load.load(path, device="cpu", threads=4, chunk=load.ALIGN)          # many small chunks, many threads
    assert set(got) == set(state) and all(_same(got[name], state[name]) for name in state)
    arrays = load.load(path, to="numpy", threads=3)
    assert np.array_equal(arrays["encoder.weight"], state["encoder.weight"].numpy())
    assert str(arrays["encoder.bias"].dtype) == "bfloat16" and arrays["steps"].tolist() == list(range(7))


def test_floats_packed_in_a_dtype_given_come_back_as_that_cast_gives(tmp_path):
    state = _state()
    load.pack(state, tmp_path / "bf16.load", dtype="bfloat16")
    got = load.load(tmp_path / "bf16.load", device="cpu")
    assert got["encoder.weight"].dtype == torch.bfloat16
    assert _same(got["encoder.weight"], state["encoder.weight"].to(torch.bfloat16))
    assert got["steps"].dtype == torch.int64 and got["mask"].dtype == torch.bool       # only floats are cast


def test_jax_arrays_come_back_from_the_same_bytes_on_the_device_asked(tmp_path):
    jax = pytest.importorskip("jax")
    state = _state()
    load.pack(state, tmp_path / "model.load")
    got = load.load(tmp_path / "model.load", device="cpu", to="jax")
    assert isinstance(got["encoder.weight"], jax.Array) and got["encoder.weight"].devices() == {jax.devices("cpu")[0]}
    assert np.array_equal(np.asarray(got["encoder.weight"]), state["encoder.weight"].numpy())


def test_buffers_in_memory_start_aligned_for_o_direct():
    assert all(load._aligned(size).ctypes.data % load.ALIGN == 0 for size in (1, 4096, 5 << 20, (64 << 20) + 3))
    assert load._aligned(5000).nbytes == 5000


def test_every_chunk_after_a_blocks_head_starts_aligned():
    head = {"blocks": [{"offset": 4096 * 3 + 100, "bytes": 10_000}, {"offset": 4096 * 8, "bytes": 9000}]}
    chunks = load._plan(head, 4096)
    assert chunks[0] == (0, 4096 * 3 + 100, 0, 4096 - 100)                    # the head, up to the boundary
    assert all(offset % 4096 == 0 for _, offset, _, _ in chunks[1:])
    for block in (0, 1):
        assert sum(n for b, _, _, n in chunks if b == block) == head["blocks"][block]["bytes"]


def test_every_chunk_goes_through_the_page_cache_where_memory_holds_what_it_lacks_else_only_where_it_is_there(
        monkeypatch, tmp_path):
    if not hasattr(load.os, "O_DIRECT"):
        assert load._decide(str(tmp_path), [(0, 0, 0, 4096)]) == [True]        # no O_DIRECT here: all buffered
        monkeypatch.setattr(load.os, "O_DIRECT", 0o40000, raising=False)
    path = tmp_path / "f"
    path.write_bytes(b"\0" * (load._mmap.PAGESIZE * 8))
    pages = np.array([1, 1, 1, 1, 0, 0, 1, 0], dtype=np.uint8)               # the first half cached, then not
    monkeypatch.setattr(load, "_resident", lambda p, size: pages)
    size = load._mmap.PAGESIZE * 4
    chunks = [(0, 0, 0, size), (0, size, size, size)]
    monkeypatch.setattr(load, "_available", lambda: size + load.SPARE)       # memory holds the half it lacks
    assert load._decide(str(path), chunks) == [True, True]
    assert load._decide(str(path), chunks, host=1) == [True, False]           # not beside a buffer in memory too
    monkeypatch.setattr(load, "_available", lambda: 0)                        # no room: O_DIRECT where it is not there
    assert load._decide(str(path), chunks) == [True, False]
    monkeypatch.setattr(load, "_resident", lambda p, size: None)              # the kernel says nothing of its pages
    assert load._decide(str(path), chunks) == [False, False]
    monkeypatch.setattr(load, "_resident", lambda p, size: np.zeros(8, dtype=np.uint8))
    monkeypatch.setattr(load, "_sampled_gbps", lambda p, size: 20.0)         # "nothing cached", yet read at RAM speed
    assert load._decide(str(path), chunks) == [True, True]


def test_the_memory_the_kernel_can_give_is_read_where_it_says_it():
    got = load._available()
    assert got == 0 if not load.os.path.exists("/proc/meminfo") else got > 0


def test_a_torch_checkpoint_and_a_safetensors_one_are_read_with_their_names(tmp_path):
    state = _state()
    torch.save({"state_dict": {"net": {"encoder.weight": state["encoder.weight"]}, "head": state["steps"]}},
               tmp_path / "model.pth")
    got = load.read_checkpoint(tmp_path / "model.pth")
    assert set(got) == {"net.encoder.weight", "head"} and _same(got["head"], state["steps"])
    load.pack_from(tmp_path / "model.pth", tmp_path / "model.load")
    assert _same(load.load(tmp_path / "model.load", device="cpu")["net.encoder.weight"], state["encoder.weight"])
    safetensors = pytest.importorskip("safetensors.torch")
    folder = tmp_path / "hf"
    folder.mkdir()
    safetensors.save_file({"a": state["encoder.weight"]}, str(folder / "model-00001.safetensors"))
    safetensors.save_file({"b": state["encoder.bias"]}, str(folder / "model-00002.safetensors"))
    assert sorted(load.read_checkpoint(folder)) == ["a", "b"]


def test_a_file_that_is_not_packed_is_refused(tmp_path):
    (tmp_path / "x.load").write_bytes(b"not packed at all")
    with pytest.raises(ValueError, match="not a packed weights file"):
        load.load(tmp_path / "x.load", device="cpu")


def test_a_checkpoints_tree_comes_back_from_its_names():
    flat = {"params/llm/w": 1, "params/llm/b": 2, "params/head": 3}
    assert load.nest(flat) == {"params": {"llm": {"w": 1, "b": 2}, "head": 3}}
    assert load.nest({"a.b": 1}, ".") == {"a": {"b": 1}}
