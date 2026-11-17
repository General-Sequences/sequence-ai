# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
bench/: the measuring tool, run end to end against its own server on this machine. What a block measures depends on the
link; what is checked here is that every block runs on the one connection and writes the result it documents.
"""
import importlib.util
import json
import pathlib
import socket
import struct
import sys
import threading

import pytest

BENCH = pathlib.Path(__file__).resolve().parents[2] / "bench"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"bench_{name}", BENCH / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def far_end():
    server = _load("server")
    srv = server.serve("127.0.0.1", 0)
    threading.Thread(target=server.accept_forever, args=(srv,), daemon=True).start()
    yield srv.getsockname()[1]
    srv.close()


def test_every_block_runs_on_one_connection_and_writes_its_result(far_end, tmp_path, capsys):
    client = _load("client")
    out = tmp_path / "result.json"
    res = client.main(["--host", "127.0.0.1", "--port", str(far_end), "--sizes", "10k", "--samples", "3",
                       "--idle-ms", "60", "--ping-ms", "10", "--stream", "50x2k", "--seconds", "0.2", "--warm", "100k",
                       "--modes", "idle,ping,keepwarm,continuous,stream,twowrite", "--out", str(out)])
    by = {r["mode"]: r for r in res["results"]}
    assert list(by) == ["idle", "ping", "keepwarm", "continuous", "stream", "twowrite"]
    assert all(by[m]["n"] == 3 and by[m]["size"] == 10_000 for m in ("idle", "ping", "keepwarm", "continuous"))
    assert by["ping"]["pongs"] >= 3                    # six silences of 60 ms, a ping every 10 ms
    assert by["keepwarm"]["sent_while_idle"] >= 3     # and this end's own byte during each of them
    assert by["stream"]["n"] == 10 and by["stream"]["mbps"] == 0.8
    assert by["twowrite"]["size"] == 4196              # 200 + 4000 bytes written, minus the 4 bytes of the length
    assert res["warm_ms"] > 0 and res["baseline"]["n"] == 3
    saved = json.loads(out.read_text())
    assert saved["finished_utc"] and len(saved["results"]) == 6
    table = capsys.readouterr().out
    assert "x rtt" in table and "50Hz" in table


def test_a_block_can_have_its_own_number_of_samples(far_end, tmp_path):
    client = _load("client")
    res = client.main(["--host", "127.0.0.1", "--port", str(far_end), "--sizes", "2k", "--samples", "2",
                       "--idle-ms", "5", "--modes", "continuous:5,idle", "--out", str(tmp_path / "r.json")])
    assert [(r["mode"], r["n"]) for r in res["results"]] == [("continuous", 5), ("idle", 2)]


def test_sizes_are_decimal_and_an_unknown_block_is_refused(far_end, tmp_path):
    client = _load("client")
    assert client.parse_size("100k") == 100_000 and client.parse_size("1m") == 1_000_000
    assert client.parse_size("42") == 42
    assert client.parse_stream("30x50k") == (30.0, 50_000) and client.parse_pair("200+4k") == (200, 4000)
    assert client.pct([1, 2, 3, 4], 50) == 2.5
    with pytest.raises(SystemExit):
        client.main(["--host", "127.0.0.1", "--port", str(far_end), "--modes", "idle,nonsense",
                     "--out", str(tmp_path / "r.json")])


def test_the_kernels_state_is_read_by_the_platform_not_by_the_name(monkeypatch):
    """Linux's struct tcp_info on Linux, macOS's struct tcp_connection_info on a Mac - even where the Mac's Python
    defines socket.TCP_INFO (its own Python 3.9 does, for another struct), and nothing elsewhere."""
    client = _load("client")
    linux = bytearray(160)
    u = {0: 210_000, 2: 1448, 9: 120, 15: 42_000, 17: 50, 18: 260, 23: 2}      # rto, mss, idle, rtt, ssthresh, cwnd
    for i, value in u.items():
        struct.pack_into("<I", linux, 8 + 4 * i, value)
    struct.pack_into("<I", linux, 148, 40_000)                                 # min_rtt
    mac = bytearray(112)
    struct.pack_into("<4I", mac, 12, 210, 1364, 1364 * 50, 1364 * 260)        # rto, maxseg, ssthresh, cwnd (bytes)
    struct.pack_into("<I", mac, 44, 42)                                         # srtt
    struct.pack_into("<Q", mac, 104, 2)                                         # retransmitted packets

    class Sock:
        def getsockopt(self, level, opt, size):
            return {11: bytes(linux), 0x200: b"\0" * size, 0x106: bytes(mac)}[opt][:size]     # 0x106: macOS's

    monkeypatch.setattr(socket, "TCP_INFO", 11, raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    assert client.tcp_info(Sock()) == {"cwnd": 260, "ssthresh": 50, "rto_ms": 210.0, "rtt_ms": 42.0, "idle_ms": 120,
                                       "mss": 1448, "retrans": 2, "min_rtt_ms": 40.0}
    monkeypatch.setattr(socket, "TCP_INFO", 0x200, raising=False)               # as /usr/bin/python3 3.9 defines it
    monkeypatch.setattr(sys, "platform", "darwin")
    assert client.tcp_info(Sock()) == {"cwnd": 260, "ssthresh": 50, "rto_ms": 210, "rtt_ms": 42, "idle_ms": None,
                                       "mss": 1364, "retrans": 2, "min_rtt_ms": None}
    monkeypatch.setattr(sys, "platform", "win32")
    assert client.tcp_info(Sock()) is None
