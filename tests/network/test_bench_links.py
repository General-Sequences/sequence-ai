# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
bench/ws_bench.py and bench/zenoh_bench.py run: a server and a client on this machine, a few small uploads each. What
they measure needs a real link; this checks only that the two ends communicate and report the fields they document.
"""
import importlib.util
import json
import os
import pathlib
import socket
import subprocess
import sys
import time

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
# the scripts import the checkout under test, whatever is installed
ENV = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))}
SMALL = ["--size", "2000", "--pause", "0.05", "--samples", "3"]


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _listening(port: int, within_s: float = 20.0) -> None:
    deadline = time.time() + within_s
    while time.time() < deadline:
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise AssertionError(f"nothing listened on 127.0.0.1:{port} within {within_s:.0f} s")


def _measure(script: str, server_args: list, client_args: list) -> dict:
    port = _free_port()
    path = str(ROOT / "bench" / script)
    server = subprocess.Popen([sys.executable, path, "server", str(port), *server_args], cwd=ROOT, env=ENV,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        _listening(port)
        done = subprocess.run([sys.executable, path, "client", "127.0.0.1", str(port), *client_args, *SMALL],
                              capture_output=True, text=True, timeout=120, cwd=ROOT, env=ENV)
        assert done.returncode == 0, done.stderr[-2000:]
        line = next(row for row in done.stdout.splitlines() if row.startswith("RESULT "))
        return json.loads(line[len("RESULT "):])
    finally:
        server.terminate()
        server.wait(timeout=10)


@pytest.mark.parametrize("variant, ping", [("plain", False), ("plain", True), ("bbr", False)])
def test_the_websocket_bench_measures_each_variant(variant, ping):
    if importlib.util.find_spec("websockets") is None:
        pytest.skip("needs websockets")
    result = _measure("ws_bench.py", ["--ping"] if ping else [], [variant])
    assert result["variant"] == variant and result["n"] == 3 and len(result["samples_ms"]) == 3
    assert all(ms > 0 for ms in result["samples_ms"])


def test_the_zenoh_bench_measures_its_link():
    if importlib.util.find_spec("zenoh") is None:
        pytest.skip("needs eclipse-zenoh")
    result = _measure("zenoh_bench.py", [], [])
    assert result["n"] == 3 and all(ms > 0 for ms in result["samples_ms"])
