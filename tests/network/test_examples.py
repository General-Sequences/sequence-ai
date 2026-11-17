# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
The examples run: each component's own, and the robot against the server. An example that stops running is a line in the
README that is no longer true.
"""
import importlib.util
import os
import pathlib
import socket
import subprocess
import sys
import time

import pytest

EXAMPLES = pathlib.Path(__file__).resolve().parents[2] / "examples"
NEEDS = {"frames": "PIL", "one_write": "uvicorn", "ping": "websockets", "compress": "numpy", "msgpack_io": "numpy"}


# the examples import the checkout under test, whatever is installed
ENV = {**os.environ,
       "PYTHONPATH": os.pathsep.join(filter(None, [str(EXAMPLES.parent), os.environ.get("PYTHONPATH")]))}


def _run(path, *args, timeout=60):
    return subprocess.run([sys.executable, str(path), *args], capture_output=True, text=True, timeout=timeout,
                          cwd=EXAMPLES.parent, env=ENV)


@pytest.mark.parametrize("name", sorted(p.stem for p in (EXAMPLES / "components").glob("*.py")))
def test_each_components_own_example_runs(name):
    need = NEEDS.get(name)
    if need and importlib.util.find_spec(need) is None:
        pytest.skip(f"needs {need}")
    done = _run(EXAMPLES / "components" / f"{name}.py")
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip()


def test_the_robot_runs_against_the_server():
    for need in ("uvicorn", "httpx", "numpy"):
        if importlib.util.find_spec(need) is None:
            pytest.skip(f"needs {need}")
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", 8000)) == 0:
            pytest.skip("something else is listening on 127.0.0.1:8000")
    server = subprocess.Popen([sys.executable, str(EXAMPLES / "server.py")], cwd=EXAMPLES.parent, env=ENV,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 20
        while time.time() < deadline:
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", 8000)) == 0:
                    break
            time.sleep(0.1)
        done = _run(EXAMPLES / "robot.py")
        assert done.returncode == 0, done.stderr[-2000:]
        assert "request 4: got (8, 7) actions" in done.stdout and "pings while idle:" in done.stdout
    finally:
        server.terminate()
        server.wait(10)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_the_quickstart_robot_gets_its_first_actions_from_the_quickstart_server():
    for need in ("uvicorn", "httpx", "numpy"):
        if importlib.util.find_spec(need) is None:
            pytest.skip(f"needs {need}")
    port = _free_port()
    server = subprocess.Popen([sys.executable, str(EXAMPLES / "quickstart_server.py"), "127.0.0.1", str(port)],
                              cwd=EXAMPLES.parent, env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 20
        while time.time() < deadline:
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(0.1)
        done = _run(EXAMPLES / "quickstart_robot.py", f"http://127.0.0.1:{port}")
        assert done.returncode == 0, done.stderr[-2000:]
        # the line docs/quickstart.md and the README show
        assert "request 0: 8 actions of 7 numbers, the first [128. 128. 128.]" in done.stdout
        assert "3 requests, the slowest took" in done.stdout
    finally:
        server.terminate()
        server.wait(10)


def test_the_openpi_quickstart_keeps_openpis_own_client():
    for need in ("openpi_client", "websockets", "numpy"):
        if importlib.util.find_spec(need) is None:
            pytest.skip(f"needs {need}")
    done = _run(EXAMPLES / "quickstart_openpi.py")
    assert done.returncode == 0, done.stderr[-2000:]
    assert "infer 2: a chunk of 10 actions" in done.stdout and "pings answered while it was idle:" in done.stdout


def test_the_lerobot_quickstart_writes_the_setting(tmp_path):
    done = subprocess.run([sys.executable, str(EXAMPLES / "quickstart_lerobot.py")], capture_output=True, text=True,
                          timeout=60, cwd=tmp_path, env=ENV)          # it writes its file where it is run
    assert done.returncode == 0, done.stderr[-2000:]
    assert '"keep_alive": 100' in (tmp_path / "zenoh-keep-alive.json5").read_text()
    assert "zenoh.config_file" in done.stdout and "zenoh_config_path" in done.stdout
