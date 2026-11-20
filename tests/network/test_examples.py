# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
The examples run: each component's own, and the quickstart's robot against its server. An example that stops running
is a line in the README that is no longer true.
"""
import importlib.util
import os
import pathlib
import re
import socket
import subprocess
import sys
import time

import pytest

EXAMPLES = pathlib.Path(__file__).resolve().parents[2] / "examples"
NEEDS = {"frames": "PIL", "one_write": "uvicorn", "ping": "websockets", "msgpack_io": "numpy"}


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


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_the_readmes_quick_start_runs_as_a_reader_copies_it(tmp_path):
    """README.md's Quick start: its policy class saved as my_policy_user/policy.py and served with `seq serve`, on a free port
    where it says 8000, and its robot's lines run against it get a chunk of 10 steps of 7 joints."""
    for need in ("uvicorn", "httpx", "numpy", "sequence_base", "av"):
        if importlib.util.find_spec(need) is None:
            pytest.skip(f"needs {need}")
    quick = (EXAMPLES.parent / "README.md").read_text().split("## Quick start", 1)[1].split("\n## ", 1)[0]
    policy, robot = re.findall(r"```python\n(.*?)```", quick, re.S)[:2]
    assert "@seq.sample" in policy and "gpu=" not in policy and "wait_until_ready()" in robot
    (tmp_path / "my_policy_user").mkdir()
    (tmp_path / "my_policy_user" / "policy.py").write_text(policy)
    port = _free_port()
    (tmp_path / "robot.py").write_text(robot.replace("8000", str(port)))
    server = subprocess.Popen([sys.executable, "-m", "seq", "serve", "my_policy_user", "--port", str(port)], cwd=tmp_path,
                              env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.time() + 30
        while time.time() < deadline and server.poll() is None:
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(0.1)
        if server.poll() is not None:
            said = server.stdout.read()
            if "requires gpu=" in said:
                pytest.xfail("@seq.policy without gpu= is served by a release in which gpu= is optional")
            raise AssertionError(said[-2000:])
        done = subprocess.run([sys.executable, str(tmp_path / "robot.py")], capture_output=True, text=True,
                              timeout=60, cwd=tmp_path, env=ENV)
        assert done.returncode == 0, done.stderr[-2000:]
        assert done.stdout.strip().splitlines()[-1] == "(10, 7)"
    finally:
        server.terminate()
        server.wait(10)


# a server answering as examples/quickstart_policy does: the frame's mean, 8 steps of 7 joints
SERVER = """import sys
import numpy as np
from sequence_ai.engine.serve import serve
def model(observation):
    return {"actions": np.full((8, 7), float(observation["image"].mean()), dtype=np.float32)}
serve(model, host="127.0.0.1", port=int(sys.argv[1]),
      example={"image": np.zeros((224, 224, 3), np.uint8), "state": np.zeros(7, np.float32)})
"""


def test_the_quickstart_robot_plays_its_servers_answers():
    for need in ("uvicorn", "httpx", "numpy"):
        if importlib.util.find_spec(need) is None:
            pytest.skip(f"needs {need}")
    port = _free_port()
    server = subprocess.Popen([sys.executable, "-c", SERVER, str(port)],
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
        # the lines docs/quickstart.md and the README show
        assert "the first answer: 8 steps of 7 joints, the first [128. 128. 128.]" in done.stdout
        assert "steps played at 15 a second; " in done.stdout
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
