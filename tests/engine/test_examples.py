# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
The inference layer's examples run: the quickstart, graphing a module, and three components together. Each component's
own example is run by tests/network/test_examples.py. An example that stops running is a line in the README that
stopped being true.
"""
import importlib.util
import os
import pathlib
import subprocess
import sys

import pytest

EXAMPLES = pathlib.Path(__file__).resolve().parents[2] / "examples"
RUN = ["quickstart_batching.py", "graph_a_module.py", "serve_with_replicas.py", "fleet_on_a_clock.py"]
NEEDS_TORCH = {"graph_a_module.py", "serve_with_replicas.py"}


@pytest.mark.parametrize("name", RUN)
def test_the_example_runs(name):
    if name in NEEDS_TORCH and importlib.util.find_spec("torch") is None:
        pytest.skip("needs torch")
    paths = filter(None, [str(EXAMPLES.parent), os.environ.get("PYTHONPATH")])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(paths)}
    done = subprocess.run([sys.executable, str(EXAMPLES / name)], capture_output=True, text=True, timeout=300,
                          cwd=EXAMPLES.parent, env=env)      # the checkout under test, whatever is installed
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip()
