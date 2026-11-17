# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
llms-full.txt is what tools/make_llms_full.py writes from the README and docs/ as they are now. A page changed
without it would leave a reader of that one file with yesterday's numbers.
"""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_llms_full_is_current():
    done = subprocess.run([sys.executable, str(ROOT / "tools" / "make_llms_full.py"), "--check"], capture_output=True,
                          text=True, timeout=60, cwd=ROOT)
    assert done.returncode == 0, done.stderr


def test_llms_full_links_no_file_by_a_relative_path():
    text = (ROOT / "llms-full.txt").read_text()
    assert "](../" not in text and "](docs/" not in text and "](examples/" not in text
