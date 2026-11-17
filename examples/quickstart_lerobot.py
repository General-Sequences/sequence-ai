# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
Quickstart for LeRobot users: the one Zenoh setting that keeps the connection of LeRobot 0.7's remote inference from
going idle.

    python examples/quickstart_lerobot.py

It writes zenoh-keep-alive.json5 in this folder and prints the setting to pass on each side. It needs neither LeRobot
nor Zenoh to run. docs/how-do-i-reduce-the-latency-of-lerobot-remote-inference.md lists what was verified and what was
not.
"""
from sequence_ai.network import lerobot

path = lerobot.keep_alive_file("zenoh-keep-alive.json5")       # an idle link then sends a frame every 100 ms
print(f"wrote {path}:")
print(open(path).read(), end="")
print("on the GPU, the policy server's setting:   zenoh.config_file =", path)
print("on the robot, remote inference's setting:  zenoh_config_path =", path)
