# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
wifi_power, alone: is power saving on for the Wi-Fi interface of this machine's default route?

    python examples/components/wifi_power.py        # warns on Linux, on Wi-Fi, with power saving on
"""
from sequence_ai.network import wifi_power

iface = wifi_power.check()                                     # warns once, with how to disable it; otherwise nothing
print("Wi-Fi power saving is on:", iface if iface else "no (or this is not Linux, or not Wi-Fi)")
