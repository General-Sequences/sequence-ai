# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
wifi_power on macOS, where power saving cannot be read: a robot whose requests go over Wi-Fi is told it was not checked.
"""
import subprocess
import threading

import pytest

from sequence_ai.network import wifi_power

ROUTE = "   route to: default\ndestination: default\n       mask: default\n    gateway: 192.0.2.1\n  interface: en0\n"
PORTS = ("\nHardware Port: Ethernet\nDevice: en1\nEthernet Address: 00:00:5e:00:53:01\n\n"
         "Hardware Port: Wi-Fi\nDevice: en0\nEthernet Address: 00:00:5e:00:53:02\n")


def answering(route: str, ports: str):
    def run(argv, **kw):
        out = route if argv[0].endswith("route") else ports
        return subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")
    return run


def test_the_wi_fi_device_is_found_when_the_default_route_goes_over_it_and_nothing_otherwise():
    assert wifi_power.mac_wifi_iface(run=answering(ROUTE, PORTS)) == "en0"
    assert wifi_power.mac_wifi_iface(run=answering(ROUTE.replace("en0", "en1"), PORTS)) is None    # wired

    def broken(argv, **kw):
        raise OSError("no such tool")

    assert wifi_power.mac_wifi_iface(run=broken) is None


def test_on_macos_a_robot_on_wi_fi_is_told_once_that_it_was_not_checked(monkeypatch):
    monkeypatch.setattr(wifi_power, "_STARTED", threading.Event())
    monkeypatch.setattr(wifi_power, "mac_wifi_iface", lambda: "en0")
    with pytest.warns(UserWarning, match=r"over Wi-Fi \(en0\).*not checked on this Mac"):
        thread = wifi_power.check_once(linux=False, darwin=True)
        thread.join(5)
    assert wifi_power.check_once(linux=False, darwin=True) is None            # once a process
    monkeypatch.setattr(wifi_power, "_STARTED", threading.Event())
    assert wifi_power.check_once(linux=False, darwin=False) is None           # neither: nothing started
