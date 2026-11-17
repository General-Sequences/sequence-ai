# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
Wi-Fi power saving on a robot's link, detected and reported (sequence_ai/network/wifi_power.py): when it is on for the
interface of the default route, a warning says how to disable it, and nothing is changed; when it is off, cannot be
read, or the link is not Wi-Fi, nothing is reported.

    python -m pytest -q tests/network/test_wifi_power.py
"""
from __future__ import annotations

import subprocess
import warnings

# ── Wi-Fi power saving on the interface the robot sends its requests over (Linux) ─────────────────────────────────────
ROUTES = ("Iface\tDestination\tGateway \tFlags\tRefCnt\tUse\tMetric\tMask\t\tMTU\tWindow\tIRTT\n"
          "eth0\t00000000\t0100A8C0\t0002\t0\t0\t100\t00000000\t0\t0\t0\n"         # a default route that is down
          "wlan0\t0001A8C0\t00000000\t0001\t0\t0\t600\t00FFFFFF\t0\t0\t0\n"        # not a default route
          "wlan0\t00000000\t0101A8C0\t0003\t0\t0\t600\t00000000\t0\t0\t0\n"
          "wlan1\t00000000\t0102A8C0\t0003\t0\t0\t700\t00000000\t0\t0\t0\n")


def _warnings(record, needle):
    return [str(w.message) for w in record if needle in str(w.message)]


def _machine(tmp_path, *wireless):
    route = tmp_path / "route"
    route.write_text(ROUTES)
    for iface in ("eth0", "wlan0", "wlan1"):
        (tmp_path / "net" / iface).mkdir(parents=True)
    for iface in wireless:
        (tmp_path / "net" / iface / "wireless").mkdir()
    return str(route), str(tmp_path / "net")


def _tools(answers):
    """Returns a `run` that responds to each tool's command with `answers[tool]` (raising it when it is an exception), a
    `tool` that finds each tool named in `answers`, and the list of commands run, in order."""
    ran = []

    def run(cmd, **kw):
        ran.append(cmd)
        out = answers[cmd[0].rsplit("/", 1)[-1]]
        if isinstance(out, BaseException):
            raise out
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    return run, (lambda name: f"/usr/sbin/{name}" if name in answers else None), ran


def test_wifi_power_saving_on_the_default_route_warns_with_how_to_disable_it(tmp_path, monkeypatch):
    """The machine's default route (the lowest metric among those that are up) uses wlan0, a Wi-Fi interface whose power
    saving `iw` reports as on: one warning is issued, with the command that disables it and the setting that keeps it
    disabled, and nothing is changed."""
    from sequence_ai.network import wifi_power

    route, net = _machine(tmp_path, "wlan0")
    assert wifi_power.default_iface(route) == "wlan0"
    run, tool, ran = _tools({"iw": "Power save: on\n"})
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        assert wifi_power.check(route, net=net, run=run, tool=tool) == "wlan0"
    assert ran == [["/usr/sbin/iw", "dev", "wlan0", "get", "power_save"]]          # a read: nothing set
    assert _warnings(record, "power saving") == [
        "Wi-Fi power saving is on for wlan0, so its adapter can delay this robot's packets while it sleeps; `sudo iw "
        "dev wlan0 set power_save off` disables it until the next reboot, and `wifi.powersave = 2` under [connection] "
        "in a file in /etc/NetworkManager/conf.d/ keeps it disabled"]


def test_wifi_power_saving_off_unreadable_or_on_a_link_that_is_not_wifi_reports_nothing(tmp_path):
    """When power saving is off nothing is reported; iwconfig is used where iw is not installed; a link that is not
    Wi-Fi, a tool that fails, neither tool installed, or a machine with no readable routes (not Linux) reports
    nothing."""
    from sequence_ai.network import wifi_power

    route, net = _machine(tmp_path, "wlan0")
    assert wifi_power.power_save_on("wlan0", net=net, run=_tools({"iw": "Power save: off\n"})[0],
                                    tool=_tools({"iw": ""})[1]) is False
    run, tool, ran = _tools({"iwconfig": "wlan0  IEEE 802.11  ESSID:\"home_user\"\n          Power Management:on\n"})
    assert wifi_power.power_save_on("wlan0", net=net, run=run, tool=tool) is True and ran[0][0] == "/usr/sbin/iwconfig"
    assert wifi_power.power_save_on("eth0", net=net, run=run, tool=tool) is None           # not Wi-Fi
    assert wifi_power.power_save_on("wlan0", net=net, run=_tools({"iw": OSError("no")})[0],
                                    tool=_tools({"iw": OSError("no")})[1]) is None
    assert wifi_power.power_save_on("wlan0", net=net, run=run, tool=lambda name: None) is None
    assert wifi_power.default_iface(str(tmp_path / "no-such-route")) is None
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        assert wifi_power.check(route, net=net, run=_tools({"iw": "Power save: off\n"})[0],
                                tool=_tools({"iw": ""})[1]) is None
    assert record == []
