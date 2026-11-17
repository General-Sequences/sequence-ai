# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
A connection's send window kept through a robot's pauses (sequence_ai/network/window.py): BBR and TCP keepalive are each
tried on a separate socket, and an option the machine rejects is left out; BBR is set on the caller's own socket; a
machine that restarts a window after a pause and cannot use BBR is warned once, with how to keep it; a connection is
pinged while it is idle after a send, and not before a send or long after one.

    python -m pytest -q tests/network/test_window.py
"""
from __future__ import annotations

import errno
import socket
import threading
import time
import warnings
from types import SimpleNamespace

from sequence_ai.network import window


def _probe(tried, refuse=(), fail=None):
    class Probe:
        def __init__(self, *a):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def setsockopt(self, level, name, value):
            tried.append((level, name, value))
            if fail is not None or name in refuse:
                raise OSError(fail or errno.ENOPROTOOPT, "refused")
    return Probe


def test_bbr_is_tried_on_a_separate_socket_and_anything_but_success_is_none(monkeypatch):
    """An HTTP client fails the whole connect on a socket option it cannot set, so the option is tried on a separate
    socket first: ENOENT (tcp_bbr not loaded, Ubuntu's default), EPERM, or no TCP_CONGESTION at all (not Linux) gives
    None."""
    tried = []

    def socket_module(fail):
        return SimpleNamespace(socket=_probe(tried, fail=fail), AF_INET=socket.AF_INET, SOCK_STREAM=socket.SOCK_STREAM,
                               IPPROTO_TCP=socket.IPPROTO_TCP)

    monkeypatch.setattr(window, "_TCP_CONGESTION", 13)
    for fail in (errno.ENOENT, errno.EPERM):
        monkeypatch.setattr(window, "socket", socket_module(fail))
        assert window.bbr_options() is None
    monkeypatch.setattr(window, "socket", socket_module(None))
    assert window.bbr_options() == [(socket.IPPROTO_TCP, 13, b"bbr")]
    assert tried == [(socket.IPPROTO_TCP, 13, b"bbr")] * 3
    monkeypatch.setattr(window, "_TCP_CONGESTION", None)
    assert window.bbr_options() is None and len(tried) == 3, "nothing to try where there is no TCP_CONGESTION"


def test_keepalive_options_are_tried_on_a_separate_socket_and_one_rejected_is_left_out(monkeypatch):
    """Each is tried on a separate socket first; one the machine rejects is left out and the rest are kept (macOS names
    the idle time TCP_KEEPALIVE), and none at all without SO_KEEPALIVE."""
    tried = []

    def socket_module(refuse, **names):
        return SimpleNamespace(socket=_probe(tried, refuse=refuse), AF_INET=socket.AF_INET,
                               SOCK_STREAM=socket.SOCK_STREAM, IPPROTO_TCP=6, SOL_SOCKET=0xFFFF, SO_KEEPALIVE=8,
                               **names)

    monkeypatch.setattr(window, "socket", socket_module({0x102}, TCP_KEEPALIVE=0x10, TCP_KEEPINTVL=0x101,
                                                        TCP_KEEPCNT=0x102))
    assert window.keepalive_options() == [(0xFFFF, 8, 1), (6, 0x10, 20), (6, 0x101, 5)]
    monkeypatch.setattr(window, "socket", socket_module(set(), TCP_KEEPIDLE=4, TCP_KEEPINTVL=5, TCP_KEEPCNT=6))
    assert window.keepalive_options() == [(0xFFFF, 8, 1), (6, 4, 20), (6, 5, 5), (6, 6, 3)]
    assert window.keepalive_options(idle_s=7, every_s=2, probes=9) == [(0xFFFF, 8, 1), (6, 4, 7), (6, 5, 2), (6, 6, 9)]
    tried.clear()
    monkeypatch.setattr(window, "socket", socket_module({8}, TCP_KEEPIDLE=4))
    assert window.keepalive_options() == [] and [name for _, name, _ in tried] == [8]


def test_the_socket_options_are_bbr_where_available_and_keepalive(monkeypatch):
    monkeypatch.setattr(window, "bbr_options", lambda: [("tcp", "congestion", b"bbr")])
    monkeypatch.setattr(window, "keepalive_options", lambda: [("sol", "keepalive", 1)])
    assert window.socket_options() == [("tcp", "congestion", b"bbr"), ("sol", "keepalive", 1)]
    monkeypatch.setattr(window, "bbr_options", lambda: None)
    assert window.socket_options() == [("sol", "keepalive", 1)]


def test_bbr_is_set_on_the_callers_own_socket_and_its_congestion_control_is_read_back(monkeypatch):
    class Sock:
        def __init__(self, refuse=False):
            self.refuse, self.control = refuse, b"cubic\0\0\0"

        def setsockopt(self, level, name, value):
            if self.refuse:
                raise OSError(errno.ENOENT, "no tcp_bbr")
            self.control = value + b"\0"

        def getsockopt(self, level, name, size):
            return self.control

    monkeypatch.setattr(window, "_TCP_CONGESTION", 13)
    took, refused = Sock(), Sock(refuse=True)
    assert window.keep(took) is True and window.congestion_of(took) == "bbr"
    assert window.keep(refused) is False and window.congestion_of(refused) == "cubic"
    assert window.congestion_of(None) is None
    monkeypatch.setattr(window, "_TCP_CONGESTION", None)                    # not Linux: nothing to set or to read
    assert window.keep(Sock()) is False and window.congestion_of(Sock()) is None


def test_a_machine_that_restarts_a_window_after_a_pause_is_warned_once_with_how_to_keep_it(tmp_path, monkeypatch):
    setting = tmp_path / "tcp_slow_start_after_idle"
    monkeypatch.setattr(window, "_SLOW_START_AFTER_IDLE", str(setting))
    monkeypatch.setattr(window, "_WARNED", window.threading.Event())
    monkeypatch.setattr(window, "bbr_options", lambda: None)
    assert window.restarts_after_idle() is False and window.check() is False       # unreadable (not Linux): no warning
    setting.write_text("1\n")
    assert window.restarts_after_idle() is True
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        assert window.check(window.BIG_UPLOAD) is False                             # an upload too small to matter
        assert window.check(2_400_000) is True
        assert window.check(2_400_000) is False                                     # once per process
    messages = [str(w.message) for w in record]
    assert len(messages) == 1 and "an upload of 2.4 MB" in messages[0] and "tcp_slow_start_after_idle=0" in messages[0]
    assert record[0].filename == __file__                                            # the caller's own line

    for kept in ("0\n", "1\n"):                                                      # kept by the setting, or by BBR
        setting.write_text(kept)
        monkeypatch.setattr(window, "_WARNED", window.threading.Event())
        monkeypatch.setattr(window, "bbr_options", (lambda: None) if kept == "0\n" else (lambda: [("bbr",)]))
        assert window.check() is False


def test_a_connection_is_pinged_while_idle_after_a_send_and_not_before_one_or_long_after():
    now, pings = [100.0], []
    warm = window.Warm(lambda: pings.append(now[0]), every_s=0.1, for_s=60.0, clock=lambda: now[0])
    warm._stop.set()                                    # no thread here: each iteration is run by hand
    assert warm.due() is False and warm.tick() is True and pings == []          # never sent on: never pinged
    warm.used()
    now[0] += 0.05
    assert warm.due() is False                                                   # sent on a moment ago
    now[0] += 0.06
    assert warm.tick() is True and pings == [now[0]]                             # idle for every_s: a ping
    now[0] += 0.1
    assert warm.tick() is True and len(pings) == 2 and warm.pings == 2           # and again, while it stays idle
    warm.used()
    assert warm.due() is False                                                   # a send postpones the next ping
    now[0] += 60.5
    assert warm.tick() is True and len(pings) == 2                               # long after its last send: none


def test_a_ping_that_raises_stops_the_pings_and_a_real_thread_pings_until_closed():
    def gone():
        raise ConnectionError("closed")

    dead = window.Warm(gone, every_s=0.1, clock=lambda: 5.0)
    dead._stop.set()
    dead._used_at = 4.0
    assert dead.tick() is False and dead.pings == 0

    pinged = window.threading.Event()
    warm = window.Warm(pinged.set, every_s=0.01)
    warm.used()
    assert pinged.wait(2.0)
    warm.close()
    warm._thread.join(2.0)
    assert not warm._thread.is_alive()


def test_close_waits_for_a_ping_in_flight_and_none_follows():
    started, release = threading.Event(), threading.Event()

    def slow_ping():
        started.set()
        release.wait(5.0)                      # a ping that is still waiting for its pong

    warm = window.Warm(slow_ping, every_s=0.01)
    warm.used()
    assert started.wait(2.0)                   # a ping is in flight
    threading.Timer(0.1, release.set).start()  # its pong arrives 100 ms later
    warm.close()
    in_flight = warm.pings
    assert in_flight == 1                      # close() returned after that ping ended, not during it
    time.sleep(0.1)
    assert warm.pings == in_flight             # and no ping follows a close
