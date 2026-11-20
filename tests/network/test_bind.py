# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
network.bind: a connection that leaves by one network interface - the socket option for each system and address family,
an interface this machine has not named with the ones it has, an httpx transport opening its sockets by its own backend
(a follow sent by one of a robot's lines), and, on a Mac, a socket bound to an interface before it connects - an httpx
one's and a robot's stream's alike (one bound to the Wi-Fi does not reach the loopback).

    python -m pytest -q tests/network/test_bind.py
"""
from __future__ import annotations

import socket
import sys

import httpx
import pytest

from sequence_ai._core import worker_stream
from sequence_ai.network import bind


def test_linux_binds_by_name_for_either_family():
    for family in (socket.AF_INET, socket.AF_INET6):
        [(level, option, value)] = bind.options("wwan0", family, platform="linux")
        assert (level, value) == (socket.SOL_SOCKET, b"wwan0")
        assert option == getattr(socket, "SO_BINDTODEVICE", 25)


def test_a_mac_binds_by_index_with_the_option_of_the_family(monkeypatch):
    monkeypatch.setattr(socket, "if_nametoindex", lambda name: {"en0": 11}[name])
    assert bind.options("en0", socket.AF_INET, platform="darwin") == [(socket.IPPROTO_IP, 25, 11)]
    assert bind.options("en0", socket.AF_INET6, platform="darwin") == [(socket.IPPROTO_IPV6, 125, 11)]


def test_another_system_is_refused_saying_so():
    with pytest.raises(OSError, match="on Linux or macOS, and this is win32"):
        bind.options("eth0", socket.AF_INET, platform="win32")


def test_an_interface_this_machine_has_not_is_named_with_the_ones_it_has():
    loopback = socket.if_nameindex()[0][1]
    with pytest.raises(OSError, match=f"no network interface 'nope0' on this machine: it has .*{loopback}"):
        bind.check("nope0")


def test_a_bound_transport_opens_its_sockets_by_its_own_backend():
    """httpx 0.28's transport keeps its pool as _pool and the pool opens connections with its _network_backend: a name
    either changes, and the binding is not where bind.transport puts it."""
    made = bind.transport(httpx, socket.if_nameindex()[0][1], limits=httpx.Limits(max_connections=1))
    assert type(made._pool._network_backend).__name__ == "Bound"
    assert made._pool._network_backend.interface == socket.if_nameindex()[0][1]


@pytest.mark.skipif(sys.platform != "darwin", reason="a Mac's lo0 and en0")
def test_on_a_mac_the_binding_comes_before_connect():
    """Bound to the loopback, a connection reaches a server on 127.0.0.1; bound to the Wi-Fi, it does not: the route was
    the interface's, chosen as it connected. Bound after connecting (httpcore's socket_options), both would reach it."""
    names = {name for _, name in socket.if_nameindex()}
    if "en0" not in names:
        pytest.skip("no en0 here")
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(4)
    port = listener.getsockname()[1]
    try:
        backend = bind.transport(httpx, "lo0")._pool._network_backend
        backend.connect_tcp("127.0.0.1", port, timeout=2.0).close()
        with pytest.raises(Exception) as refused:
            bind.transport(httpx, "en0")._pool._network_backend.connect_tcp("127.0.0.1", port, timeout=2.0)
        assert "by en0" in str(refused.value)
    finally:
        listener.close()


@pytest.mark.skipif(sys.platform != "darwin", reason="a Mac's lo0 and en0")
def test_on_a_mac_a_robots_stream_is_bound_before_it_connects():
    """A robot's stream by one of its lines (worker_stream.open_stream bind=): bound to the loopback, its connection
    reaches a server on 127.0.0.1; bound to the Wi-Fi, it does not, and the error names the interface."""
    names = {name for _, name in socket.if_nameindex()}
    if "en0" not in names:
        pytest.skip("no en0 here")
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(4)
    port = listener.getsockname()[1]
    try:
        worker_stream._connect_by("lo0", "127.0.0.1", port, 2.0).close()
        with pytest.raises(OSError) as refused:
            worker_stream._connect_by("en0", "127.0.0.1", port, 2.0)
        assert "by en0" in str(refused.value)
    finally:
        listener.close()

