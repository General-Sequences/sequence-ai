# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
on_unreachable: called with the link when an act's request cannot reach the server (refused, or no answer to its
connect); True, and the act goes once more on a new connection - as a robot whose address changed has its GPU let it in
again and goes on. Never on a read timeout: that act may have reached the server.
"""
import socket
import threading

import httpx
import pytest

pytest.importorskip("uvicorn")

from sequence_ai.network import connect  # noqa: E402
from tests.network.test_link import serving  # noqa: E402
from tests.network.test_link_defaults import Server, observation  # noqa: E402


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def forward(port: int, target: int) -> socket.socket:
    """127.0.0.1:port forwarding each connection to `target`, from now on."""
    ls = socket.socket()
    ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ls.bind(("127.0.0.1", port))
    ls.listen(8)

    def pump(a, b):
        try:
            while True:
                data = a.recv(65536)
                if not data:
                    break
                b.sendall(data)
        except OSError:
            pass
        for x in (a, b):
            try:
                x.close()
            except OSError:
                pass

    def accept():
        while True:
            try:
                c, _ = ls.accept()
            except OSError:
                return
            s = socket.create_connection(("127.0.0.1", target))
            threading.Thread(target=pump, args=(c, s), daemon=True).start()
            threading.Thread(target=pump, args=(s, c), daemon=True).start()

    threading.Thread(target=accept, daemon=True).start()
    return ls


def test_an_act_that_cannot_reach_the_server_goes_once_more_after_on_unreachable_lets_it_in():
    server, port, called, held = Server(), free_port(), [], []
    with serving(server) as url:
        target = int(url.rsplit(":", 1)[1])

        def let_in(link):
            called.append(link)
            held.append(forward(port, target))       # the GPU lets the robot in again: the address answers now
            return True

        with connect(f"http://127.0.0.1:{port}", ping_path=None, check=False, on_unreachable=let_in) as link:
            assert link.act(observation())["image"].shape == (240, 320, 3)
            assert called == [link] and link.stats()["counters"]["unreachable_retries"] == 1
    for ls in held:
        ls.close()


def test_without_a_way_in_the_robot_meets_the_request_s_own_error_once_with_the_callback_s_beneath_it():
    port, called = free_port(), []
    with connect(f"http://127.0.0.1:{port}", ping_path=None, check=False,
                 on_unreachable=lambda link: called.append(1) or False) as link:
        with pytest.raises(httpx.ConnectError, match="no response from"):
            link.act(observation())
    assert called == [1]                                  # once an act
    with connect(f"http://127.0.0.1:{port}", ping_path=None, check=False,
                 on_unreachable=lambda link: called.append(2) or True) as link:
        with pytest.raises(httpx.ConnectError):
            link.act(observation())                       # sent once more, and refused again: said
    assert called == [1, 2]

    def broken(link):
        raise RuntimeError("the gateway did not answer")

    with connect(f"http://127.0.0.1:{port}", ping_path=None, check=False, on_unreachable=broken) as link:
        with pytest.raises(httpx.ConnectError) as refused:
            link.act(observation())
    assert isinstance(refused.value.__cause__, RuntimeError)


def test_wait_until_ready_asks_on_unreachable_while_nothing_answers():
    port, called = free_port(), []
    with connect(f"http://127.0.0.1:{port}", check=False, on_unreachable=lambda link: called.append(1) or False) as link:
        with pytest.raises(TimeoutError):
            link.wait_until_ready(timeout_s=0.5, poll_s=0.1)
    assert len(called) >= 2                               # once a round
