# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
lerobot: the Zenoh settings and the file they are written to. With Zenoh installed, the file is passed to two peers on
this machine, and the frames an idle link then sends are counted through a relay that timestamps them.
"""
import json
import socket
import statistics
import threading
import time

import pytest

from sequence_ai.network import lerobot


def test_the_settings_send_a_frame_every_interval_of_the_lease():
    assert lerobot.keep_alive_settings() == {"transport": {"link": {"tx": {"keep_alive": 100}}}}
    assert lerobot.keep_alive_settings(50, 2000) == {"transport": {"link": {"tx": {"keep_alive": 40, "lease": 2000}}}}
    for every_ms, lease_ms in ((0, 10_000), (100, 0), (20_000, 10_000)):
        with pytest.raises(ValueError):
            lerobot.keep_alive_settings(every_ms, lease_ms)


def test_the_file_keeps_settings_of_your_own_beside_them(tmp_path):
    path = lerobot.keep_alive_file(str(tmp_path / "zenoh.json5"), base={
        "transport": {"link": {"tx": {"batch_size": 65535}, "tls": {"root_ca_certificate": "ca.pem"}}},
        "access_control": {"enabled": True}})
    assert json.loads(open(path).read()) == {
        "transport": {"link": {"tx": {"batch_size": 65535, "keep_alive": 100},
                               "tls": {"root_ca_certificate": "ca.pem"}}},
        "access_control": {"enabled": True}}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _gaps_ms(zenoh, config_file, seconds: float = 2.5) -> list:
    """Two peers, the second reaching the first through a relay: the gaps between the frames the second sends while
    idle."""
    first_port, relay_port, stamps = _free_port(), _free_port(), []
    relay = socket.create_server(("127.0.0.1", relay_port))

    def pump(a, b, stamped):
        try:
            while data := a.recv(65536):
                if stamped:
                    stamps.append(time.perf_counter())
                b.sendall(data)
        except OSError:
            pass

    def accept():
        try:
            conn, _ = relay.accept()
        except OSError:
            return
        up = socket.create_connection(("127.0.0.1", first_port))
        threading.Thread(target=pump, args=(conn, up, True), daemon=True).start()
        threading.Thread(target=pump, args=(up, conn, False), daemon=True).start()

    threading.Thread(target=accept, daemon=True).start()

    def peer(**keys):
        config = zenoh.Config.from_file(config_file) if config_file else zenoh.Config()
        for key, value in {"mode": "peer", "scouting/multicast/enabled": False, "scouting/gossip/enabled": False,
                           **keys}.items():
            config.insert_json5(key, json.dumps(value))
        return zenoh.open(config)

    first = peer(**{"listen/endpoints": [f"tcp/127.0.0.1:{first_port}"]})
    second = peer(**{"connect/endpoints": [f"tcp/127.0.0.1:{relay_port}"]})
    time.sleep(seconds)
    second.close()
    first.close()
    relay.close()
    settled = [t for t in stamps if t >= stamps[0] + 1.0]      # after the handshake
    return [(b - a) * 1000 for a, b in zip(settled, settled[1:])]


def test_with_the_file_an_idle_zenoh_link_sends_a_frame_about_every_100_ms(tmp_path):
    zenoh = pytest.importorskip("zenoh")
    path = lerobot.keep_alive_file(str(tmp_path / "zenoh.json5"))
    gaps = _gaps_ms(zenoh, path)
    assert len(gaps) >= 8 and 80 <= statistics.median(gaps) <= 130
    assert len(_gaps_ms(zenoh, None)) < 3                      # Zenoh's default: a frame every second or less often
