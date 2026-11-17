# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
openpi's own policy server with a robot's pictures sent as JPEG and each connection pinged (openpi.patch, Pictures,
image_tools), against a server module built as openpi's websocket_policy_server.py is: the websockets library's server
module imported as `_server`, a WebsocketPolicyServer given its policy when it is made and started through
`_server.serve`, its metadata sent first and then a response for each observation (SERVER below). And
`python -m sequence_ai.network.openpi` running a script as openpi's scripts/serve_policy.py is run. Neither openpi nor
openpi_client is needed: msgpack_io carries the messages, as it reads openpi's array layout.
"""
import asyncio
import base64
import contextlib
import json
import os
import pathlib
import socket
import subprocess
import sys
import threading
import time
import types

import pytest

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")
pytest.importorskip("websockets")

import websockets.asyncio.server as library  # noqa: E402
from websockets.sync.client import connect  # noqa: E402

from sequence_ai.network import frames, msgpack_io, openpi  # noqa: E402
from sequence_ai.network.openpi import image_tools  # noqa: E402

TREE = pathlib.Path(__file__).resolve().parents[2]

SERVER = '''
import asyncio

import websockets.asyncio.server as _server

from sequence_ai.network import msgpack_io


class WebsocketPolicyServer:
    def __init__(self, policy, host="0.0.0.0", port=None, metadata=None):
        self._policy, self._host, self._port, self._metadata = policy, host, port, metadata or {}
        self.on_start = None                       # the tests' own: the server once it listens

    def serve_forever(self):
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(self._handler, self._host, self._port, compression=None, max_size=None) as server:
            if self.on_start is not None:
                self.on_start(server)
            await server.serve_forever()

    async def _handler(self, websocket):
        await websocket.send(msgpack_io.pack(self._metadata))
        async for message in websocket:
            await websocket.send(msgpack_io.pack(self._policy.infer(msgpack_io.unpack(message))))
'''


def _module():
    """A new server module each time, so what one test changes another never sees."""
    module = types.ModuleType("websocket_policy_server")
    exec(SERVER, module.__dict__)
    return module


@contextlib.contextmanager
def _serving(module, policy, metadata=None):
    made = module.WebsocketPolicyServer(policy, "127.0.0.1", 0, metadata)
    ready, box = threading.Event(), {}

    def started(server):
        box.update(server=server, port=server.sockets[0].getsockname()[1], loop=asyncio.get_running_loop(),
                   task=asyncio.current_task())
        ready.set()

    def run():
        with contextlib.suppress(asyncio.CancelledError):
            made.serve_forever()

    made.on_start = started
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert ready.wait(5)
    try:
        yield made, box["port"]
    finally:
        box["loop"].call_soon_threadsafe(box["task"].cancel)
        thread.join(5)


def _pings_in(port: int, seconds: float) -> int:
    """The ping frames a server sends one connection in `seconds`, each answered with a pong as a robot's library
    answers it (tests/network/test_ping.py has the same client)."""
    key = base64.b64encode(os.urandom(16)).decode()
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    s.sendall((f"GET / HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
               f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
    buf = b""
    while b"\r\n\r\n" not in buf:
        buf += s.recv(4096)
    assert buf.startswith(b"HTTP/1.1 101"), buf[:80]
    buf = buf.split(b"\r\n\r\n", 1)[1]
    s.settimeout(0.05)
    end, pings = time.time() + seconds, 0
    while time.time() < end:
        try:
            buf += s.recv(65536)
        except (socket.timeout, TimeoutError):
            pass
        while len(buf) >= 2:
            opcode, length = buf[0] & 0x0F, buf[1] & 0x7F
            if length == 126:                         # the metadata frame: a 16-bit length follows
                if len(buf) < 4:
                    break
                head, length = 4, int.from_bytes(buf[2:4], "big")
            else:
                head = 2
            if len(buf) < head + length:
                break
            if opcode == 0x9:
                pings += 1
                mask, payload = os.urandom(4), buf[head:head + length]
                s.sendall(bytes([0x8A, 0x80 | length]) + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload)))
            buf = buf[head + length:]
    s.close()
    return pings


def _scene(h: int = 480, w: int = 640):
    """A camera frame of gradients: JPEG keeps it within a few levels."""
    yy, xx = np.mgrid[0:h, 0:w]
    return np.stack([xx * 255 // w, yy * 255 // h, (xx + yy) * 255 // (h + w)], -1).astype(np.uint8)


class _Policy:
    metadata = {"robot": "droid", "action_horizon": 10}

    def __init__(self):
        self.seen = []

    def infer(self, obs, *, noise=None):
        self.seen.append((obs, noise))
        return {"actions": np.zeros((10, 8), np.float32)}


def test_the_patched_server_hands_the_policy_each_jpeg_picture_as_its_array_and_everything_else_unchanged():
    module, policy = _module(), _Policy()
    openpi.patch(server=module)
    scene, wrist = _scene(), _scene(224, 224)[::-1].copy()
    observation = {
        "observation/exterior_image_1_left": image_tools.convert_to_uint8(image_tools.resize_with_pad(scene, 224, 224)),
        "observation/wrist_image_left": wrist,                                   # sent as an array, as before
        "images": {"cam_high": image_tools.resize_with_pad(scene, 224, 224)},
        "observation/joint_position": np.arange(7, dtype=np.float64),
        "prompt": "put the marker in the cup",
    }
    with _serving(module, policy, _Policy.metadata) as (made, port):
        assert isinstance(made._policy, openpi.Pictures)
        with connect(f"ws://127.0.0.1:{port}", compression=None, max_size=None) as robot:
            assert msgpack_io.unpack(robot.recv()) == _Policy.metadata
            robot.send(msgpack_io.pack(observation))
            assert msgpack_io.unpack(robot.recv())["actions"].shape == (10, 8)
    (seen, _), = policy.seen
    fitted = np.asarray(frames.resize_image(scene, 224, 224, mode="pad", filter="bilinear"))
    for picture in (seen["observation/exterior_image_1_left"], seen["images"]["cam_high"]):
        assert picture.shape == (224, 224, 3) and picture.dtype == np.uint8
        assert np.abs(picture.astype(int) - fitted.astype(int)).mean() < 2
    assert np.array_equal(seen["observation/wrist_image_left"], wrist)
    assert np.array_equal(seen["observation/joint_position"], np.arange(7, dtype=np.float64))
    assert seen["prompt"] == "put the marker in the cup"


def test_the_patched_server_pings_each_connection_and_one_not_patched_does_not():
    module = _module()
    openpi.patch(server=module, every_s=0.05)
    with _serving(module, _Policy()) as (_, port):
        assert _pings_in(port, 0.6) >= 6
    with _serving(_module(), _Policy()) as (_, port):
        assert _pings_in(port, 0.4) == 0                      # the library's default interval is 20 s


def test_a_second_patch_changes_the_interval_alone_and_the_policy_is_wrapped_once():
    module, policy = _module(), _Policy()
    openpi.patch(server=module, every_s=0.05)
    openpi.patch(server=module, every_s=0.2)
    assert module._server.options == {"ping_interval": 0.2, "ping_timeout": 20.0}
    made = module.WebsocketPolicyServer(policy=openpi.Pictures(policy), port=0)
    assert made._policy.policy is policy and made._port == 0
    assert module.WebsocketPolicyServer(policy)._policy.policy is policy
    assert module._server.ServerConnection is library.ServerConnection        # the rest is the library's own


def test_a_server_not_built_as_openpis_is_refused():
    for module in (types.ModuleType("empty"), types.SimpleNamespace(_server=object(), WebsocketPolicyServer=object),
                   types.SimpleNamespace(_server=_module()._server, WebsocketPolicyServer=None)):
        with pytest.raises(TypeError, match="not built as openpi"):
            openpi.patch(server=module)


def test_pictures_keeps_the_policys_own_attributes_and_arguments_and_names_a_picture_it_cannot_read():
    policy = _Policy()
    wrapped = openpi.Pictures(policy)
    assert wrapped.metadata == _Policy.metadata
    noise = np.ones(3)
    wrapped.infer({"observation/image": frames.encode(_scene(32, 32)), "prompt": "go"}, noise=noise)
    (seen, got_noise), = policy.seen
    assert seen["observation/image"].shape == (32, 32, 3) and got_noise is noise
    with pytest.raises(ValueError, match="images/cam_low: not a JPEG or PNG image"):
        wrapped.infer({"images": {"cam_low": "not a picture"}})
    assert openpi.decode_pictures({"prompt": "an image of a cup", "state": [1, 2]}) == {
        "prompt": "an image of a cup", "state": [1, 2]}


def test_image_tools_fit_a_frame_as_openpi_does_and_send_it_as_jpeg():
    scene = _scene()
    sent = image_tools.resize_with_pad(scene, 224, 224)
    assert sent == frames.encode(frames.resize_image(scene, 224, 224, mode="pad", filter="bilinear"))
    assert frames.image_size(sent) == (224, 224)
    assert image_tools.resize_with_pad(scene, 180, 320, method=Image.NEAREST) == frames.encode(
        frames.resize_image(scene, 320, 180, mode="pad", filter="nearest"))
    assert image_tools.convert_to_uint8(sent) is sent
    assert image_tools.convert_to_uint8(np.full((2, 2, 3), 0.5)).tolist() == np.full((2, 2, 3), 127).tolist()
    whole = np.zeros((2, 2, 3), np.uint8)
    assert image_tools.convert_to_uint8(whole) is whole
    with pytest.raises(ValueError, match="one picture at a time"):
        image_tools.resize_with_pad(np.zeros((2, 480, 640, 3), np.uint8), 224, 224)


SCRIPT = '''
import json
import sys

from openpi.serving import websocket_policy_server


class Policy:
    metadata = {"robot": "droid"}

    def infer(self, obs):
        return obs


made = websocket_policy_server.WebsocketPolicyServer(policy=Policy(), host="0.0.0.0", port=7447,
                                                     metadata=Policy.metadata)
print(json.dumps({"name": __name__, "argv": sys.argv, "path0": sys.path[0], "policy": type(made._policy).__name__,
                  "options": websocket_policy_server._server.options}))
'''


def test_python_dash_m_runs_the_script_as_its_own_command_with_openpis_server_patched(tmp_path):
    serving = tmp_path / "site" / "openpi" / "serving"
    serving.mkdir(parents=True)
    (serving.parent / "__init__.py").write_text("")
    (serving / "__init__.py").write_text("")
    (serving / "websocket_policy_server.py").write_text(SERVER)
    scripts = tmp_path / "repo" / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "serve_policy.py").write_text(SCRIPT)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(TREE), str(tmp_path / "site"),
                                                                      os.environ.get("PYTHONPATH")]))}

    def run(*args):
        return subprocess.run([sys.executable, "-m", "sequence_ai.network.openpi", *args], capture_output=True,
                              text=True, timeout=120, cwd=tmp_path / "repo", env=env)

    done = run("scripts/serve_policy.py", "--env=DROID", "--port=7447")
    assert done.returncode == 0, done.stderr[-2000:]
    said = json.loads(done.stdout.strip().splitlines()[-1])
    assert said["name"] == "__main__" and said["argv"] == ["scripts/serve_policy.py", "--env=DROID", "--port=7447"]
    assert os.path.realpath(said["path0"]) == os.path.realpath(scripts)
    assert said["policy"] == "Pictures" and said["options"] == {"ping_interval": 0.1, "ping_timeout": 20.0}
    usage = run()
    assert usage.returncode == 2 and usage.stdout.startswith("usage: python -m sequence_ai.network.openpi")
    missing = run("scripts/no_such.py")
    assert missing.returncode != 0 and "no script at scripts/no_such.py" in missing.stderr
