#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
The far end of bench/client.py: it sends one byte back for each complete frame it receives, and pings when the client
requests them.

    python3 bench/server.py [--host 0.0.0.0] [--port 9000]

Run it on the machine the GPU is on (or any machine at that distance); run bench/client.py where the robot is. Standard
library only; Linux and macOS.

The protocol, client to server:
    a frame     4 bytes of big-endian length, then that many bytes. A length is under 16 MiB, so a frame's first byte
                is always 0x00, which leaves every other first byte available as a control byte.
    0xFE HH     start (or change) a ping every HH milliseconds; HH is 2 bytes, big-endian
    0xFD        stop the pings
    0xFF        the client's own byte (a pong, or its keep-warm ping): read and discarded
Server to client: b"A", a complete frame was received; b"P", a ping.
"""
import argparse
import socket
import struct
import threading

ACK, PING = b"A", b"P"
START_PING, STOP_PING, PONG = 0xFE, 0xFD, 0xFF
SCRATCH = 256 * 1024      # a frame's bytes are not kept: they are read into one reused buffer


def recv_exact(conn, view):
    """Fill `view`. A peer that closes midway raises: a partial frame is never acknowledged."""
    got = 0
    while got < len(view):
        k = conn.recv_into(view[got:])
        if k == 0:
            raise ConnectionError("peer closed mid-message")
        got += k


def drain(conn, n, scratch):
    """Read `n` bytes of a frame and discard them: only their arrival is measured, so no buffer of that size is
    allocated."""
    while n:
        k = conn.recv_into(scratch, min(n, len(scratch)))
        if k == 0:
            raise ConnectionError("peer closed mid-frame")
        n -= k


class Pinger:
    """One connection's pings: started, restarted at another interval, stopped; one sending thread at most."""

    def __init__(self, conn, lock):
        self.conn, self.lock = conn, lock
        self.stop_evt = None          # the running thread's stop signal

    def start(self, interval_s):
        """Send a ping every `interval_s` from now, replacing any pings already running."""
        self.stop()
        evt = self.stop_evt = threading.Event()
        threading.Thread(target=self._run, args=(evt, interval_s), daemon=True).start()

    def stop(self):
        if self.stop_evt:
            self.stop_evt.set()
            self.stop_evt = None

    def _run(self, evt, interval_s):
        while not evt.wait(interval_s):
            try:
                with self.lock:           # the lock the acks are written under: bytes of the two never interleave
                    self.conn.sendall(PING)
            except OSError:
                return                    # the connection has closed


def handle(conn, peer):
    """Serve one connection until its peer closes: a frame is read completely and acknowledged with b"A"; a control byte
    is handled."""
    # Nagle off: a one-byte ack or ping must be sent immediately, not wait for earlier bytes to be acknowledged
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    lock = threading.Lock()
    pinger = Pinger(conn, lock)
    scratch = memoryview(bytearray(SCRATCH))
    hdr, two = bytearray(4), bytearray(2)
    frames = 0
    try:
        while True:
            first = conn.recv(1)
            if not first:
                return
            b = first[0]
            if b == PONG:
                continue
            if b == START_PING:
                recv_exact(conn, memoryview(two))
                pinger.start(struct.unpack(">H", two)[0] / 1000)
                continue
            if b == STOP_PING:
                pinger.stop()
                continue
            if b != 0:
                # a frame starts with 0x00; anything else means the stream is out of sync, and closing is safer than
                # acknowledging arbitrary bytes
                raise ValueError(f"unexpected first byte 0x{b:02x}")
            hdr[0] = b
            recv_exact(conn, memoryview(hdr)[1:])
            drain(conn, struct.unpack(">I", hdr)[0], scratch)
            with lock:
                conn.sendall(ACK)
            frames += 1
    except (ConnectionError, OSError):
        pass
    finally:
        pinger.stop()
        conn.close()
        print(f"{peer[0]}:{peer[1]} closed after {frames} frames", flush=True)


def serve(host="0.0.0.0", port=9000):
    """The listening socket; port 0 selects a free port (read it back with getsockname)."""
    return socket.create_server((host, port), backlog=16)


def accept_forever(srv):
    """Handle each connection on its own thread; a measurement usually has one."""
    while True:
        try:
            conn, peer = srv.accept()
        except OSError:
            return                        # the listening socket was closed
        print(f"{peer[0]}:{peer[1]} connected", flush=True)
        threading.Thread(target=handle, args=(conn, peer), daemon=True).start()


def main():
    ap = argparse.ArgumentParser(description="bench server: a one-byte ack for each frame, pings on request")
    ap.add_argument("--host", default="0.0.0.0", help="address to listen on (default 0.0.0.0)")
    ap.add_argument("--port", type=int, default=9000, help="TCP port (default 9000)")
    args = ap.parse_args()
    srv = serve(args.host, args.port)
    print(f"listening on {args.host}:{srv.getsockname()[1]}", flush=True)
    accept_forever(srv)


if __name__ == "__main__":
    main()
