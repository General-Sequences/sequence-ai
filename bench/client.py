#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
How long an upload takes on one TCP connection, by its size and by how it is sent: after a pause, with the far end
pinging, with this end pinging, back to back, and as a stream.

    python3 bench/client.py --host HOST [--port 9000] [--sizes 100k,500k,1m] [--idle-ms 500] [--ping-ms 100]
        [--stream 30x50k,10x100k,10x500k] [--seconds 10] [--samples 20] [--out result.json]
        [--modes idle,ping,continuous,stream] [--bbr] [--warm 1m] [--two-writes 200+4k] [--start-at UNIX_TIME]

The far end runs bench/server.py. Standard library only. On Linux and macOS it also records the kernel's state of the
connection (congestion window, retransmissions), which is what explains a slow upload.

WHY one connection: a robot keeps one long-lived connection open, and a measurement has to incur the same costs: one
handshake, and whatever is left of the window after each pause.

What runs, in order, on that one connection (its handshake is timed once, separately):
    baseline    an empty frame and its ack, `samples` times: the median is the round-trip time (RTT) with no data
    idle        for each size: a frame, its ack, then `idle-ms` of silence, and the next
    ping        as idle, with the server sending a one-byte ping every `ping-ms` and this end replying to each with a
                byte immediately, as a WebSocket library does with a ping
    keepwarm    as idle, with this end sending a byte every `ping-ms` while it waits, as a robot does when it cannot
                change the server
    continuous  for each size: the next frame as soon as the last is acknowledged
    stream      for each HZxSIZE: frames at a fixed rate, not waiting for acks; each frame is timed from the start of
                its send to its ack
    twowrite    not run by default. Nagle left on and a frame written in two calls (`--two-writes FIRST+SECOND`): the
                second write stalls until the first is acknowledged, and the far end's delayed ack adds to that
`--modes` selects the blocks and their order: for each size idle / ping / keepwarm / continuous as given, then stream,
then twowrite. `NAME:N` gives one block its own number of samples: `continuous:5,ping` warms the window with five
uploads and then measures ping.
Each block sends WARMUP frames first, not counted. All times are this machine's perf_counter: the clocks of two
machines are never compared.

`--bbr` requests BBR on this connection's socket (Linux, where the tcp_bbr module is loaded); the result records which
congestion control the connection got. `--warm SIZE` uploads one frame of that size before anything is measured, as a
client that widens its window on connect does.

The result: `--out`, a JSON file (each block's p50, p95, `rtts` = p50 divided by the baseline RTT, and every sample),
and a table on stdout.

To reproduce a robot that sends a request every 0.53 s with 24 KB of frames:
    python3 bench/client.py --host HOST --sizes 24k --idle-ms 530 --samples 40 --modes idle,ping,keepwarm --warm 1m
"""
import argparse
import json
import os
import platform
import queue
import socket
import struct
import sys
import threading
import time

WARMUP = 3                      # frames before a block's samples: the window settles to this block's steady state
ACK, PING = 0x41, 0x50          # server to client: b"A", b"P"
START_PING, STOP_PING, PONG = 0xFE, 0xFD, b"\xff"
TCP_CONGESTION = getattr(socket, "TCP_CONGESTION", None)
# macOS's TCP_CONNECTION_INFO (netinet/tcp.h). A Mac's own Python 3.9 (/usr/bin/python3) does not define the name and
# defines TCP_INFO instead; a Homebrew or python.org Python defines this one and not TCP_INFO.
TCP_CONNECTION_INFO = getattr(socket, "TCP_CONNECTION_INFO", 0x106)
pc = time.perf_counter


def log(msg):
    """Print progress to stderr; stdout is reserved for the table."""
    print(time.strftime("%H:%M:%S"), msg, file=sys.stderr, flush=True)


def parse_size(text):
    """'100k' is 100000 and '1m' 1000000: decimal, so 30 Hz of 50 KB frames is exactly 12 Mbit/s."""
    t = text.strip().lower()
    mult = {"k": 1000, "m": 1000_000}.get(t[-1], 1)
    return int(float(t[:-1] if mult > 1 else t) * mult)


def parse_stream(text):
    """'30x50k' is (30.0, 50000): frames per second, bytes per frame."""
    hz, size = text.lower().split("x")
    return float(hz), parse_size(size)


def parse_pair(text):
    """'200+4k' is (200, 4000): the bytes of the first write (including the 4 bytes of the length) and of the second."""
    first, second = text.split("+")
    return parse_size(first), parse_size(second)


def pct(xs, p):
    """A percentile by linear interpolation: more stable than nearest rank on a few dozen samples."""
    xs = sorted(xs)
    r = p / 100 * (len(xs) - 1)
    lo = int(r)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (r - lo)


def tcp_info(sock):
    """The kernel's state of this connection: congestion window and slow-start threshold (in segments), retransmission
    timeout, smoothed RTT, how long it has sent nothing, and retransmissions so far.

    Linux reads TCP_INFO (struct tcp_info: eight one-byte fields, then 32-bit ones; min_rtt at byte 148). macOS reads
    TCP_CONNECTION_INFO (struct tcp_connection_info: rto at 12, maxseg 16, ssthresh 20 and cwnd 24, in bytes and
    converted here to segments; srtt at 44, retransmitted packets at 104), which has no idle time and no minimum RTT.
    Anywhere else: None.

    WHY the platform and not the name decides: a Mac's Python can define socket.TCP_INFO as well (0x200, for Apple's
    own struct tcp_info, another layout), and read as Linux's it gave a window of 0 and a growing threshold on every
    upload."""
    if sys.platform.startswith("linux") and hasattr(socket, "TCP_INFO"):
        raw = sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_INFO, 160)
        u = struct.unpack_from("<24I", raw, 8)
        return {"cwnd": u[18], "ssthresh": u[17], "rto_ms": u[0] / 1000, "rtt_ms": u[15] / 1000,
                "idle_ms": u[9], "mss": u[2], "retrans": u[23],
                "min_rtt_ms": struct.unpack_from("<I", raw, 148)[0] / 1000}
    if sys.platform == "darwin":
        raw = sock.getsockopt(socket.IPPROTO_TCP, TCP_CONNECTION_INFO, 112)
        rto, mss, ssthresh, cwnd = struct.unpack_from("<4I", raw, 12)
        return {"cwnd": cwnd // mss, "ssthresh": ssthresh // mss, "rto_ms": rto,
                "rtt_ms": struct.unpack_from("<I", raw, 44)[0], "idle_ms": None, "mss": mss,
                "retrans": struct.unpack_from("<Q", raw, 104)[0], "min_rtt_ms": None}
    return None


def frame(size):
    """A frame: 4 bytes of big-endian length and that many random bytes, so nothing on the path can compress it."""
    return struct.pack(">I", size) + os.urandom(size)


class Conn:
    """The one connection. The main thread sends frames; a reader thread handles what comes back: b"A" is timestamped on
    arrival and queued, and b"P" is replied to with a byte immediately.

    WHY the arrival is timestamped in the reader: by the time the main thread takes it off the queue a thread switch has
    happened; the timestamp is taken where recv returned."""

    def __init__(self, host, port, bbr=False):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Nagle off: a frame's last segment, shorter than a full one, must not wait for the earlier ones to be acked
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.bbr = False
        if bbr and TCP_CONGESTION is not None:
            try:
                self.sock.setsockopt(socket.IPPROTO_TCP, TCP_CONGESTION, b"bbr")
                self.bbr = True
            except OSError:
                pass                          # not loaded, or not allowed: the result records what it got
        t0 = pc()
        self.sock.connect((host, port))
        self.handshake_ms = (pc() - t0) * 1000
        self.lock = threading.Lock()          # one writer at a time: a frame and a pong never interleave
        self.acks = queue.Queue()
        self.pongs = 0
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        while True:
            try:
                data = self.sock.recv(4096)
            except OSError:
                return
            t = pc()
            if not data:
                return
            for b in data:
                if b == ACK:
                    self.acks.put(t)
                elif b == PING:
                    # as a WebSocket library replies: after the frame being written, never inside it
                    try:
                        with self.lock:
                            self.sock.sendall(PONG)
                    except OSError:
                        return
                    self.pongs += 1
                else:
                    raise ValueError(f"unexpected byte 0x{b:02x} from server")

    def send(self, data):
        with self.lock:
            self.sock.sendall(data)

    def ack(self):
        """The next ack's arrival time; an ack that takes a minute means the connection is broken."""
        return self.acks.get(timeout=60)

    def info(self):
        return tcp_info(self.sock)

    def congestion(self):
        """The congestion control this connection uses ("bbr", "cubic"), or None where it cannot be read."""
        if TCP_CONGESTION is None:
            return None
        try:
            return self.sock.getsockopt(socket.IPPROTO_TCP, TCP_CONGESTION, 16).split(b"\0", 1)[0].decode() or None
        except OSError:
            return None

    def close(self):
        # shut down first, so the reader's recv returns, then close: a closed descriptor is never read by a thread
        self.sock.shutdown(socket.SHUT_RDWR)
        self.reader.join(timeout=5)
        self.sock.close()


def baseline(c, samples):
    """An empty frame and its ack: this connection's RTT with no data."""
    out = []
    empty = struct.pack(">I", 0)
    for _ in range(samples):
        t0 = pc()
        c.send(empty)
        out.append((c.ack() - t0) * 1000)
        time.sleep(0.02)
    return out


def wait_idle(c, idle_s, keep_ms):
    """Wait `idle_s` in silence; with `keep_ms`, send a byte from this end every `keep_ms` during it. Returns how many
    bytes were sent."""
    if not keep_ms:
        time.sleep(idle_s)
        return 0
    sent, end = 0, pc() + idle_s
    while True:
        left = end - pc()
        if left <= keep_ms / 1000:
            time.sleep(max(0.0, left))
            return sent
        time.sleep(keep_ms / 1000)
        c.send(PONG)
        sent += 1


def request_response(c, size, samples, idle_s, keep_ms=0):
    """A frame and its ack, `samples` times; after each, `idle_s` of silence (idle, ping, keepwarm) or none
    (continuous).

    The silence is counted from the ack, not on a fixed period: a large frame to a distant server takes hundreds of
    milliseconds by itself, and a fixed period would leave a pause near the retransmission timeout, so the results would
    alternate between a restarted window and a kept one."""
    data = frame(size)
    rows, kept = [], 0
    for i in range(WARMUP + samples):
        before = c.info()         # the window and the idle time before the send: a restart after a pause shows here
        t0 = pc()
        c.send(data)
        sent = c.info()           # the window as sendall returned: a restart takes effect at the start of the send
        t1 = c.ack()
        if i >= WARMUP:
            rows.append({"ms": round((t1 - t0) * 1000, 3), "before": before, "sent": sent})
        if idle_s:
            kept += wait_idle(c, idle_s, keep_ms)
    return rows, kept


def stream(c, hz, size, seconds):
    """Frames at a fixed rate, without waiting for acks; the acks return in order, so the k-th ack belongs to the k-th
    frame."""
    data = frame(size)
    n = WARMUP + int(round(seconds * hz))
    sends = []
    t_start = pc() + 0.05
    for k in range(n):
        # an absolute schedule: frame k is due at t_start + k / hz; a late one is sent immediately, and none is skipped
        target = t_start + k / hz
        delay = target - pc()
        if delay > 0:
            time.sleep(delay)
        t0 = pc()
        c.send(data)
        sends.append((target, t0))
    acks = [c.ack() for _ in range(n)]
    rows = [{"ms": round((a - t0) * 1000, 3), "slip_ms": round((t0 - target) * 1000, 3)}
            for (target, t0), a in zip(sends, acks)]
    return rows[WARMUP:], c.info()


def two_writes(c, first, second, samples, idle_s):
    """A frame written in two calls (`first` bytes, including the 4 bytes of the length, then `second`) and its ack.

    WHY: a client that leaves Nagle on and writes a header and a body separately hits this. Nagle holds the second
    write's last segment until the first write is acknowledged, and a far end that delays its acks adds tens of
    milliseconds. Nagle is turned on for this block only and off again after it."""
    data = frame(first + second - 4)
    c.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 0)
    rows = []
    for i in range(WARMUP + samples):
        t0 = pc()
        with c.lock:
            c.sock.sendall(data[:first])
            c.sock.sendall(data[first:])
        t1 = c.ack()
        if i >= WARMUP:
            rows.append({"ms": round((t1 - t0) * 1000, 3)})
        time.sleep(idle_s)
    c.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return rows, c.info()


def summary(ms, base_ms, with_p99=False):
    """p50 and p95 (p99 for a stream), and `rtts`: the p50 divided by the baseline RTT."""
    s = {"n": len(ms), "p50_ms": round(pct(ms, 50), 2), "p95_ms": round(pct(ms, 95), 2)}
    if with_p99:
        s["p99_ms"] = round(pct(ms, 99), 2)
    s["rtts"] = round(s["p50_ms"] / base_ms, 2) if base_ms > 0 else None
    return s


def sysctls():
    """This machine's TCP settings, needed to interpret a result (tcp_slow_start_after_idle above all); empty outside
    Linux."""
    out = {}
    for k in ["ipv4/tcp_congestion_control", "ipv4/tcp_slow_start_after_idle",
              "ipv4/tcp_rmem", "ipv4/tcp_wmem", "core/default_qdisc"]:
        try:
            with open(f"/proc/sys/net/{k}") as f:
                out[k] = f.read().strip()
        except OSError:
            pass
    return out


MODES = ("idle", "ping", "keepwarm", "continuous", "stream", "twowrite")


def main(argv=None):
    ap = argparse.ArgumentParser(description="bench client: how long an upload takes on one TCP connection")
    ap.add_argument("--host", required=True, help="where bench/server.py runs")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--sizes", default="100k,500k,1m", help="frame sizes for idle, ping, keepwarm and continuous")
    ap.add_argument("--idle-ms", type=int, default=500, help="silence after each ack in idle, ping and keepwarm")
    ap.add_argument("--ping-ms", type=int, default=100,
                    help="the interval of pings (ping) and of this end's bytes (keepwarm)")
    ap.add_argument("--stream", default="30x50k,10x100k,10x500k", help="HZxSIZE, comma separated")
    ap.add_argument("--seconds", type=float, default=10, help="how long each stream runs")
    ap.add_argument("--samples", type=int, default=20, help="samples for each block and size")
    ap.add_argument("--out", default="result.json")
    ap.add_argument("--modes", default="idle,ping,continuous,stream",
                    help="blocks, in order: " + ",".join(MODES) + "; NAME:N gives a block its own samples")
    ap.add_argument("--bbr", action="store_true", help="request BBR on this connection's socket")
    ap.add_argument("--warm", default="", help="upload one frame of this size before measuring, e.g. 1m")
    ap.add_argument("--two-writes", default="200+4k", help="FIRST+SECOND bytes, for the block twowrite")
    ap.add_argument("--start-at", type=float, default=0.0,
                    help="unix time to start the first block at: synchronizes several clients")
    a = ap.parse_args(argv)
    sizes = [parse_size(x) for x in a.sizes.split(",")]
    streams = [parse_stream(x) for x in a.stream.split(",")]
    pairs = [parse_pair(x) for x in a.two_writes.split(",")]
    plan = [(name, int(n) if n else a.samples)
            for name, _, n in (spec.partition(":") for spec in a.modes.split(","))]
    unknown = {name for name, _ in plan} - set(MODES)
    if unknown:
        ap.error(f"unknown mode(s): {', '.join(sorted(unknown))}")

    res = {"tool": "sequence-ai bench/client.py", "host": a.host, "port": a.port,
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "client": {"os": platform.platform(), "python": platform.python_version(), "sysctl": sysctls()},
           "params": {"sizes": sizes, "idle_ms": a.idle_ms, "ping_ms": a.ping_ms, "stream": streams,
                      "seconds": a.seconds, "samples": a.samples, "modes": a.modes, "two_writes": pairs,
                      "warmup": WARMUP, "bbr_requested": a.bbr, "warm": a.warm},
           "results": []}

    def save():
        # saved after each block, so an interrupted run keeps what it measured
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)

    c = Conn(a.host, a.port, bbr=a.bbr)
    res["handshake_ms"] = round(c.handshake_ms, 3)
    res["congestion"] = c.congestion()
    rtt = baseline(c, a.samples)
    base = pct(rtt, 50)
    res["baseline"] = {"n": len(rtt), "min_ms": round(min(rtt), 3), "p50_ms": round(base, 3),
                       "samples_ms": [round(x, 3) for x in rtt]}
    log(f"handshake {c.handshake_ms:.1f} ms, baseline p50 {base:.1f} ms, congestion control {res['congestion']}")
    if a.warm:
        t0 = pc()
        c.send(frame(parse_size(a.warm)))
        res["warm_ms"] = round((c.ack() - t0) * 1000, 3)
        res["after_warm"] = c.info()
    save()

    if a.start_at:
        late = time.time() - a.start_at
        res["start_late_s"] = round(max(late, 0.0), 3)
        if late < 0:
            time.sleep(-late)

    for size in sizes:
        for mode, n in plan:
            if mode not in ("idle", "ping", "keepwarm", "continuous"):
                continue
            extra = {}
            if mode == "ping":
                c.send(bytes([START_PING]) + struct.pack(">H", a.ping_ms))
                p0 = c.pongs
            rows, kept = request_response(c, size, n, 0 if mode == "continuous" else a.idle_ms / 1000,
                                          keep_ms=a.ping_ms if mode == "keepwarm" else 0)
            if mode == "ping":
                c.send(bytes([STOP_PING]))
                extra["pongs"] = c.pongs - p0
            if mode == "keepwarm":
                extra["sent_while_idle"] = kept
            res["results"].append({"mode": mode, "size": size, **extra,
                                   **summary([r["ms"] for r in rows], base), "samples": rows})
            save()
        log(f"size {size} done")

    if any(name == "stream" for name, _ in plan):
        for hz, size in streams:
            start_unix = round(time.time() + 0.05, 3)      # the first frame's wall-clock time, to align runs
            rows, end_info = stream(c, hz, size, a.seconds)
            res["results"].append({"mode": "stream", "hz": hz, "size": size, "start_unix": start_unix,
                                   "mbps": round(hz * size * 8 / 1e6, 1),
                                   **summary([r["ms"] for r in rows], base, with_p99=True),
                                   "end_info": end_info, "samples": rows})
            save()
            log(f"stream {hz:g}x{size} done")

    for mode, n in plan:
        if mode != "twowrite":
            continue
        for first, second in pairs:
            rows, end_info = two_writes(c, first, second, n, a.idle_ms / 1000)
            res["results"].append({"mode": "twowrite", "first": first, "second": second, "size": first + second - 4,
                                   **summary([r["ms"] for r in rows], base), "end_info": end_info, "samples": rows})
            save()
            log(f"twowrite {first}+{second} done")

    c.close()
    res["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save()

    print(f"host {a.host}  handshake {res['handshake_ms']:.1f} ms  "
          f"baseline rtt min {res['baseline']['min_ms']:.1f} / p50 {base:.1f} ms  congestion {res['congestion']}")
    print(f"{'mode':<11}{'size':>9}{'p50 ms':>10}{'p95 ms':>10}{'p99 ms':>10}{'x rtt':>8}")
    for r in res["results"]:
        name = f"{r['hz']:g}Hz" if r["mode"] == "stream" else r["mode"]
        p99 = f"{r['p99_ms']:.1f}" if "p99_ms" in r else "-"
        rtts = f"{r['rtts']:.2f}" if r["rtts"] is not None else "-"
        print(f"{name:<11}{r['size']:>9}{r['p50_ms']:>10.1f}{r['p95_ms']:>10.1f}{p99:>10}{rtts:>8}")
    return res


if __name__ == "__main__":
    main()
