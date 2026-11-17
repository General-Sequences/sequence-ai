# What do the components save, measured with the tools in this repository?

On a clean link between two cloud machines, an upload that follows a robot's half-second pause took two to three round
trips with the defaults, and one with any of the components that keep the connection from going idle. That held over
plain TCP, over a WebSocket and over Zenoh, at round-trip times (RTTs) from 11 to 101 ms. At 242 ms the pause cost
nothing, and under 1% packet loss a ping did not help where BBR did.

Everything on this page was measured on 2026-11-15 with [`bench/`](../bench/), as released. It is one pair of machines
on one night. **It is not a promise for your link:**
[measure yours](how-do-i-measure-the-link-between-my-robot-and-my-gpu.md).

## The setup

- **Two machines** of one cloud provider, in two regions 11 to 12 ms apart by `ping`, each with 2 shared vCPUs and
  2 GB of memory. Ubuntu 24.04, Linux 7.0, CUBIC as the default congestion control, and
  `net.ipv4.tcp_slow_start_after_idle=1`, the kernel's default.
- **One is the server and one the client.** There is no GPU and no model: the far end responds with a few bytes, so
  every millisecond below is the transport's.
- **Longer RTTs are the same link with delay added** where the robot's packets leave
  (`tc qdisc add dev IF root netem delay Nms`).
- **No run inherits another's window:** `net.ipv4.tcp_no_metrics_save=1`, and `ip tcp_metrics flush all` before
  each run.
- **A cell is 40 uploads** after 3 that are not counted (20 in the table of sizes). An upload "took more than one
  round trip" when it took over 1.5 times its run's baseline: the median of small requests on the warm connection.
- **The pause is counted from the response.** After the response to one upload arrives, the client waits (0.53 s,
  or 0.5 s in the table of sizes) and uploads again. The connection has then sent nothing for the pause plus one
  RTT.

## An upload after a pause, over TCP

A 24 KB upload 0.53 s after each response, on a connection first widened with one 1 MB upload
(`bench/client.py --sizes 24k --idle-ms 530 --samples 40 --warm 1m`). Each cell is the median upload in round trips,
and how many of the 40 took more than one.

| RTT | Default | The server pings every 100 ms | The client sends a byte every 100 ms | BBR on the robot's socket | `tcp_slow_start_after_idle=0` on the robot |
|---|---|---|---|---|---|
| 12.4 ms | **1.99 (40 of 40)** | 1.01 (0) | 1.01 (0) | 1.01 (0) | 1.01 (0) |
| 21.4 ms | **2.00 (40 of 40)** | 1.01 (0) | 1.01 (0) | 1.00 (0) | 1.01 (0) |
| 40.5 ms | **1.99 (40 of 40)** | 1.00 (0) | 1.00 (0) | 1.00 (0) | 1.00 (0) |
| 65.5 ms | **2.00 (40 of 40)** | 1.00 (0) | 1.00 (0) | 1.01 (0) | 1.00 (0) |
| 81.3 ms | **2.00 (40 of 40)** | 1.00 (0) | 1.00 (0) | 1.00 (0) | 1.00 (0) |
| 100.7 ms | **2.00 (40 of 40)** | 1.00 (0) | 1.00 (0) | 1.00 (0) | 1.00 (0) |
| 242.2 ms | 1.00 (0 of 40) | 1.00 (0) | 1.00 (0) | 1.01 (0) | 1.00 (0) |

In milliseconds, the default against the server's ping: 24.7 against 12.5, 42.7 against 21.5, 80.6 against 40.4,
130.6 against 65.5, 162.3 against 81.4, 201.3 against 100.9, and 242.4 against 242.3.

**Why 242 ms took no extra round trip.** The kernel halves an idle connection's window once for each retransmission
timeout that passed, and the timeout is about the RTT plus 200 ms. Up to 101 ms, the pause plus one RTT spanned two
timeouts: the window, 28 packets before each upload, was cut to 10, the least it is cut to, and the upload left in two
flights. At 242 ms they spanned one: the window, 36 packets before each upload, became 18, and the whole upload left at
once.

**Why this differs from an earlier measurement at about 101 ms.** With a robot that uploaded on a fixed period of
0.53 s, counted from the start of one upload to the start of the next, the same 24 KB upload took no extra round trip
at about 101 ms: that silence spans one timeout, not two. Here the 0.53 s is counted from the response, which matches
a robot that starts its pause when the response arrives. **At about 100 ms the two ways of counting the same half
second fall on either side of the edge.** A robot's pause, its RTT and its upload's size decide together, and a small
change in any of them can move a link across.

## Larger uploads

Uploads 0.5 s after each response, with no widening first
(`bench/client.py --sizes 100k,500k,1m --idle-ms 500 --samples 20 --modes idle,ping,keepwarm,continuous`). Median in
round trips; in brackets, how many of the 20 took more than one.

| RTT | Upload | Default | The server pings | The client sends | One after another, no pause |
|---|---|---|---|---|---|
| 13.2 ms | 100 KB | **2.98 (20)** | 1.10 (0) | 1.08 (0) | 1.06 (0) |
| 13.2 ms | 500 KB | **2.99 (20)** | 1.23 (1) | 1.24 (0) | 1.15 (0) |
| 13.2 ms | 1 MB | **3.01 (20)** | 1.31 (1) | 1.36 (0) | 1.35 (8) |
| 81.3 ms | 100 KB | **2.99 (20)** | 1.00 (0) | 1.00 (0) | 1.00 (0) |
| 81.3 ms | 500 KB | **3.00 (20)** | 1.01 (0) | 1.01 (0) | 1.01 (0) |
| 81.3 ms | 1 MB | **2.01 (20)** | 1.02 (0) | 1.01 (0) | 1.01 (0) |

At 13 ms, 1 MB takes over one round trip even with no pause at all: at that distance the time to push a megabyte
through is no longer small beside the RTT. That is the link's rate, and none of these components changes it.

## A camera stream

Frames at a fixed rate with no pause longer than the frame period
(`bench/client.py --modes stream --stream 30x50k,10x100k --seconds 10`), at the 12 ms RTT: 50 KB frames at
30 Hz took a median of 1.02 round trips each, none of 300 over one; 100 KB frames at 10 Hz took 1.09, none of 100.
**A robot that streams has no pause that costs a round trip, and nothing here to gain.**

## With 1% packet loss

The 24 KB upload again, with `netem` dropping 1% of the robot's packets.

| RTT | CUBIC, default | CUBIC, the client sends through the pause | BBR on the robot's socket |
|---|---|---|---|
| 22 ms | **2.00 (40 of 40)** | **2.00 (31 of 40)** | 1.01 (0 of 40) |
| 101.5 ms | **2.00 (40 of 40)** | **2.00 (35 of 40)** | not measured |

**Under loss the ping did not help.** The window before each upload was a median of 14 packets, fewer than the upload
needs: the losses had cut it, not the pause, and a ping keeps a window from being restarted but does not grow one. BBR
does not set its window by losses, and took no extra round trip at 22 ms. BBR at 101.5 ms with loss was not measured.

## Over a WebSocket, and over Zenoh

A 100 KB upload 0.53 s after each response, 40 per cell, on a connection first widened with one 1 MB upload
([`bench/ws_bench.py`](../bench/ws_bench.py), [`bench/zenoh_bench.py`](../bench/zenoh_bench.py)). Each cell is the
median in milliseconds and, in brackets, in round trips. Without a component, 40 of 40 uploads took more than one round
trip in every row; with one, none did.

| RTT | WebSocket, as it is | The server pings ([`ping`](../sequence_ai/network/ping.py)) | The client keeps it warm ([`openpi.keep`](../sequence_ai/network/openpi.py)) | BBR alone on the robot's socket | Zenoh, as it is | Zenoh with the keep-alive file ([`lerobot`](../sequence_ai/network/lerobot.py)) |
|---|---|---|---|---|---|---|
| 11.3 ms | **37.6 (3.3)** | 13.4 (1.2) | 12.7 (1.1) | 12.7 (1.1) | **38.1 (3.4)** | 13.9 (1.2) |
| 21.4 ms | **44.4 (2.1)** | 23.5 (1.1) | 22.7 (1.1) | 23.7 (1.1) | **44.8 (2.1)** | 23.2 (1.1) |
| 40.5 ms | **82.4 (2.0)** | 41.8 (1.0) | 41.8 (1.0) | 43.7 (1.1) | **82.8 (2.0)** | 42.3 (1.0) |
| 80.5 ms | **163.8 (2.0)** | 82.8 (1.0) | 82.6 (1.0) | 87.1 (1.1) | **164.3 (2.0)** | 82.2 (1.0) |

What these two columns are:

- **The WebSocket rows use the `websockets` library on both ends**, the library openpi's server and client are built on,
  with compression off as openpi has it. `openpi.keep()` ran on a stand-in for openpi's client: an object with the two
  members `keep()` uses, the connection and `infer`. Not tested with openpi's own policy server and client, or with a
  model. In the `openpi.keep` column the machine had no BBR to give, so what kept the window was the ping.
- **The Zenoh rows are two Zenoh peers**, one answering the other's requests, with and without the file
  `lerobot.keep_alive_file()` writes, given to both. Not tested with LeRobot's own policy server and robot.

## `openpi.keep()` with openpi's own server and client

One closed-loop run with openpi's own policy server (the `pi05_droid` checkpoint, on an L4) and openpi's own client,
with and without `kept = openpi.keep(policy)`. The client ran in a Linux virtual machine (kernel 6.8, default settings,
CUBIC) on a laptop on a home connection; its kernel measured a smoothed RTT of 38 to 66 ms and a retransmission timeout
of about 240 to 270 ms. Each request uploaded 301 KB, and each row is one 45 s episode. "Outside inference" is the time
`policy.infer()` took on the client minus the inference time the server reported (about 195 ms).

| Setting | Outside inference, p50 (p95) | Idle before a request, p50 | Uploads during which the window fell to about half | Retransmissions |
|---|---|---|---|---|
| As it is | **181.4** (237.1) ms | 346 ms | 86 of 89 | 25 |
| As it is | **151.9** (223.5) ms | 378 ms | 85 of 89 | 58 |
| `openpi.keep()` | 92.1 (132.9) ms | 65 ms | 0 of 89 | 0 |
| `openpi.keep()` | 156.7 (334.8) ms | 90 ms | 4 of 87 | 48 |
| `openpi.keep()`, BBR module loaded | 85.6 (109.8) ms | 80 ms | 3 of 90 | 9 |
| `openpi.keep()`, BBR module loaded | 90.6 (136.3) ms | 78 ms | 2 of 90 | 25 |

Without `keep()` the connection was idle for longer than its retransmission timeout before almost every request, and
the kernel reduced the congestion window before the upload. With `keep()` the idle time just before each request was
under 100 ms, and the window fell during 0 to 4 uploads in an episode. Neither measure sees a cut made earlier, at the
robot's first ping after the answer: openpi's server answers no ping while it runs inference, so the robot's ping waits,
and the robot can go longer than a retransmission timeout without sending (below). The second `keep()` episode lost
packets (48 retransmissions) and its median was no better
than without `keep()`: a ping does not help against loss. With the BBR module loaded (`sudo modprobe tcp_bbr`, after
which `kept.bbr` is true) both episodes matched the loss-free `keep()` episode.

This is one link and two episodes per setting, and the client was a virtual machine whose packets the laptop forwards,
not a robot's own Linux computer.

A later run counted where the window is cut. Two Linux cloud machines ran openpi's own client (CUBIC, no BBR), 30 and
75 ms from openpi's own server (the `pi05_droid` checkpoint, on an L4), each six 45 s episodes with and without
`keep()`, 301 KB per request. "Halved before the request" counts the requests whose window just before the request was
half the window after the previous upload: Linux cut it at a send in between, the robot's first ping after a silence
longer than its retransmission timeout.

| Robot | `keep()` | Halved before the request | Window left, p50 (packets) | Outside inference, p50 |
|---|---|---|---|---|
| 30 ms away | no | 3 of 429 | 72 | 91.2 ms |
| 30 ms away | yes | 373 of 482 | 215 | 37.1 ms |
| 75 ms away | no | 3 of 477 | 215 | 86.7 ms |
| 75 ms away | yes | 270 of 477 | 216 | 100.5 ms |

Without `keep()` the window was cut at the request itself, twice at 30 ms (286 to 72 packets). With `keep()` it was cut
once, earlier, and the 301 KB upload (209 packets) still fitted in the 215 left in most requests (79% at 30 ms, all at
75 ms); a larger one would not have. BBR on the robot avoids the restart
([`window.py`](../sequence_ai/network/window.py)).

## Pictures as JPEG with openpi's own server and client

One closed-loop run with openpi's own policy server (the `pi05_droid` checkpoint, on an L4) and openpi's own client,
the robot a simulated arm on a Mac on a home connection, two pictures of 224 x 224 in each request: first as arrays, as
openpi's guide sends them, then as JPEG, each fitted and encoded as `openpi.image_tools` does it, with the server
pinging each connection every 100 ms and decoding the pictures as `openpi.patch()` makes it do. Each row is one 45 s
episode on the same GPU. "Outside the server" is the round trip the client measured minus the inference time the
server reported; the RTT is a TCP echo to the GPU, measured during the episode.

| Pictures | Upload per request | Outside the server, p50 (p95) | RTT, p50 | Frame to answer, p50 (p95) |
|---|---|---|---|---|
| Arrays | 301.5 KB | 153.8 (270.8) ms | 56.5 ms | 351.8 (468.0) ms |
| JPEG, the server pinging | 18.8 KB | **58.1** (207.8) ms | 37.5 ms | 279.6 (429.6) ms |

The link was faster during the second episode, so take its RTT out: the time outside the server beyond one RTT fell
from about 97 ms to about 21 ms. The server took longer for each inference in the second episode, 219.7 against
195.0 ms at the median, while the model's own time was the same (149.8 and 149.3 ms). That server was started through
another script, as `python -m sequence_ai.network.openpi` starts it, and it was as slow with the pings and the decoding
taken out: the difference is in how it was started, and its cause is not known.

This is one link and one episode per setting, and the pings and the JPEG were not measured apart.

## From a Linux client on a home connection

An upload over a WebSocket 0.5 s after the previous response, 16 per cell: the median in milliseconds and, in brackets,
in round trips of that row. The client was the same Linux virtual machine (kernel 6.8, default settings, CUBIC) on a
laptop on a home connection, or the laptop itself (macOS). The server used the `websockets` library, and in the ping
rows it pinged with `ping_interval=0.1`, which is what `ping.websockets_options()` sets.

| Client | RTT | 100 KB | 500 KB | 1 MB |
|---|---|---|---|---|
| Linux, as it is | 34.9 ms | **192.5 (5.5)** | 359.7 (10.3) | 467.0 (13.4) |
| Linux, the server pings every 0.1 s | 35.0 ms | 59.8 (1.7) | 271.6 (7.8) | 445.2 (12.7) |
| Linux, `tcp_slow_start_after_idle=0` | 29.1 ms | 57.7 (2.0) | 392.1 (13.5) | 465.4 (16.0) |
| Linux, BBR on the client's socket | 30.6 ms | 60.0 (2.0) | 157.4 (5.1) | 255.5 (8.3) |
| macOS, as it is | 34.2 ms | 89.4 (2.6) | 135.9 (4.0) | 229.2 (6.7) |
| macOS, the server pings every 0.1 s | 33.1 ms | 89.5 (2.7) | 121.0 (3.7) | 247.2 (7.5) |

At 100 KB the idle restart is the cost: with default settings the window was at 38 segments after 547 ms of idle time,
and the ping, the kernel option and BBR each brought the upload to about two round trips. At 500 KB and 1 MB the line's
upload rate and its packet loss were the cost: six 500 KB uploads sent back to back with default settings took a median
of 14.9 round trips, more than after a pause. There the ping and the kernel option changed little, and BBR took 5.1 and
8.3 round trips where the default settings took 10.3 and 13.4. macOS keeps its window through a pause, and the ping
made no consistent difference there.

This is one home line, a virtual machine and 16 uploads per cell.

## Limitations

- **One clean link for the tables above the last two.** A datacenter network between two machines: no Wi-Fi, no home
  line, no cellular hop, and loss only as `netem`'s random 1%. The two sections above this one are from one home
  connection. On the same kind of connection with a cellular uplink, an earlier measurement found the server's ping
  gave no benefit that six uploads could show: [the README's limitations](../README.md#limitations).
- **No model.** A real policy takes hundreds of milliseconds to respond. A round trip saved is the same number of
  milliseconds either way, and a smaller share of the whole.
- **One pause.** 0.53 s after the response (0.5 s for the sizes). A robot that sends its next request sooner than its
  RTT plus 200 ms never takes the extra round trip; one that pauses for seconds has its window cut to its floor.
- **One upload size for the first table.** At about 100 ms a 24 KB upload is at the boundary described above, and a
  smaller or larger one may fall on the other side of it.
- **The last hop can matter more than the distance.** On the home connection, a laptop that scanned for Wi-Fi
  networks every 20 seconds raised a 30 Hz stream of 50 KB frames from a median of 43 ms per frame to 181 ms.

## Repeat it

On the server side:

```bash
python3 bench/server.py --port 9000
python3 bench/ws_bench.py server 9101 & python3 bench/ws_bench.py server 9102 --ping
python3 bench/zenoh_bench.py server 7447 & python3 bench/zenoh_bench.py server 7448 --keep
```

On the client side (Linux, as root for the first three lines):

```bash
sysctl -w net.ipv4.tcp_no_metrics_save=1 && ip tcp_metrics flush all
tc qdisc replace dev IF root netem delay 30ms                 # a longer RTT: the delay to add; `loss 1%` too
modprobe tcp_bbr                                              # lets a socket request BBR; the default stays CUBIC

python3 bench/client.py --host HOST --sizes 24k --idle-ms 530 --samples 40 --modes idle,ping,keepwarm --warm 1m
python3 bench/client.py --host HOST --sizes 24k --idle-ms 530 --samples 40 --modes idle --warm 1m --bbr
python3 bench/client.py --host HOST --sizes 100k,500k,1m --idle-ms 500 --samples 20 \
    --modes idle,ping,keepwarm,continuous
python3 bench/client.py --host HOST --modes stream --stream 30x50k,10x100k --seconds 10

python3 bench/ws_bench.py client HOST 9101 plain              # as it is
python3 bench/ws_bench.py client HOST 9102 plain              # the server pings
python3 bench/ws_bench.py client HOST 9101 keep               # openpi.keep on the client side
python3 bench/ws_bench.py client HOST 9101 bbr                # BBR alone
python3 bench/zenoh_bench.py client HOST 7447                 # as it is
python3 bench/zenoh_bench.py client HOST 7448 --keep          # the keep-alive file on both ends
```

The kernel option's column is the first command again after `sysctl -w net.ipv4.tcp_slow_start_after_idle=0`.
[`bench/README.md`](../bench/README.md) describes what each block sends.
