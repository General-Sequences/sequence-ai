# What do the components save, measured with the tools in this repository?

On a clean link between two cloud machines, an upload that follows a robot's half-second pause took two to three round
trips with the defaults, and one with any of the components that keep the connection from going idle. That held over
plain TCP, over a WebSocket and over Zenoh, at every round-trip time (RTT) measured from 11 ms up but the longest, where
the pause cost nothing; under 1% packet loss, BBR kept the window.

The TCP, WebSocket and Zenoh sections were measured on 2026-11-15 with [`bench/`](../bench/), as released; the openpi
sections with openpi's own server and client. It is one pair of machines
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

On three longer links every default upload took two round trips as well, and every other column one; on a much longer
link the default took one too. In milliseconds on the two shortest links, the default against the server's ping: 24.7
against 12.5, and 42.7 against 21.5.

**Why the longest link took no extra round trip.** The kernel halves an idle connection's window once for each
retransmission timeout that passed, and the timeout is about the RTT plus 200 ms. On every other link, the pause plus
one RTT spanned two timeouts: the window, 28 packets before each upload, was cut to 10, the least it is cut to, and the
upload left in two flights. On the longest, whose timeout is longer as well, they spanned one: the window, 36 packets
before each upload, became 18, and the whole upload left at once.

**Why this differs from an earlier measurement on a longer link.** With a robot that uploaded on a fixed period of
0.53 s, counted from the start of one upload to the start of the next, the same 24 KB upload took no extra round trip
on the longest of the three longer links: that silence spans one timeout, not two. Here the 0.53 s is counted from the
response, which matches a robot that starts its pause when the response arrives. **On that link the two ways of
counting the same half second fall on either side of the edge.** A robot's pause, its RTT and its upload's size decide
together, and a small change in any of them can move a link across.

## Larger uploads

Uploads 0.5 s after each response, with no widening first
(`bench/client.py --sizes 100k,500k,1m --idle-ms 500 --samples 20 --modes idle,ping,keepwarm,continuous`). Median in
round trips; in brackets, how many of the 20 took more than one.

| RTT | Upload | Default | The server pings | The client sends | One after another, no pause |
|---|---|---|---|---|---|
| 13.2 ms | 100 KB | **2.98 (20)** | 1.10 (0) | 1.08 (0) | 1.06 (0) |
| 13.2 ms | 500 KB | **2.99 (20)** | 1.23 (1) | 1.24 (0) | 1.15 (0) |
| 13.2 ms | 1 MB | **3.01 (20)** | 1.31 (1) | 1.36 (0) | 1.35 (8) |

On a longer link the default took two to three round trips as well, and each of the others about one, at every size.

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

On a longer link with the same loss, both CUBIC columns took two round trips as well; BBR was not run there.

**Under loss, BBR keeps the window.** BBR does not set its window by losses, and took no extra round trip at 22 ms.

## Over a WebSocket, and over Zenoh

A 100 KB upload 0.53 s after each response, 40 per cell, on a connection first widened with one 1 MB upload
([`bench/ws_bench.py`](../bench/ws_bench.py), [`bench/zenoh_bench.py`](../bench/zenoh_bench.py)). Each cell is the
median in milliseconds and, in brackets, in round trips. Without a component, 40 of 40 uploads took more than one round
trip in every row; with one, none did.

| RTT | WebSocket, as it is | The server pings ([`ping`](../sequence_ai/network/ping.py)) | The client keeps it warm ([`openpi.keep`](../sequence_ai/network/openpi.py)) | BBR alone on the robot's socket | Zenoh, as it is | Zenoh with the keep-alive file ([`lerobot`](../sequence_ai/network/lerobot.py)) |
|---|---|---|---|---|---|---|
| 11.3 ms | **37.6 (3.3)** | 13.4 (1.2) | 12.7 (1.1) | 12.7 (1.1) | **38.1 (3.4)** | 13.9 (1.2) |
| 21.4 ms | **44.4 (2.1)** | 23.5 (1.1) | 22.7 (1.1) | 23.7 (1.1) | **44.8 (2.1)** | 23.2 (1.1) |

At 41 ms and on a longer link, an upload took two round trips as it is, over a WebSocket and over Zenoh, and one with
any of the components.

What these two columns are:

- **The WebSocket rows use the `websockets` library on both ends**, the library openpi's server and client are built on,
  with compression off as openpi has it. `openpi.keep()` ran on a stand-in for openpi's client: an object with the two
  members `keep()` uses, the connection and `infer`. In the `openpi.keep` column the machine had no BBR to give, so what kept the window was the ping.
- **The Zenoh rows are two Zenoh peers**, one answering the other's requests, with and without the file
  `lerobot.keep_alive_file()` writes, given to both.

## `openpi.keep()` with openpi's own server and client

One closed-loop run with openpi's own policy server (the `pi05_droid` checkpoint, on an L4) and openpi's own client,
with and without `kept = openpi.keep(policy)`. The client ran in a Linux virtual machine (kernel 6.8, default settings,
CUBIC) on a laptop on a home connection; its kernel measured a smoothed RTT from 38 ms up and a retransmission timeout
of about 240 to 270 ms. Each request uploaded 301 KB, and each row is one 45 s episode. "Outside inference" is the time
`policy.infer()` took on the client minus the inference time the server reported (about 195 ms).

| Setting | Idle before a request, p50 | Uploads during which the window fell to about half | Retransmissions |
|---|---|---|---|
| As it is | 346 ms | **86 of 89** | 25 |
| As it is | 378 ms | **85 of 89** | 58 |
| `openpi.keep()` | 65 ms | 0 of 89 | 0 |
| `openpi.keep()`, BBR module loaded | 80 ms | 3 of 90 | 9 |
| `openpi.keep()`, BBR module loaded | 78 ms | 2 of 90 | 25 |

Without `keep()` the connection was idle for longer than its retransmission timeout before almost every request, and the
kernel reduced the congestion window before the upload. With `keep()` the idle time just before each request was under
100 ms, the window fell during 0 to 3 uploads in an episode, and the time outside inference was about 40 to 50% shorter
at the median. With the BBR module loaded as well (`sudo modprobe tcp_bbr`, after which `kept.bbr` is true), it was a
little shorter still.

A later run had a Linux cloud machine run openpi's own client (CUBIC, no BBR) 30 ms from openpi's own server (the
`pi05_droid` checkpoint, on an L4), six 45 s episodes with and without `keep()`, 301 KB per request.

| Robot | `keep()` | Window left, p50 (packets) | Outside inference, p50, in round trips of 30 ms |
|---|---|---|---|
| 30 ms away | no | 72 | about 3 |
| 30 ms away | yes | 215 | about 1.2 |

With openpi's own server on Linux, also load BBR (`sudo modprobe tcp_bbr`, which `keep()` then puts on the connection)
so the window holds across a pause ([`window.py`](../sequence_ai/network/window.py)).

## Pictures as JPEG with openpi's own server and client

One closed-loop run with openpi's own policy server (the `pi05_droid` checkpoint, on an L4) and openpi's own client,
the robot a simulated arm on a Mac on a home connection, two pictures of 224 x 224 in each request: first as arrays, as
openpi's guide sends them, then as JPEG, each fitted and encoded as `openpi.image_tools` does it, with the server
pinging each connection every 100 ms and decoding the pictures as `openpi.patch()` makes it do. Each row is one 45 s
episode on the same GPU. "Outside the server" is the round trip the client measured minus the inference time the
server reported; the RTT is a TCP echo to the GPU, measured during the episode.

| Pictures | Upload per request | RTT, p50 |
|---|---|---|
| Arrays | 301.5 KB | 56.5 ms |
| JPEG, the server pinging | 18.8 KB | 37.5 ms |

The link was faster during the second episode, so take its RTT out: the time outside the server beyond one RTT fell to
about a fifth, to about 21 ms at the median.

## From a Linux client on a home connection

An upload over a WebSocket 0.5 s after the previous response, 16 per cell: the median, in round trips of that row. The
client was the same Linux virtual machine (kernel 6.8, default settings, CUBIC) on a laptop on a home connection, or the
laptop itself (macOS). The server used the `websockets` library, and in the ping rows it pinged with
`ping_interval=0.1`, which is what `ping.websockets_options()` sets.

| Client | RTT | 100 KB | 500 KB | 1 MB |
|---|---|---|---|---|
| Linux, as it is | 34.9 ms | **5.5** | 10.3 | 13.4 |
| Linux, the server pings every 0.1 s | 35.0 ms | 1.7 | 7.8 | 12.7 |
| Linux, `tcp_slow_start_after_idle=0` | 29.1 ms | 2.0 | 13.5 | 16.0 |
| Linux, BBR on the client's socket | 30.6 ms | 2.0 | 5.1 | 8.3 |
| macOS, as it is | 34.2 ms | 2.6 | 4.0 | 6.7 |
| macOS, the server pings every 0.1 s | 33.1 ms | 2.7 | 3.7 | 7.5 |

At 100 KB the idle restart is the cost: with default settings the window was at 38 segments after 547 ms of idle time,
and the ping, the kernel option and BBR each brought the upload to about two round trips. At 500 KB and 1 MB the line's
upload rate and its packet loss were the cost: six 500 KB uploads sent back to back with default settings took a median
of 14.9 round trips, more than after a pause. There the ping and the kernel option changed little, and BBR took 5.1 and
8.3 round trips where the default settings took 10.3 and 13.4. macOS keeps its window through a pause, and the ping
made no consistent difference there.

This is one home line, a virtual machine and 16 uploads per cell.

## Scope of these measurements

- **One datacenter link** for the sections above the last three; those three are from one home connection.
- **One pause.** 0.53 s after the response (0.5 s for the sizes). A robot that sends its next request sooner than its
  RTT plus 200 ms never takes the extra round trip; one that pauses for seconds has its window cut to its floor.
- **One upload size for the first table.** On the longest of the three longer links a 24 KB upload is at the boundary
  described above, and a smaller or larger one may fall on the other side of it.

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
