# How do I measure the link between my robot and my GPU?

Run one file where the GPU is and one where the robot is. The client prints how many round trips an upload takes after a
pause, with a ping through the pause, back to back, and as a camera stream. That tells you which network components
help on your link and which do not.

```bash
git clone https://github.com/General-Sequences/sequence-ai && cd sequence-ai     # on both machines
python3 bench/server.py --port 9000                        # where the GPU is
python3 bench/client.py --host GPU_ADDRESS --port 9000     # where the robot is
```

Both files are standard library only and run on Linux and macOS. Nothing needs installing, and no model is involved.
With the defaults a run takes about 100 seconds plus the time the uploads take on your link.

## What it prints

This is a run of the defaults with both ends on one machine. Loopback involves no network, so these numbers indicate
nothing about any link. They are here to show the table's format.

```text
host 127.0.0.1  handshake 0.2 ms  baseline rtt min 0.1 / p50 0.2 ms  congestion None
mode            size    p50 ms    p95 ms    p99 ms   x rtt
idle          100000       0.3       0.5         -    1.54
ping          100000       0.2       0.3         -    1.04
continuous    100000       0.0       0.1         -    0.20
idle          500000       0.2       0.4         -    1.14
ping          500000       0.3       0.4         -    1.29
continuous    500000       0.1       0.1         -    0.55
idle         1000000       0.4       0.7         -    2.09
ping         1000000       0.3       0.6         -    1.69
continuous   1000000       0.2       0.3         -    1.14
30Hz           50000       0.2       0.3       0.3    1.00
10Hz          100000       0.2       0.3       0.3    1.19
10Hz          500000       0.3       0.5       0.6    1.29
```

- **The first line** is the link itself: how long the TCP handshake took, the round-trip time (RTT) with no data on it
  (`baseline`), and the congestion control the connection uses (`None` where the system does not report it, as on
  macOS).
- **`mode`** is what was sent, and **`size`** how many bytes each upload had.
- **`p50`, `p95`, `p99`** are how long an upload took to be acknowledged. `p99` is given for streams only, which have
  enough frames for it.
- **`x rtt`** is the median over the baseline RTT: how many round trips an upload took. This is the column to
  read.

It also writes `result.json` with every sample and the kernel's state of the connection around each upload.

## What each block sends

| Block | What it sends | What it tells you |
|---|---|---|
| `baseline` | an empty frame and its ack | the RTT with no data on it |
| `idle` | a frame, then a pause, then the next | what a pause between uploads costs |
| `ping` | as `idle`, the server pinging through the pause | whether a ping from the server side keeps the window |
| `keepwarm` | as `idle`, the robot sending a byte through the pause | the same, done from the client side |
| `continuous` | the next frame as soon as the last is acknowledged | the upload with a warm window |
| `stream` | frames at a fixed rate, acks not waited for | what each frame of a camera stream costs |
| `twowrite` | a frame in two writes with Nagle on | what a header and a body written apart cost |

The defaults run `idle`, `ping`, `continuous` and `stream`, with uploads of 100 KB, 500 KB and 1 MB, 500 ms of
silence, a ping every 100 ms, and 20 samples of each.

## How to read your result

| You see | It means | What to add |
|---|---|---|
| `idle` near 2 or 3, `continuous` near 1 | A pause costs your uploads extra round trips | A ping or BBR: the next two rows show which |
| `ping` or `keepwarm` near 1 | A ping through the pause keeps the window | `connect()` on the robot, or [`ping`](how-do-i-keep-a-websocket-policy-servers-connection-warm.md) on a WebSocket server |
| `ping` no better than `idle` | A ping does not help on this link | Run again with `--bbr`; if that is near 1, use `window` |
| `idle` and `continuous` both near 1 | The pause costs nothing here | Nothing for the pause |
| `stream` rows well above 1 | Frames wait on the link: loss, or the last hop | Look at the last hop first: `wifi_power.check()` on the robot |
| A large `baseline` | The GPU is far away | [A GPU nearer the robot](how-do-i-get-a-gpu-near-my-robot.md) |

## Match the run to your robot

The defaults measure large uploads after half a second of silence. To measure what your robot sends, give its sizes
and its pace:

```bash
# a robot that acts every 0.53 s with 24 KB of frames: the pause, a ping from each side, then BBR
python3 bench/client.py --host HOST --sizes 24k --idle-ms 530 --samples 40 --modes idle,ping,keepwarm --warm 1m
python3 bench/client.py --host HOST --sizes 24k --idle-ms 530 --samples 40 --modes idle --warm 1m --bbr
```

| Flag | What it sets | Default |
|---|---|---|
| `--sizes` | frame sizes for `idle`, `ping`, `keepwarm` and `continuous` | `100k,500k,1m` |
| `--idle-ms` | the silence after each ack | `500` |
| `--ping-ms` | the interval of the pings and of the robot's bytes | `100` |
| `--stream` | camera streams, as `HZxSIZE` | `30x50k,10x100k,10x500k` |
| `--seconds` | how long each stream runs | `10` |
| `--samples` | samples for each block and size | `20` |
| `--modes` | the blocks to run, in order | `idle,ping,continuous,stream` |
| `--bbr` | request BBR on this connection's socket | off |
| `--warm` | upload one frame of this size before measuring | none |
| `--out` | where the samples are written | `result.json` |

## Try other distances from one pair of machines

On Linux, as root, where the client sends from:

```bash
sudo tc qdisc add dev eth0 root netem delay 80ms          # about 80 ms more on the round trip
sudo tc qdisc change dev eth0 root netem delay 80ms loss 1%
sudo tc qdisc del dev eth0 root
```

## Reference measurements

| Link | Upload | After 500 ms of silence | Server pinging every 100 ms | One after another |
|---|---|---|---|---|
| Three machines of one cloud provider, RTTs of 11.7 and 69.2 ms, 20 samples per cell | 100 KB | 3.0 round trips | 1.1 | 1.05 |
| the same | 500 KB | 3.0 | 1.1 | 1.1 |
| the same | 1 MB | 3.0 | 1.25 | 1.3 |

[`bench/README.md`](../bench/README.md) has these in full, and
[Why does my robot's upload get slow after a pause?](why-does-my-robots-upload-get-slow-after-a-pause.md) has the
table for a robot that sends two requests per second.

## Measure a server that is already running

The tool above measures the link with its own server. To see the link while your policy server runs, read it off the
server's kernel (Linux):

```python
from sequence_ai.network import tcp_link

tcp_link.read(sock_user)             # any connected socket: rtt_ms, rttvar_ms, retrans, total_retrans, cwnd
tcp_link.of(scope_user["client"])    # in an ASGI app served with one_write_http(): this request's own connection
```

and on the robot, `link.stats()["latency_us"]["act_us"]` gives each request's p50, p95 and p99 as the robot saw them.
