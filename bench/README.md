# bench: measure your own link

What a network component of SequenceLink saves depends on the link between your robot and your GPU. These two files measure
that link, so you add the components that help on it and leave the ones that do not.

```bash
python3 bench/server.py --port 9000                       # where the GPU is
python3 bench/client.py --host GPU_ADDRESS --port 9000    # where the robot is
```

Standard library only, Linux and macOS, one TCP connection. The client prints a table and writes `result.json` with
every sample and the kernel's state of the connection around each upload.

## What it measures

| Block | What it sends | What it tells you |
|---|---|---|
| `baseline` | an empty frame and its ack | the RTT with no data on it |
| `idle` | a frame, then a pause, then the next | what a pause between uploads costs |
| `ping` | as `idle`, the server pinging through the pause | whether a ping from the server side keeps the window |
| `keepwarm` | as `idle`, the robot sending a byte through the pause | the same, done from the client side |
| `continuous` | the next frame as soon as the last is acknowledged | the upload with a warm window |
| `stream` | frames at a fixed rate, without waiting for acks | the cost per frame of a camera stream |
| `twowrite` | a frame in two writes with Nagle on | the cost of writing a header and a body separately |

`x rtt` in the table is a block's median over the baseline round-trip time (RTT): how many round trips an upload took.

## Two measurements to repeat

**A robot that sends a request every 0.53 s with 24 KB of frames.**

```bash
python3 bench/client.py --host HOST --sizes 24k --idle-ms 530 --samples 40 --modes idle,ping,keepwarm --warm 1m
python3 bench/client.py --host HOST --sizes 24k --idle-ms 530 --samples 40 --modes idle --warm 1m --bbr
```

**Uploads of 100 KB to 1 MB after half a second of silence, and camera streams.**

```bash
python3 bench/client.py --host HOST          # the defaults
```

To try other distances from one pair of machines, add delay or loss where the client sends from (Linux, root):

```bash
sudo tc qdisc add dev eth0 root netem delay 40ms          # about 40 ms more on the round trip
sudo tc qdisc change dev eth0 root netem delay 40ms loss 1%
sudo tc qdisc del dev eth0 root
sudo sysctl -w net.ipv4.tcp_slow_start_after_idle=0       # the kernel setting the pause goes through
sudo sysctl -w net.ipv4.tcp_no_metrics_save=1 && sudo ip tcp_metrics flush all    # between runs
```

## Measured results

These are the runs behind the numbers in the top-level README. Each states its conditions, because each result changed
with them.

**After a pause, an upload took three round trips; with a ping it took about one.** Three machines of one cloud
provider, a near link of 11.7 ms and a far one, this tool, 20 samples per cell:

| Upload | After 500 ms of silence | Server pinging every 100 ms | One after another |
|---|---|---|---|
| 100 KB | 3.0 round trips | 1.1 | 1.05 |
| 500 KB | 3.0 | 1.1 | 1.1 |
| 1 MB | 3.0 | 1.25 | 1.3 |

The ping must be sent more often than the sender's retransmission timeout, which is about its RTT plus 200 ms.
On the near link, pings 190 ms apart still kept the window (1.18 round trips) and pings 250 ms apart did not (2.04).
Anything from 25 to 190 ms apart measured the same.

**A robot sending a request every 0.53 s took one extra round trip on every request, on a short link.** Two Linux
machines (kernels 6.8 and 7.0), an HTTP server, 24 KB per request, 40 requests per cell, the RTT set with netem. This
was measured with an earlier tool; the first command above repeats it.

| Setting | a short link, about 21 ms | a longer link | a much longer link |
|---|---|---|---|
| default | 40 of 40 requests took one extra round trip | 0 of 40 | 0 or 1 of 40 |
| `tcp_slow_start_after_idle=0` | 0 | 0 | 0 |
| a small request about every 100 ms between requests | 0 | 0 | 0 |
| BBR as the system's default | 0 | 0 | 0 |
| default, 1% loss throughout | 39 or 40 | 39 or 40 | 38 |
| default, request size varying by up to 8% | not run | 19 of 40 | 10 of 40 |

So on a clean link the cost appears where the RTT is short, and returns on longer links once the link loses
packets or the request size varies.

**Measure yours.** The two commands at the top run these measurements on the link between your robot and your
GPU.

## A WebSocket link and a Zenoh link

`client.py` speaks its own small protocol over TCP. Two more files measure the same pause on the transports a robot
may already use:

```bash
python3 bench/ws_bench.py server 9101                     # server side; --ping: the server pings every 100 ms
python3 bench/ws_bench.py client HOST 9101 plain          # client side: plain, or bbr (BBR on its socket)

python3 bench/zenoh_bench.py server 7447
python3 bench/zenoh_bench.py client HOST 7447
```

Each client uploads 100 KB half a second after each response, 40 times, and prints one `RESULT` line with every
sample. `ws_bench.py` needs `websockets`, `zenoh_bench.py` needs `eclipse-zenoh`. They configure the `websockets`
library and Zenoh as openpi and LeRobot 0.7 do; they are not openpi's or LeRobot's own server and client.

## The run made with these tools

On 2026-11-15 the three tools were run between two cloud machines 12 ms apart, with delay added for longer links. A
24 KB upload after a 0.53 s pause took two round trips on 40 of 40 uploads at every RTT but the longest, and one with
the server's ping, the robot's ping, BBR on the robot's socket or the kernel option at 0. On the longest it took one
either way. Over a WebSocket and over Zenoh, a 100 KB upload took two to three round trips as they are and one with the
components. With 1% loss, BBR kept the window.
[`docs/measurements.md`](../docs/measurements.md) has every table, the commands and the scope of these measurements.
