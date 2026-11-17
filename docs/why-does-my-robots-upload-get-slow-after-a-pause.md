# Why does my robot's upload get slow after a pause?

Because the robot's kernel shrank the connection's congestion window while the connection was silent. On Linux, a TCP
connection that has sent nothing for longer than its retransmission timeout, which is at least its round-trip time (RTT)
plus 200 ms, has its congestion window cut back before it sends again. The next observation then leaves in two or three
flights instead of one, and each extra flight waits a full round trip.

A robot that executes an action chunk and then requests the next one pauses like this on every request.

The fix is to keep the window through the pause. On the robot, `connect()` does it:

```python
from sequence_ai.network import connect

link = connect("http://YOUR_GPU:8000")     # BBR where the machine allows it, and a ping while the connection is idle
```

Or change one setting on the robot, as root, and keep your own client:

```bash
sudo sysctl -w net.ipv4.tcp_slow_start_after_idle=0     # the kernel stops restarting the window
sudo modprobe tcp_bbr                                    # or: load BBR, which window.socket_options() then uses
```

To see whether your robot's machine does this at all:

```python
from sequence_ai.network import window

window.restarts_after_idle()     # True on Linux with the default setting; False where it cannot be read
window.check()                   # warns once, with both fixes, and changes nothing
```

## It depends on the link: measured

A robot sending a request every 0.53 s with 24 KB of frames (19 packets), 40 requests per cell. Two Linux machines,
kernels 6.8 and 7.0, an HTTP server, the RTT set with netem. Each cell is how many of the 40 requests took one extra
round trip.

| Setting on the robot | about 21 ms | about 101 ms | about 241 ms |
|---|---|---|---|
| default | 40 of 40 | 0 of 40 | 0 or 1 of 40 |
| `tcp_slow_start_after_idle=0` | 0 | 0 | 0 |
| a small request about every 100 ms between requests | 0 | 0 | 0 |
| BBR as the system's default | 0 | 0 | 0 |
| default, 1% loss throughout | 39 or 40 | 39 or 40 | 38 |
| default, request size varying by up to 8% (kernel 6.8 only) | not measured | 19 of 40 | 10 of 40 |

On the short link every request took an extra round trip: its upload took a median of 21.5 ms to arrive, against 0.5 to
1.4 ms with any of the three fixes. On the two longer links the pause cost nothing until the link lost packets or the
request size varied.

**Why the short link took the extra round trip and the long ones did not.** The window is halved once for each timeout
the connection sat idle, down to 10 packets. A 0.53 s pause spans two timeouts at an RTT of about 21 ms, so a window of
29 packets became 10, and a 19-packet upload no longer fit in one flight. At about 101 and 241 ms the same pause spans
one timeout: 38 packets became 19, which just fits. A larger upload, or a longer pause, moves that line.

**The same upload, with the half second counted from the response.** Measured again with the tool in
[`bench/`](../bench/), which waits 0.53 s after each response before it uploads again, between two cloud machines: every
one of 40 uploads took one extra round trip at 12, 21, 41, 65, 81 and 101 ms, and none at 242 ms. With the server
pinging, the client sending a byte, BBR on the robot's socket or the kernel option at 0, none took an extra round trip
at any RTT. The table above had a fixed 0.53 s from the start of one upload to the start of the next: at about 101 ms
that silence spans one timeout, and half a second after the response plus one RTT spans two. **At about 100 ms, how a
robot's pause is counted decides whether it takes the extra round trip.**
[What do the components save?](measurements.md) has the table.

**Larger uploads.** After 500 ms of silence, uploads of 100 KB, 500 KB and 1 MB each took 3.0 round trips. With the
server pinging every 100 ms they took 1.1, 1.1 and 1.25; sent one after another with no pause, 1.05, 1.1 and 1.3.
Three machines of one cloud provider, RTTs of 11.7 and 69.2 ms, 20 samples per cell.

## Where keeping it warm did not help

- **A home connection.** A robot on a home connection with a cellular uplink, about 36 ms from the GPU, uploading
  921,600 bytes after 10 s of silence: a median of 264 ms without the server's pings and 368 ms with them. Six uploads
  each, alternating. The pings showed no benefit there.
- **A large upload over Wi-Fi.** From 85 ms away, 2.36 MB after a 2 s pause took a median of 504 ms without a ping,
  421 ms with one every 100 ms, and 396 ms sent at once. Three of each, which is too few to tell the last two apart.
- **Large uploads from Linux on a home connection.** At an RTT of about 35 ms, 500 KB and 1 MB uploads 0.5 s after
  the previous one took 10.3 and 13.4 round trips with default settings and 7.8 and 12.7 with the server's ping: the
  line's upload rate and packet loss were the cost. BBR took 5.1 and 8.3. At 100 KB the ping did help: 5.5 round trips
  became 1.7. [The table](measurements.md#from-a-linux-client-on-a-home-connection) has every cell.
- **macOS.** A Mac keeps a connection's window through three minutes of idle, so a robot program on a Mac does not
  incur this.

## Four ways to keep the window, and what each cannot do

| Way | Needs | Keeps a window that the pings do not fill |
|---|---|---|
| BBR on the connection: `window.socket_options()`, `window.keep(sock)` | Linux with the `tcp_bbr` module loaded | yes |
| `net.ipv4.tcp_slow_start_after_idle=0` | root on the robot | yes |
| A ping from the robot while idle: `window.Warm`, which `connect()` uses | nothing | no |
| A ping from the server: [`ping`](how-do-i-keep-a-websocket-policy-servers-connection-warm.md) | a WebSocket server | no |

A ping keeps the connection from counting as idle. It does not stop Linux from shrinking a window that the traffic
does not fill, which small pings between large uploads do not. BBR and the kernel setting keep that window as well.
`connect()` requests BBR and pings, so it has both where the machine allows BBR.

The ping interval must be shorter than the timeout. On a link of 11.7 ms, pings 190 ms apart still kept the window
(1.18 round trips for an upload after a pause) and pings 250 ms apart did not (2.04). Anything from 25 to 190 ms
apart measured the same. The components ping every 100 ms.

## Two other things a pause can cost

- **Wi-Fi power saving.** With it enabled, a Wi-Fi adapter sleeps between the access point's beacons, and the
  response to the robot's first packets is delayed until the adapter wakes. `wifi_power.check()` detects it and
  reports how to disable it.
- **A connection the network forgot.** A home or office router can drop its record of a silent connection, and the
  next send is dropped until it times out. TCP keepalive, which `window.socket_options()` enables, detects that
  before the robot's next send does.

## See it on your own link

```bash
python3 bench/server.py --port 9000                                                      # where the GPU is
python3 bench/client.py --host GPU_ADDRESS --sizes 24k --idle-ms 530 --samples 40 --modes idle,ping,keepwarm --warm 1m
python3 bench/client.py --host GPU_ADDRESS --sizes 24k --idle-ms 530 --samples 40 --modes idle --warm 1m --bbr
```

The `x rtt` column reports how many round trips each upload took. `idle` near 2 with `keepwarm` near 1 is this problem,
and the components fix it. All near 1 means your link does not have it.
[How do I measure the link between my robot and my GPU?](how-do-i-measure-the-link-between-my-robot-and-my-gpu.md)
explains every column.

## Sources

The measurements are from [`bench/README.md`](../bench/README.md) and the docstrings of
[`window.py`](../sequence_ai/network/window.py) and [`ping.py`](../sequence_ai/network/ping.py), which also name the kernel
code that does it (`tcp_cwnd_restart` and `tcp_cwnd_validate` in `tcp_output.c`, RFC 5681 section 4.1 and RFC 2861).
The table for a robot sending a request every 0.53 s was measured with an earlier version of the tool in `bench/`.
