# How do I keep a WebSocket policy server's connection warm?

Have the server ping each connection every 100 ms. A WebSocket library answers a ping with a pong by itself, so a
server that pings makes the robot send a few bytes each interval, and the robot's kernel never counts the connection
as idle. Nothing changes on the robot: not its program, and not its library.

With the `websockets` library:

```python
import websockets
from sequence_ai.network import ping

async with websockets.serve(handler, host, port, **ping.websockets_options()):    # a ping every 100 ms
    ...
```

Under uvicorn:

```python
uvicorn.run(app, **ping.uvicorn_options())
```

```bash
pip install sequence-ai          # this component imports only the standard library
```

The two calls return plain options, so you can also write them out and install nothing:

```text
ping.websockets_options()  ->  {'ping_interval': 0.1, 'ping_timeout': 20.0}
ping.uvicorn_options()     ->  {'ws_ping_interval': 0.1, 'ws_ping_timeout': 20.0}
```

Without them, `websockets` pings every 20 seconds (the default of its `serve` in version 17.4). That finds a dead
connection. It is far too slow to keep a connection from going idle between a robot's requests.

## Why it matters

On Linux, a connection that has sent nothing for longer than its retransmission timeout has its congestion window
restarted, so the robot's next observation can take more round trips to upload. What restarts it is the robot's own
silence, which is why a ping the robot must answer fixes it from the server side.
[Why does my robot's upload get slow after a pause?](why-does-my-robots-upload-get-slow-after-a-pause.md) has the
mechanism in full.

## Try it

[`examples/components/ping.py`](../examples/components/ping.py) starts a server with the options, connects one client
and leaves it idle for half a second:

```bash
pip install websockets
python examples/components/ping.py
```

```text
response: observation
the server measured the connection's RTT from its pings: 0.17 ms
```

The second line is a side benefit: each pong tells the server the connection's round-trip time (RTT), which `websockets`
keeps as `connection.latency`.

## Why 100 ms

The robot's silence must stay shorter than its retransmission timeout, which is about its RTT plus 200 ms.
A WebSocket server sends its next ping one interval after the last pong came back, so the robot is silent for the
interval plus one RTT. An interval under 200 ms is therefore sufficient on a link of any length, and at 100 ms a
ping that is lost still leaves the next one inside the limit.

Measured on an RTT of 11.7 ms: pings 190 ms apart still kept the window (an upload after a pause took 1.18
round trips), and pings 250 ms apart did not (2.04). Anything from 25 to 190 ms apart measured the same.

## What was measured

| Link | Upload | Without the ping | With it |
|---|---|---|---|
| Machines of one cloud provider, RTTs of 11.7 and 69.2 ms, 20 samples per cell | 100 KB after 500 ms of silence | 3.0 round trips | 1.1 |
| the same | 500 KB | 3.0 | 1.1 |
| the same | 1 MB | 3.0 | 1.25 |
| Two cloud machines, RTTs of 11, 21, 41 and 81 ms, 40 uploads per cell, over a WebSocket whose server has `ping.websockets_options()` | 100 KB, 0.53 s after each response | 3.3, 2.1, 2.0 and 2.0 round trips | 1.2, 1.1, 1.0 and 1.0 |
| A home connection with a cellular uplink, about 36 ms from the GPU, six uploads each, alternating | 921,600 bytes after 10 s of silence | a median of 264 ms | 368 ms: no benefit |

A window that the pings do not fill still shrinks. BBR on the robot keeps that one: `window.keep(sock)`, or
`connect()`. And a ping does not grow a window that packet loss has cut: with 1% loss, 31 and 35 of 40 uploads
still took one extra round trip with a ping from the client side, and none of 40 with BBR
([Measurements](measurements.md#with-1-packet-loss)).

## When the server cannot ping

**A plain HTTP server.** A server that answers one HTTP request with one response has no way to send anything
unsolicited. There the ping comes from the robot. `connect()` does it by itself:

```python
from sequence_ai.network import connect

link = connect("http://YOUR_GPU:8000")     # a small GET after each 100 ms of idle time
```

**A WebSocket server you cannot change.** Ping from the client side, on the connection your client already holds.
With openpi's client it is one line, `openpi.keep(policy)`: see
[How do I run π0 (openpi) on a remote GPU?](how-do-i-run-pi0-openpi-on-a-remote-gpu.md) With your own `websockets`
client:

```python
from websockets.sync.client import connect as ws_connect
from sequence_ai.network import window

with ws_connect("ws://YOUR_GPU:8000") as ws:
    window.keep(ws.socket)                               # BBR on its socket, where the machine allows

    def ping() -> None:
        if not ws.ping().wait(5.0):                      # no pong in 5 s: the connection is gone
            raise TimeoutError("no pong")

    warm = window.Warm(ping)                             # a ping after each 100 ms of idle time
    ws.send(observation)
    response = ws.recv()
    warm.used()                                          # after each request and response
```

Run against a stand-in server on one machine, that client sent four pings in 0.45 seconds of idle time.

## Tested

- `tests/network/test_ping.py` starts a `websockets` server with the options and checks that it pings every interval and
  that one without them does not. It does the same under uvicorn.
- uvicorn 0.54 takes the options on its `auto`, `websockets-sansio` and `websockets` implementations. uvicorn 0.39's
  `websockets-sansio` sends no pings, so on an older uvicorn leave `ws` at `auto` or pass `ws="websockets"`.
- The first three rows of the table were taken with the tool in [`bench/`](../bench/), whose server pings on
  request, and not with a policy server.
- The WebSocket row was taken with [`bench/ws_bench.py`](../bench/ws_bench.py): a `websockets` server with the
  options, and no policy behind it. [What do the components save?](measurements.md) has every cell.
