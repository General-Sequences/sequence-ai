# How do I keep a WebSocket policy server's connection warm?

Have the server ping each connection every 100 ms. A WebSocket library answers a ping with a pong by itself, so a
server that pings makes the robot send a few bytes each interval, and the robot's kernel never counts the connection
as idle. Nothing changes on the robot: not its program, and not its library.

With the `websockets` library:

```python
import websockets
from sequence_ai.network import ping

# websockets' own server, with a ping every 100 ms
async with ping.serve(handler_user, host_user, port_user) as server_user:
    await server_user.serve_forever()

# the same, as its options
async with websockets.serve(handler_user, host_user, port_user, **ping.websockets_options()):
    ...
```

Under uvicorn:

```python
import uvicorn
from sequence_ai.network import ping

uvicorn.run(app_user, **ping.uvicorn_options())
```

```bash
pip install sequence-ai          # this component imports only the standard library, and websockets for ping.serve
```

`ping.websockets_options()` and `ping.uvicorn_options()` return plain options, so you can also write them out and
install nothing:

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
A WebSocket server sends its next ping one interval after the last one (after its pong, in websockets' legacy
implementation), so the robot is silent for the interval plus one RTT at most. An interval under 200 ms is therefore sufficient on a link of any length, and at 100 ms a
ping that is lost still leaves the next one inside the limit.

Measured on an RTT of 11.7 ms: pings 190 ms apart still kept the window (an upload after a pause took 1.18
round trips), and pings 250 ms apart did not (2.04). Anything from 25 to 190 ms apart measured the same.

## What was measured

| Link | Upload | Without the ping | With it |
|---|---|---|---|
| Machines of one cloud provider, RTTs of 11.7 ms and a longer one, 20 samples per cell | 100 KB after 500 ms of silence | 3.0 round trips | 1.1 |
| the same | 500 KB | 3.0 | 1.1 |
| the same | 1 MB | 3.0 | 1.25 |
| Two cloud machines, RTTs of 11, 21 and 41 ms and a longer one, 40 uploads per cell, over a WebSocket whose server has `ping.websockets_options()` | 100 KB, 0.53 s after each response | 3.3, 2.1, 2.0 and 2.0 round trips | 1.2, 1.1, 1.0 and 1.0 |

A window that the pings do not fill still shrinks. BBR on the robot keeps that one: `window.keep(sock)`, or
`connect()`. With 1% loss, BBR kept the window: none of 40 uploads took an extra round trip
([Measurements](measurements.md#with-1-packet-loss)).

## When the server cannot ping

**A plain HTTP server.** A server that answers one HTTP request with one response has no way to send anything
unsolicited. There the ping comes from the robot, on the connection your client holds:

```python
import httpx
from sequence_ai.network import window

client_user = httpx.Client(base_url="http://YOUR_GPU:8000")
warm_user = window.Warm(ping=lambda: client_user.get("/ping"))     # a small GET after each 100 ms of idle time
warm_user.used()                                                   # call it each time you send on the connection
```

**A WebSocket server you cannot change.** Ping from the client side, on the connection your client already holds.
With your own `websockets` client:

```python
from websockets.sync.client import connect as ws_connect
from sequence_ai.network import window

with ws_connect("ws://YOUR_GPU:8000") as ws_user:
    window.keep(ws_user.socket)                               # BBR on its socket, where the machine allows

    def ping_user() -> None:
        if not ws_user.ping().wait(5.0):                      # no pong in 5 s: the connection is gone
            raise TimeoutError("no pong")

    warm_user = window.Warm(ping_user)                        # a ping after each 100 ms of idle time
    ws_user.send(observation_user)
    response_user = ws_user.recv()
    warm_user.used()                                          # after each request and response
```

Run against a stand-in server on one machine, that client sent four pings in 0.45 seconds of idle time.

## Tested

- `tests/network/test_ping.py` starts a `websockets` server with the options and checks that it pings every interval and
  that one without them does not; under uvicorn, that a server with the options pings every interval.
- uvicorn 0.54 takes the options on its `auto`, `websockets-sansio` and `websockets` implementations. uvicorn 0.39's
  `websockets-sansio` sends no pings, so on an older uvicorn leave `ws` at `auto` or pass `ws="websockets"`.
- The first three rows of the table were taken with the tool in [`bench/`](../bench/), whose server pings on
  request, and not with a policy server.
- The WebSocket row was taken with [`bench/ws_bench.py`](../bench/ws_bench.py): a `websockets` server with the
  options, and no policy behind it. [What do the components save?](measurements.md) has every cell.
