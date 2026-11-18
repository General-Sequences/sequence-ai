# What is on by default, and how do I turn each part off?

Everything on this page is on as soon as the GPU runs `sequence_ai.serve(policy)` and the robot runs
`sequence_ai.connect(url)`: nothing needs to be added. The last column of each table is how to change a part.

```python
sequence_ai.serve(policy_user, port=8000, example=observation_user)      # on the GPU
link_user = sequence_ai.connect("http://YOUR_GPU:8000")            # on the robot
```

## On the GPU: `serve()`

| Part | What it does by default | To change it |
|---|---|---|
| The policy's thread | The policy runs on one thread of its own, one call at a time. Requests that arrive meanwhile wait their turn, in order; pings and uploads are not held behind it | always on |
| Batching | Off: one observation per call | `batch=[1, 2, 4, 8]`: the policy takes a list of the observations waiting and returns one output for each, in order |
| Addresses | Every address of the machine: `0.0.0.0`, and `::` where it has IPv6 | `host="127.0.0.1"`: this machine only |
| Pictures as JPEG | Says it reads them (`X-Seq-Reads: jpeg`), so the robot sends each picture as JPEG; each is turned back into its array before the policy sees it | `jpeg=False`: the robot sends pictures as arrays, exactly |
| Compression | Says it reads zstd and zlib, so the robot compresses large arrays where that saves time; they are restored byte for byte | `compress=False` |
| Ping | `GET /ping` is answered at once: `503` with `Retry-After` until the warm-up is done, then `200` with what the server is, as JSON (`server`, `version`, `up_s`, `pid`, `policy`) | always on |
| Identity | `X-Seq-Server: sequence-ai/<version>` on every response, so a robot can tell `serve()` from another program at its address | always on |
| Warm-up | With `example=`, the policy runs on it twice at each batch size before `/ping` says ready, its pictures sent as JPEG as a robot's are; `GET /example` hands it out | leave out `example` |
| Inputs | The observation as the robot sent it, numpy arrays as arrays, each the policy's own to write into | always on |
| Outputs | torch tensors and JAX arrays are sent as numpy arrays (bfloat16 as float32), copied on the policy's thread | always on |
| One write | Each response leaves the server in one write | always on |
| Server time | `Server-Timing: act;dur=…, model;dur=…` on each response: the server's time and the policy's, so the robot does not count them as the link's | always on |
| Token | None, unless `$SEQ_SERVE_TOKEN` is set | `token="…"`: every request must carry `Authorization: Bearer …` |
| Once per act | Copies of one act, a robot's over each of its uplinks with the same `X-Seq-Request-Id`, run the policy once: each copy gets that answer, kept 30 s, and `X-Seq-Reads` says `once` | always on |
| Idle connections | Kept open 5 minutes | always on |
| Size | A request's body, and each compressed array or picture in it once restored, up to 64 MB | `limit=` |

## On the robot: `connect()`

| Part | What it does by default | To change it |
|---|---|---|
| Pictures as JPEG | Each picture (a numpy array of shape H×W×3 and dtype uint8, or an RGB PIL image as `frames.resize_image` returns it, at least 32×32, under any key) is sent as JPEG at quality 95 while the server says it reads JPEG; with nothing heard from the server yet, the first act asks its ping first, so it goes as JPEG too | `jpeg=False` |
| Compression | Large arrays are compressed where that saves upload time, while the server says it reads zstd or zlib | `compress=False` |
| The first wait | The first act waits up to 120 s for its response, the others 10 s | `first_timeout=10.0` |
| Pings when idle | A small GET on the same connection after each 100 ms of idle time, for up to a minute | `ping_path=None` |
| Pings out of an act's way | Once the robot's acts come at a steady spacing, a ping whose answer might still be on its way when the next act is due - judged by the slowest tenth of the last 32 pings' round trips - is not sent. On Linux without BBR, where a pause restarts the window, the one ping that alone keeps it (the connection's silence still under a round trip and 200 ms) is sent all the same; `stats()` counts the others as `pings_yielded` | `ping_yield=False` |
| One connection | Every request and ping on one kept connection, never two at once | always on |
| BBR and TCP keepalive | Requested on the connection's socket, where the machine allows them | always on |
| Machine checks | One warning when Wi-Fi power saving is on (on macOS, which gives nothing to read it from, when the requests go over Wi-Fi), and one when the machine restarts its window after a pause | `check=False` |
| Who answers | One warning when the server's answers do not say it is `serve()`: another program may answer at that address | `check=False` |
| Token | None | `token="…"`: sent as `Authorization: Bearer …` to a server started with one |
| Two uplinks | Off: one connection | `lines=["eth0", "wwan0"]`: each act sent over each interface's own connection, the first answer used, to a server that says `once` ([how](how-do-i-send-each-act-over-two-uplinks.md)) |
| Follow | Off | `follow=token` (or `$SEQUENCES_FOLLOW_TOKEN`): on a GPU rented from General Sequences, the robot's own address is let into its allow list before it connects; with `lines=`, each line's address, its follow going out by that line |

`sequence_ai.connect(url, ...)` opens this connection, with every default, and takes `token=`, `follow=`, `lines=`,
`jpeg=`, `compress=`, `first_timeout=` and `timeout_s=`; the other switches (`ping_path=`, `check=`) are those of
`sequence_ai.network.connect(url, ...)`, which opens the same connection.
`jpeg=True` and `compress=True` send pictures as JPEG and compress arrays whatever the server says, for a server of your
own that restores them without saying so.

## Check what is on

```bash
python -m sequence_ai.check http://YOUR_GPU:8000
```

Against the [Quickstart](quickstart.md)'s server on a Mac, it printed:

```text
sequence-ai check: http://127.0.0.1:8000

  ok  server              sequence-ai 0.63.0, pid 35969, up 1 s, policy policy_user; it answers /ping with 200 and restores jpeg, zlib, zstd
  ok  round trip          1.5 ms for a ping on a kept connection (median of 10); 1.0 ms for a TCP connect
  ok  send window         macOS keeps a connection's send window through pauses
  --  BBR                 macOS has no BBR, and needs none for the window
  --  Wi-Fi power saving  not checked: macOS gives nothing to read it from; this machine's requests go over Wi-Fi (en0), and a wired connection takes the question away
  ok  pings when idle     connect() pings after each 100.0 ms of idle time, and the server answers them
  ok  pictures as JPEG    the server restores them, so connect() sends each HxWx3 uint8 array as JPEG: 1 in this observation, 150,658 bytes as arrays, 1,518 with them as JPEG
  ok  compression         the server restores zlib, zstd, so connect() compresses large arrays with zstd where that saves time
  --  pings in inference  each act took 1.0 ms, too short to tell
  ok  server time         the server says it took 0.3 ms of a 1.0 ms act: 0.7 ms outside it, on the network and in this machine
  ok  first act           1.4 ms, the next ones 1.0 ms (observation from GET /example); connect() waits up to 120 s for the first
```

A `no` line says what is off and why. It takes `--token` for a server started with one, and `--observation FILE` for a
server that hands out no example. It exits 0 when no line says `no`.

## Why each part is on

- **JPEG.** At quality 95, JPEG moved a π0.5 policy's actions by about a tenth of what the policy's own sampling noise
  moves them, and two 224×224 cameras took 23.6 KB instead of 301 KB as arrays
  ([`frames.py`](../sequence_ai/network/frames.py) has the measurement). Only H×W×3 uint8 arrays are sent this way: a
  depth map or a mask is sent as it is. A policy that needs its pictures' exact values is served with `jpeg=False`.
- **Pings.** Linux restarts an idle connection's window, and the next upload then takes extra round trips:
  [Why does my robot's upload get slow after a pause?](why-does-my-robots-upload-get-slow-after-a-pause.md)
  A ping has a cost too: the connection carries one request at a time, so an act that starts while a ping is on its
  way waits for its answer, a round trip at most (`acts_behind_ping` and `ping_wait_us` in `stats()`). On a line with
  a 48 ms round trip to an L4 running π0.5, 27 of 88 acts did, 26.8 ms at the median. So a ping that would meet the
  robot's next act is left out: the next act is expected the median gap between the last acts after the last one, a
  ping's answer by the slowest tenth of the last 32 pings' round trips - on a line out cellular, whose round trips run
  52 ms at the median and 128 ms at the 90th percentile, the median let one ping in twenty go that met the act - and
  should it not come by a tenth of that gap late, the ping goes then. One ping is sent all the same, on Linux without
  BBR: the one that alone keeps the window, while the connection's silence is still under a round trip and 200 ms,
  Linux's least retransmission timeout. Once a long inference has spent that silence, the next send restarts the window
  whatever it is, and a ping would only meet the act; on macOS, or with BBR, no pause restarts it. With a 200 ms
  stand-in policy behind a 60 ms round trip and a robot asking every 533 ms (15 Hz, every 8 steps), 24 and 26 of 83
  acts waited behind a ping without it, and 1 and 1 with it (2 and 2 under Linux's rule); an act's p95 went from 321
  and 312 ms to 293 and 295 ms. With a 75 ms one behind a 48 ms round trip, 59 and 57 of 83 did without it, and 2 and
  2 with it.
- **The policy's own thread.** A forward pass takes tens of milliseconds, and on the server's event loop it would hold
  every ping and every upload for as long. With a 200 ms stand-in model and two robots, a ping's p50 was 153 ms with
  the model on the event loop and 1.0 ms with it on its own thread. It does not make a robot's request faster: with one
  policy, requests wait their turn either way (p50 316 ms against 320 ms in the same test). What serves several robots
  sooner is batching. CUDA and JAX keep state per thread, so the policy always runs on the same one.
- **Server time.** Without it, a slow model looks like a slow link to the robot, which then compresses more than it
  should.
- **The first wait.** A policy compiled on its first inference took 32 to 43 s, past a 10 s timeout.
- **One write.** Behind a tunnel, a response written in two pieces took a round trip more:
  [`one_write.py`](../sequence_ai/network/one_write.py).
- **Batching is off** because it changes what the policy is called with: a list instead of one observation. Turned on,
  it lets one forward pass serve several robots: one policy took 68.1 ms at batch 1 and 39.6 ms per sample at batch 16
  on one GPU ([`batcher.py`](../sequence_ai/engine/batcher.py)), and
  [How many robots can one GPU serve?](how-many-robots-can-one-gpu-serve.md) has what that gains in robots.

## See also

- [How do I serve a robot policy from my own GPU?](how-do-i-serve-a-robot-policy-from-my-own-gpu.md)
- [Quickstart](quickstart.md)
