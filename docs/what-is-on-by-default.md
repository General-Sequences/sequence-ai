# What is on by default, and how do I turn each part off?

Everything on this page is on as soon as the GPU runs `seq serve` and the robot runs `sequence_ai.connect(url)`:
nothing needs to be added. The last column of each table is how to change a part.

```bash
seq serve                                                  # on the GPU, in the policy's folder
```

```python
policy_user = sequence_ai.connect("http://YOUR_GPU:8000")  # on the robot
```

## On the GPU: `seq serve`

| Part | What it does by default | To change it |
|---|---|---|
| The policy's thread | The policy runs on one thread of its own, one call at a time. Acts that arrive meanwhile wait their turn, in order; pings and uploads are not held behind it | always on |
| Batching | Off: one observation per call | `@seq.concurrent(sizes=[1, 2, 4, 8])` on `@seq.infer`: it takes the observations waiting, stacked, and returns a row for each |
| Seats | With `hz` on `@seq.policy`: a robot is let in at its first act only where every robot let in is still answered in time; another robot's act is refused `503`, saying why | `@seq.policy` without `hz`: off |
| Addresses | Every address of the machine: `0.0.0.0`, and `::` where it has IPv6 | `seq serve --host 127.0.0.1`: this machine only |
| The robot's stream | Each robot's observations and acts on one WebSocket (`GET /stream`), each message answered as a request; `X-Seq-Act-Formats` names it on every response | always on |
| Observations | Kept by episode, in order; a robot's `Video`, `Depth` and `Tactile` items turned back into the arrays they were. One that did not come is asked for again, and the act waits for it | always on |
| Ping | `GET /ping` is answered at once: `503` with `Retry-After` until the warm-up is done, then `200` with what the server is, as JSON (`server`, `version`, `up_s`, `pid`, `policy`, `optimized`) | always on |
| Identity | `X-Seq-Server: sequence-ai/<version>` on every response, so a robot can tell `seq serve` from another program at its address | always on |
| Warm-up | With `@seq.sample`, the policy runs on its first observation twice at each batch size, four times when it declares `hz`, before `/ping` says ready; `GET /example` hands it out | leave out `@seq.sample` |
| Inputs | The observation as the robot made it, numpy arrays as arrays, each the policy's own to write into | always on |
| Outputs | torch tensors and JAX arrays are sent as numpy arrays (bfloat16 as float32), copied on the policy's thread | always on |
| One write | Each response leaves the server in one write | always on |
| Server time | Each act's answer says how long the server took (`perf`: `handler_ms` from the act's last byte to its answer, `inference_ms` the policy call's), so the robot does not count it as the link's | always on |
| Recording | The last minute of the robots' steps kept in memory, for a recorder to read as they come (`GET /record`, `python -m sequence_ai.network.record`) | always on |
| Token | None, unless `$SEQ_SERVE_TOKEN` is set | `seq serve --token …`, or `SEQ_SERVE_TOKEN=…`: every request and every robot's stream must carry `Authorization: Bearer …` |
| Idle connections | Kept open 5 minutes | always on |
| Size | A request's body, or a message of a robot's stream, up to 64 MB | `limit=` |

## On the robot: `connect()`

| Part | What it does by default | To change it |
|---|---|---|
| The stream | Every observation and act on one WebSocket to the server, kept open: each observation sent as it is made, each act naming how many came before it | always on |
| Video | A `sequence_ai.Video` item goes as low-latency video (H.264), a keyframe every second and after anything lost | always on |
| Pictures | A picture you did not mark (a numpy array H×W×3 of uint8, each side 32 pixels or more, under any key) goes as a video keyframe of its own, lossy, decoded alone | `pictures="exact"`: every byte as it is |
| Depth and touch | A `sequence_ai.Depth` item goes losslessly; a `sequence_ai.Tactile` image as high-quality video | always on |
| Compression | Large arrays are compressed where that saves upload time | always on |
| Lost and refused | An observation the server lost is sent again; one it refused is left out, and the video's next frame is a keyframe | always on |
| Pings when idle | A ping on the stream after each 100 ms of idle time, from the first observation until 60 s after the last observation or act | always on |
| BBR and TCP keepalive | Requested on the stream's socket, where the machine allows them | always on |
| Machine checks | On Linux, one warning when Wi-Fi power saving is on, and one when the machine restarts its window after a pause | always on |
| Token | None | `token="…"`: sent as `Authorization: Bearer …` to a server started with one |
| Two uplinks | Off: one stream, by the default route | `lines=["eth0", "wwan0"]`: a stream by each interface, every observation and act on every line, the first answer used ([how](how-do-i-send-each-act-over-two-uplinks.md)) |

`sequence_ai.connect(url, ...)` opens the stream with every default, and takes `token=`, `lines=`, `pictures=` and
`timeout_s=`.

## Check what is on

```bash
python -m sequence_ai.check http://YOUR_GPU:8000
```

Against the [Quickstart](quickstart.md)'s server on a Mac, it printed:

```text
sequence-ai check: http://127.0.0.1:8000

  ok  server              sequence-ai 0.66.0, pid 68489, up 9 s, policy QuickstartPolicy_user.infer_user, no speed-up on; it answers /ping with 200 and takes the robot's stream (x-seq-stream/3)
  ok  round trip          10 pings answered on a kept connection, and a TCP connect
  ok  send window         macOS keeps a connection's send window through pauses
  --  BBR                 macOS has no BBR, and needs none for the window
  --  Wi-Fi power saving  not checked: macOS gives nothing to read it from; this machine's requests go over Wi-Fi (en0), and a wired connection takes the question away
  ok  pings when idle     connect() pings after each 100.0 ms of idle time, and the server answers /ping with 200: the pings go on the robot's stream, from its first observation until 60 s after its last observation or act, and keep the connection, and the Wi-Fi radio, from going idle
  ok  pings in inference  answered while the model ran (3 pings in 3 acts)
  ok  server time         the server says it took 0.3 ms of a 1.7 ms act; the rest was outside it, on the network and in this machine
  ok  first act           108.3 ms, the next ones 1.7 ms (observation from GET /example)
```

A `no` line says what is off and why. It takes `--token` for a server started with one, and `--observation FILE` for a
server that hands out no example. It exits 0 when no line says `no`.

## Why each part is on

- **Video, and a keyframe for a picture.** A camera's frames sent as video carry what changed: on our test scene, a
  640×480 camera with a textured background and three moving discs, about 2.5 KB a frame against 921,600 bytes as an
  array. A picture you did not mark goes as a keyframe of its own, about 24 KB a frame there, decoded with nothing
  before it, so a policy asked once a frame, whatever the gaps between frames, reads each as it was sent. Only H×W×3
  uint8 arrays are sent this way: a depth map or a mask goes as it is. A policy that needs its pictures' exact values
  is reached with `pictures="exact"`.
- **Pings.** Linux restarts an idle connection's window, and the next upload then takes extra round trips:
  [Why does my robot's upload get slow after a pause?](why-does-my-robots-upload-get-slow-after-a-pause.md)
  A ping on the stream is a WebSocket control frame: the server's WebSocket answers it on its own, and an act never
  waits for its answer.
- **The policy's own thread.** A forward pass takes tens of milliseconds, and on the server's event loop it would hold
  every ping and every upload for as long. With a 200 ms stand-in model and two robots, a ping waited for the forward
  pass under way with the model on the event loop, and its p50 was 1.0 ms with the model on its own thread. It does not
  make a robot's act faster: with one policy, acts wait their turn either way (p50 316 ms against 320 ms in the same
  test). What serves several robots sooner is batching. CUDA and JAX keep state per thread, so the policy always runs on
  the same one.
- **Server time.** Without it, a slow model looks like a slow link to the robot.
- **One write.** Behind a tunnel, a response written in two pieces took a round trip more:
  [`one_write.py`](../sequence_ai/network/one_write.py).
- **Batching is off** because it changes what the policy is called with: a list instead of one observation. Turned on,
  it lets one forward pass serve several robots: one policy took 68.1 ms at batch 1 and 39.6 ms per sample at batch 16
  on one GPU ([`batcher.py`](../sequence_ai/engine/batcher.py)), and
  [How many robots can one GPU serve?](how-many-robots-can-one-gpu-serve.md) has what that gains in robots.

## See also

- [How do I serve a robot policy from my own GPU?](how-do-i-serve-a-robot-policy-from-my-own-gpu.md)
- [Quickstart](quickstart.md)
