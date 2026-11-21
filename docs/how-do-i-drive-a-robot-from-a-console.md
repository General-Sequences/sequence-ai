# How do I drive a robot from a console, and record what I do?

Run `seq teleop serve` on a machine the robot and the person can both reach, often the robot's own computer. The robot
connects to it with `sequence_ai.connect(address)`, and a person opens the page it serves, or drives from Python with
`sequence_ai.teleop(address)`. The server only pairs the two: pictures and commands then go directly between the browser
and the robot over WebRTC. With `--record`, the robot writes every step to a file, and `seq data export` turns the files
into a LeRobot v3.0 dataset.

```bash
pip install 'sequence-ai[teleop]'          # on the robot, and on a Python console (Python 3.11 or later)
seq teleop serve --robot arm_1_user --record
```

```
teleop for robot 'arm_1_user' at http://192.168.1.20:8200
  the robot:    sequence_ai.connect("http://192.168.1.20:8200", token=TOKEN)
  a console:    sequence_ai.teleop("http://192.168.1.20:8200", token=TOKEN), or the page at http://192.168.1.20:8200/
  its Record is on: the robot writes its steps to a file on its own machine ($SEQUENCES_RECORD_DIR, else the folder it runs in)
  TOKEN, made for this run (--token or $SEQ_TELEOP_TOKEN keeps one): ...
  the page with it: http://192.168.1.20:8200/#token=...
```

## The robot

The robot's handle has the same `observe()` and `act()` as a policy's. `act()` returns the newest chunk the person sent,
once; while no one drives the robot, or after a stop, it returns `hold`: `{}`, unless `connect(..., hold=...)` gives
another.

```python
import sequence_ai

with sequence_ai.connect("http://192.168.1.20:8200", token=TOKEN_USER) as robot_user:
    robot_user.wait_until_ready()
    while running_user:
        robot_user.observe({"wrist_user": sequence_ai.Video(frame_user), "joints_user": joints_user,
                            "task": "pick up the cube"})
        chunk_user = robot_user.act()          # the person's newest chunk, once; `hold` otherwise
        play_user(chunk_user)
    robot_user.label(True, "went well")       # how the episode went, kept with its steps
```

`robot_user.driver` is who drives now: `"model"` (no one), `"human"` or `"stopped"`.

## The person

The page at the server's address shows each camera, who drives, Take, Release and Stop, and how late the picture is
(Frame: from the robot's `observe()` to the screen, p50 / p95; Round trip: browser to robot and back). Esc is Stop. One
person drives at a time; a robot whose driver goes silent for 3 seconds stops.

From Python:

```python
import sequence_ai

with sequence_ai.teleop("http://192.168.1.20:8200", token=TOKEN_USER, operator="alice_user") as op_user:
    op_user.take()
    for step_user in op_user.stream():                # the robot's newest view, as it observed it
        op_user.send({"actions": plan_user(step_user["observation"])})
    op_user.release()                                 # op_user.stop() stops it where it is
```

`op_user.metrics()` has each command's one-way time and each step's time from capture to the console, p50, p95 and p99.

## From a headset, a gamepad or the keys

Where the browser offers WebXR's `immersive-vr`, the page shows Enter VR while you drive. In the headset, the robot's
cameras are on screens in a row in front of you, where you looked as you went in. Two cameras named alike but for `left` and `right`, as `head_left` and
`head_right`, are a stereo pair: one screen, each eye its own camera. The head, each hand or controller and, with hand
tracking, each hand's 25 joints go to the robot every frame the headset draws. Out of a headset, tick Gamepad and keys,
and a gamepad's sticks and buttons and the keys you hold go the same way. WebXR needs the page over `https://` or at
`localhost`: on another device, put the server behind TLS (see Good to know).

The console sends what it reads, as it reads it, under `inputs`. The robot turns that into its joints with
`sequence_ai.retarget`, since only the robot knows its arm and its limits:

```python
import sequence_ai
from sequence_ai import retarget

chain_user = retarget.Chain.from_urdf("arm_user.urdf", "base_link", "gripper_link")
box_user = ([0.1, -0.4, 0.02], [0.6, 0.4, 0.5])                   # where the tip may go, in the base, metres
headset_user = retarget.Arm(chain_user, workspace=box_user, max_speed=0.5)
pad_user = retarget.Twist(chain_user, workspace=box_user)

with sequence_ai.connect("http://192.168.1.20:8200", token=TOKEN_USER) as robot_user:
    robot_user.wait_until_ready()
    while running_user:
        robot_user.observe({"wrist_user": sequence_ai.Video(frame_user), "joints_user": q_user})
        chunk_user = robot_user.act()
        inputs_user = chunk_user.get("inputs") if isinstance(chunk_user, dict) else None
        if inputs_user:
            by_user = pad_user if "pad" in inputs_user or "keys" in inputs_user else headset_user
            step_user = by_user.step(inputs_user, q_user, dt=1 / 30)
            write_joints_user(step_user.q, step_user.gripper)   # gripper: 0 open .. 1 closed, None: leave it
            robot_user.teleop_written(executed=step_user.q)
```

- **Hold to drive.** The arm follows a controller only while its grip is squeezed, or a tracked hand only while its
  thumb touches its middle finger. Let go and the arm stays where it is; take hold again and it follows from where your
  hand is then, without a jump. A controller put down for a hand does not jump the arm either.
- **A hand the headset loses** holds the arm, and the arm follows again from where the hand comes back.
- **Limits.** The tip stays inside `workspace` and moves no faster than `max_speed` (m/s), and each joint no faster than
  `max_joint_speed`. For an arm whose joints cannot turn its tip every way, as an arm of fewer than six cannot, pass
  `orientation=False` and the tip follows your hand's position alone.
- **The gripper** follows a controller's trigger, or a hand's thumb-to-index pinch.
- **A gamepad and the keys** drive while you hold the left shoulder button or Shift: the left stick or W/S and A/D
  move the tip forward and sideways, the right stick or R/F up and down, and Q/E turn it. C closes the gripper and V opens
  it, or the right trigger closes it as far as you press. Let go of the shoulder button or Shift and the arm stays. In a
  headset, only the headset drives.
- **A dexterous hand**: `retarget.Hand(config_path)` gives its joints from your hand's, with dex-retargeting
  (`pip install dex-retargeting torch six`, a CPU torch will do) and its config for that hand.

With `--record`, the robot keeps the person's inputs, as they came, beside each step it played, and the joints it said
it wrote (`teleop_written(executed=...)`). `seq data export` puts those joints in `action` and the person's own motion
in `human_pose`: the head's position and quaternion, then each hand's or controller's (tracked or not, position,
quaternion, trigger, squeeze), 27 numbers a frame. `--format jsonl` keeps everything as it came, a hand's 25 joints with
it, so another retargeting can make the actions again from the same motion.

Each of these was checked with a headset emulated in Chrome (a script playing its head, a controller and a tracked
hand), a gamepad and the keys, driving a simulated arm.

## A dataset from what you recorded

```bash
seq data export ./recordings_user --out ./dataset_user            # LeRobot v3.0; --format jsonl keeps every item
```

A frame for every step the robot observed: each camera an image column (`observation.images.<key>`), every other number
an array (`observation.<key>`), the action the person's chunk gave that step, `driver` (0 model, 1 human, 2 stopped),
each episode's instruction as its task and its label as `next.success` on its last frame. `--fps` sets the frame rate;
by default it is the pace the robot observed at. The LeRobot format needs pyarrow: `pip install 'sequence-ai[data]'`.

## On General Sequences, one line changes

The same robot script, paired through General Sequences instead, with the robot's own key and `sequence-cloud`:

```python
robot_user = sequence_ai.connect("http://192.168.1.20:8200", token=TOKEN_USER)   # your own server
robot_user = sequence_ai.connect("teleop", api_key=ROBOT_KEY_USER)               # paired by General Sequences
```

and on the console, `sequence_ai.teleop(key=ROBOT_KEY_USER)` in place of the address. The requests, the page and the
recording's format are the same.

## When the two cannot reach each other directly

Paired by General Sequences, a console and its robot connect directly, with no relay in between: pictures and commands
take the shortest way. Where a firewall or a NAT keeps the two apart, the console says so within seconds, with
`Unavailable` (`details.reason` `"direct_failed"`) and what kept them apart.

A server of your own can take a relay of your own. Where both sides are behind a NAT that gives each peer a port of its
own (symmetric NAT), or a network lets no UDP out, name a TURN server, and the two meet through it. Each side is handed
a username and password for that server, good for `--turn-ttl` (12 hours by default), made with its shared secret -
coturn's `use-auth-secret`:

```bash
turnserver --use-auth-secret --static-auth-secret="$TURN_SECRET_USER" --realm=robots_user --listening-port=3478 \
  --denied-peer-ip=10.0.0.0-10.255.255.255 --denied-peer-ip=172.16.0.0-172.31.255.255 \
  --denied-peer-ip=192.168.0.0-192.168.255.255
seq teleop serve --robot arm_1_user --turn turn:turn.example_user:3478 --turn-secret "$TURN_SECRET_USER"
```

Refuse private addresses as peers, as above: a relay on the internet reaches no one's local address, and a TURN server
whose send to one fails ends that allocation (coturn does), the connection with it. Both sides try every direct way as
well, and `sequence_ai.teleop()` tries the relay's way only once no direct way is left that may answer, or a second on:
a connection goes through the relay where no direct way answers (on the page, the browser's own rules choose).
`op_user.path`, and the page's header, say which way the connection went: `"direct"` or `"relay"`; through a
relay, `op_user.why_relay` (and `metrics()["why_relay"]`) and the page say why not straight, from each side's NAT - both
behind symmetric NAT, a network that lets no UDP out, or a firewall between them. A console that never takes a relay,
`sequence_ai.teleop(..., direct_only=True)` or the page's Direct only, fails with `Unavailable` (`details.reason`
`"direct_failed"`) and says why, and its robot relays nothing either. Without `--turn` there is no relay, as on General
Sequences, and that is how every console fails where the two cannot meet.

## Many watching one robot

Each console connected straight to the robot is one more upload from it. For people who only watch, serve a Media over
QUIC relay: the robot uploads its cameras and steps once, to the relay, and the relay sends each watcher a copy.

```bash
pip install 'sequence-ai[moq]'          # on this machine, the robot and each Python watcher (Python 3.10 or later)
seq teleop serve --robot arm_1_user --moq-port 4443
```

```python
import sequence_ai

with sequence_ai.teleop("http://192.168.1.20:8200", token=TOKEN_USER, via="moq") as watch_user:
    for step_user in watch_user.stream():             # the robot's newest view, as a driving console reads it
        show_user(step_user["observation"])
```

On the page, tick Watch only: it plays each camera through the relay, in a browser with WebTransport (Chrome, Edge,
Firefox). A watcher reads `stream()`, `states()` and `metrics()` as a driving console does, and does not drive or stop the
robot: one person drives, over WebRTC, straight to the robot, and anyone who has to stop it connects straight too. The robot encodes a camera only while someone watches it through the relay, and a
watcher who comes in starts at a new keyframe; the first view it gets is the last one the relay held, with its own
`t_us`. The relay takes a session only with the token its server hands out with its address, and every client checks
the relay's certificate against the fingerprint handed out with it, so only UDP port 4443 has to be open to the
watchers. `--moq https://relay.example_user:4443/` names a relay of your own instead (moq-relay 0.17; `http://` for one
that made its own certificate). Watchers see the cameras and the arrays, not Depth or Tactile items.

## The encoder and the transport

Each camera goes to a console as H.264. The first time a camera is sent, the robot encodes a few frames with libx264
and with each hardware encoder this machine's PyAV opens, and sends with the fastest, or with a hardware one nearly as
fast, which spares the robot's CPU. `robot_user.teleop_metrics()["encoder"]` names the one it chose and what each one
took. `SEQ_TELEOP_ENCODER` picks one instead: `libx264`, `h264_videotoolbox`, `h264_nvenc`, `h264_qsv`, `h264_amf`,
`h264_mf` or `h264_v4l2m2m`. When the one it names does not open on this machine, the robot measures them all as
before.

The robot's side of each connection runs on aiortc. `SEQ_TELEOP_TRANSPORT=gstreamer` runs it on GStreamer's webrtcbin
instead, with the H.264 encoder GStreamer finds on the machine, and x264 where it finds none. Consoles connect to it
unchanged. There `SEQ_TELEOP_ENCODER` names a GStreamer element: `nvv4l2h264enc`, `v4l2h264enc`, `mpph264enc`,
`vah264enc`, `vtenc_h264_hw`, `x264enc` or `openh264enc`. It needs GStreamer 1.20 or later with webrtcbin, libnice's
elements and PyGObject:

```bash
apt install python3-gi gir1.2-gst-plugins-bad-1.0 gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-nice   # Debian, Ubuntu
brew install gstreamer libnice-gstreamer pygobject3                                                                     # a Mac
```

In a virtualenv on a Mac, also `pip install PyGObject` and set `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib`.

## Good to know
- **One server per robot.** `--robot` names the robot a server pairs.
- **On a network you do not trust, put the server behind TLS** (a reverse proxy that sets `X-Forwarded-Proto: https`).
  Over plain `http://`, the token travels as it is.
- **Python 3.11 or later** on the robot and on a Python console, as aiortc 1.16 needs.
