# How do I send each act over two uplinks?

A robot with two ways out - a Starlink terminal and a cellular modem, say - can send each act over both and use
whichever answer comes back first. A slow moment on one line then costs nothing while the other is quick, and a line
that drops leaves the acts to the other.

```python
import sequence_ai

robot_user = sequence_ai.connect("http://YOUR_GPU:8000", lines=["eth0", "wwan0"])   # one connection by each interface
robot_user.wait_until_ready()
action_user = robot_user.act(observation_user)               # sent on both lines; the first answer
print(robot_user.stats()["lines"])                           # each line's acts, first answers and errors
```

`lines=` names two or more of the robot's network interfaces, one uplink each, for a policy server at an address: one
started with `serve()` on your own GPU, or on a GPU you rent. `ip link` lists a Linux machine's interfaces, and
`ifconfig` a Mac's; a name the machine does not have is refused with the names it has.

## What it does

- **Each act goes on every line at once.** The observation is packed once, and the same body goes on each line's own
  connection, bound to its interface before it connects, so it leaves by that line whatever the default route.
- **The first answer is used.** The others are read when they come. An act fails only when every line has failed.
- **The policy runs once.** The copies of an act carry one request id, `X-Seq-Request-Id`; `serve()` runs the policy
  once for them and gives each copy that answer, and says so with `once` in its `X-Seq-Reads` header. To a server
  that does not say `once`, each act goes on one line alone, the first that is free, and on the next only when that
  one could not connect: an act is never run twice.
- **A line that stops answering collects no queue.** A line still carrying the act before is not given the next one.
- **Each line stays warm.** Each line's connection keeps its own pings when idle, as one connection does
  ([What is on by default](what-is-on-by-default.md)).

Where the server has another address by each line - IPv6 on one, IPv4 alone on the other - name each line's:

```python
robot_user = sequence_ai.connect(url_user, lines={"eth0": "http://[2001:db8::7]:8000",     # IPv6 by the terminal
                                                  "wwan0": "http://203.0.113.7:8000"})     # IPv4 by the modem
```

With an address by each line, each act first compares the servers the lines' pings named: where their process, version
or start time differ, the lines reach two servers, and the act raises `ValueError` and is sent on none.

## On a GPU rented from General Sequences

With a follow token, each line's address is let into the GPU's allow list before the robot connects, each by a follow
that goes out by that line:

```python
robot_user = sequence_ai.connect("http://YOUR_GPU:8000", follow=follow_token_user, lines=["eth0", "wwan0"])
```

`seq gpu follow --line wwan0` does the same for one line from a shell, and `seq gpu list` says where each line was let
in from. [How do I get a GPU near my robot?](how-do-i-get-a-gpu-near-my-robot.md) has the follow token.

A line follows over the address family it reaches the GPU by. Over IPv4, a cellular or satellite line's public address
is usually shared by many of its provider's customers (carrier-grade NAT), and letting that address in lets them all in.
Where `seq gpu list` shows the GPU an IPv6 address, connect to that address or to the GPU's name, so that each line
with IPv6 follows over it; and start the server with a token either way (`serve(policy, token=...)`).

## What it costs

Every act's body is sent on each line: two lines, twice the upload. Two cameras' pictures as JPEG are about 24 KB an
act ([`frames.py`](../sequence_ai/network/frames.py) has the measurement), so a policy asked every 533 ms, as π0.5 is,
sends about 160 MB an hour on each line. On a line billed by the gigabyte, count it. The answers come back on each line
too, but an answer is a chunk of actions, small beside the pictures.

## Check both lines

```bash
python -m sequence_ai.check http://YOUR_GPU:8000 --lines eth0,wwan0
```

Beside its other lines, it pings the server by each interface and says whether the server runs an act's copies once.
With an observation (the server's example, or `--observation FILE`), it sends 5 acts on every line and says how many
each line answered first, and how many megabytes an hour each line sends at one act a second.

## Needs

Linux 5.7 or later, or macOS: no root, since either lets a program bind its own socket to an interface. A policy
reached by its deployment's name is reached over one line.
