# How do I send each act over two uplinks?

A robot with two ways out (a satellite terminal and a cellular modem, say) can send its stream over both and use
whichever answer comes back first. A slow moment on one line then costs nothing while the other is quick, and a line
that drops leaves everything to the other.

```python
import sequence_ai

policy_user = sequence_ai.connect("http://YOUR_GPU:8000", lines=["eth0", "wwan0"])  # a stream by each interface
policy_user.wait_until_ready()
policy_user.observe(observation_user)                        # sent on both lines
action_user = policy_user.act()                              # asked on both lines; the first answer
print(policy_user.stats()["lines"])                          # each line's acts, first answers and errors
```

`lines=` names two or more of the robot's network interfaces, one uplink each, for a policy server at an address, one
started with `seq serve`. `ip link` lists a Linux machine's interfaces, and `ifconfig` a Mac's; a name the machine does
not have is refused with the names it has, before anything is opened.

## What it does

- **Each observation and each act goes on every line.** Each line has a stream of its own, its connection bound to its
  interface before it connects, so it leaves by that line whatever the default route. An act waits at the server for
  the observations before it, and they are there by whichever line was quicker.
- **The first answer is used.** The others are read when they come, and `stats()["lines"]` says how much later. An act
  fails only when every line has failed.
- **The policy runs once on a hosted worker.** The copies of an act carry one request id, `X-Seq-Request-Id`, and a
  hosted worker runs the act once and gives each copy that answer; `seq serve` runs it for each copy. An observation that
  comes by both lines is kept once.
- **A line that stops answering collects no queue.** A line still carrying the act before is not given the next one.
- **A line that drops opens again by itself.** The other lines carry everything meanwhile; the robot's stream ends only
  when every line has.

Where the server has another address on each line (IPv6 on one, IPv4 alone on the other), name each line's:

```python
policy_user = sequence_ai.connect(url_user, lines={"eth0": "http://[2001:db8::7]:8000",    # IPv6 by the terminal
                                                  "wwan0": "http://203.0.113.7:8000"})     # IPv4 by the modem
```

A deployed policy is reached at the address the platform gives, so for one reached by its name, `lines=` is a list.

A cellular or satellite line's public IPv4 address is usually shared by many of its provider's customers
(carrier-grade NAT): a port opened to it is open to them all. Where the GPU has an IPv6 address, connect to that, and
start the server with a token either way (`SEQ_SERVE_TOKEN=... seq serve`).

## What it costs

Everything the robot sends goes on each line: two lines, twice the upload. Two 640 x 480 cameras sent as
`sequence_ai.Video` at 15 frames a second came to about 240 KB/s in our measurement, about 860 MB an hour on each line;
your cameras' scenes will give other numbers. On a line billed by the gigabyte, count it. The answers come back on each
line too, but an answer is a chunk of actions, small beside the pictures.

## Needs

Linux 5.7 or later, or macOS: no root, since either lets a program bind its own socket to an interface.
