# How do I get a GPU near my robot?

Use any GPU the robot can reach: one you own, or one in a datacenter near the robot. Whichever it is,
the robot's code changes in one line, the address it connects to, and the components in this repository are used on
that GPU in the same way.

```python
import sequence_ai

policy_user = sequence_ai.connect("http://ADDRESS:8000")     # the GPU's address
```

## Why near

A robot that requests its next actions from a remote GPU waits for the model and for the network, and the network's
share grows with distance. The inference layer (`sequence_ai.engine`) reduces the model's time, the network layer
(`sequence_ai.network`) removes round trips the connection does not need, and the distance itself depends on where the
GPU is.

## The same line, for each client

| Your robot uses | The line that changes |
|---|---|
| `sequence_ai.connect()` | `sequence_ai.connect("http://ADDRESS:8000")` |
| The quickstart's robot | `python quickstart_robot.py http://ADDRESS:8000` |
| openpi's client | `WebsocketClientPolicy(host="ADDRESS", port=8000)` |
| LeRobot's remote inference | its `endpoint`, `tcp/ADDRESS:PORT` |

The inference components need no change on a new GPU:

```bash
pip install sequence-ai
python examples/check_on_a_gpu.py        # the capture, checked against eager mode on that GPU
```

## Before the robot connects

- **The server listens on every address.** `127.0.0.1` is reachable from the GPU itself only. `seq serve` listens on
  every address unless it is given `--host`.
- **The port is open to the robot.** A GPU in a datacenter usually admits no one until you open a port. Open the
  server's port to the address or range the robot connects from. If you open it to everyone, start the server with a
  token, `SEQ_SERVE_TOKEN=... seq serve`, and `connect(url, token="...")` on the robot.

## Then measure

A GPU nearer the robot shortens the RTT, and the network components remove round trips that are not needed at all.
Which of them help on the new link is something to measure:
[How do I measure the link between my robot and my GPU?](how-do-i-measure-the-link-between-my-robot-and-my-gpu.md)
The measuring tool's server listens on port 9000 by default, so open that port too.

## Tested

The client side was tested against a server at another address of the same machine:
`python quickstart_robot.py http://ADDRESS:8017` printed its first answer and played 48 steps at 15 a second.

## See also

- [How do I serve a robot policy from my own GPU?](how-do-i-serve-a-robot-policy-from-my-own-gpu.md)
- [How do I run π0 (openpi) on a remote GPU?](how-do-i-run-pi0-openpi-on-a-remote-gpu.md)
- [How many robots can one GPU serve?](how-many-robots-can-one-gpu-serve.md)
