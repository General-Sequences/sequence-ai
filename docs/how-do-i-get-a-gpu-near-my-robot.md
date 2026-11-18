# How do I get a GPU near my robot?

Use any GPU the robot can reach: one you own, one from a cloud provider, or one rented near the robot. Whichever it is,
the robot's code changes in one line, the address it connects to, and the components in this repository are used on
that GPU in the same way.

```python
import sequence_ai

link_user = sequence_ai.connect("http://ADDRESS:8000")       # the GPU's address
```

GPUs are in short supply, and few are near any given robot. [General Sequences](https://generalsequences.com) rents
them by location, with this package's `seq gpu` commands: you name a city, a GPU is started near it, and you have root
in a container on it.

```bash
seq gpu places san fr                                           # the US cities a place can be
seq gpu add --place "San Francisco, CA" --gpu L4 --port 8000    # one GPU near it, and the port your server listens on
seq gpu ssh <id>                                                # root in its container: start your policy server
seq gpu remove <id>                                             # ends it, and its disk with it
```

The port is open to the address `seq gpu add` was run from; `seq gpu allow <id> <the robot's address>` opens it to the
robot, an IPv6 range as well as an IPv4 one. A robot whose address changes, on a cellular or satellite line, gets a
follow token, `seq gpu follow-token <id> --robot <name>`, and connects with `sequence_ai.connect(url, follow=token)`
(or `$SEQUENCES_FOLLOW_TOKEN` set): its address of the moment is let in before it connects, and with two uplinks
(`lines=["eth0", "wwan0"]`) each line's, by a follow that goes out by that line. Over IPv4, a cellular or satellite
line's public address is usually shared by many of its provider's customers (carrier-grade NAT), and letting that
address in lets them all in: where `seq gpu list` shows the GPU an IPv6 address, connect to that address or to the
GPU's name, so that the robot follows over IPv6, and start the server with a token either way.

This is a paid service, billed while the GPU runs. Everything else on this page, and every component, works the same on
any other GPU. [Its documentation](https://generalsequences.com/docs/gpus/) has the rest.

## Why near

A robot that requests its next actions from a remote GPU waits for the model and for the network, and the network's
share grows with distance. The inference layer (`sequence_ai.engine`) reduces the model's time, the network layer
(`sequence_ai.network`) removes round trips the connection does not need, and the distance itself depends on where the
GPU is.

## The same line, for each client

| Your robot uses | The line that changes |
|---|---|
| `sequence_ai.network.connect()` | `connect("http://ADDRESS:8000")` |
| The quickstart's robot | `python quickstart_robot.py http://ADDRESS:8000` |
| openpi's client | `WebsocketClientPolicy(host="ADDRESS", port=8000)` |
| LeRobot's remote inference | its `endpoint`, `tcp/ADDRESS:PORT` |

The inference components need no change on a new GPU:

```bash
pip install sequence-ai
python examples/check_on_a_gpu.py        # the capture, checked against eager mode on that GPU
```

## Before the robot connects

- **The server listens on every address.** `127.0.0.1` is reachable from the GPU itself only. `serve()` listens on
  every address unless it is given `host`: `python quickstart_server.py` does.
- **The port is open to the robot.** A rented GPU's firewall usually admits no one until you open a port. Open the
  server's port to the address or range the robot connects from, or follow the robot's address with `follow=`. If you
  open it to everyone, start the server with a token: `serve(policy, token="...")`, and `connect(url, token="...")` on
  the robot.

## Then measure

A GPU nearer the robot shortens the RTT, and the network components remove round trips that are not needed at all.
Which of them help on the new link is something to measure:
[How do I measure the link between my robot and my GPU?](how-do-i-measure-the-link-between-my-robot-and-my-gpu.md)
The measuring tool's server listens on port 9000 by default, so open that port too.

## Tested

The client side was tested against a server at another address of the same machine:
`python quickstart_robot.py http://ADDRESS:8017` printed its three requests.

## See also

- [How do I serve a robot policy from my own GPU?](how-do-i-serve-a-robot-policy-from-my-own-gpu.md)
- [How do I run π0 (openpi) on a remote GPU?](how-do-i-run-pi0-openpi-on-a-remote-gpu.md)
- [How many robots can one GPU serve?](how-many-robots-can-one-gpu-serve.md)
