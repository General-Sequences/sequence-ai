# How do I reduce the latency of LeRobot remote inference?

Start with the link, because LeRobot 0.7's remote inference already manages the robot's timing itself: put the
policy server near the robot, measure the link, and keep the link from going idle between observations. `sequence-ai`
has one small component for the last of these, a Zenoh setting, and it is the only thing here that touches LeRobot.

```python
from sequence_ai.network import lerobot

path = lerobot.keep_alive_file("zenoh-keep-alive.json5")     # an idle link then sends a frame every 100 ms
```

```bash
pip install sequence-ai          # it needs neither LeRobot nor Zenoh to write the file
```

Give the file to both sides, since each side sends its own frames:

| Side | LeRobot's setting | As a flag |
|---|---|---|
| The GPU: `lerobot-policy-server` | `zenoh.config_file` | `--zenoh.config_file=zenoh-keep-alive.json5` |
| The robot: `lerobot-rollout --inference.type=remote` | `zenoh_config_path` | `--inference.zenoh_config_path=zenoh-keep-alive.json5` |

The setting names are read from LeRobot 0.7.0's source (`ZenohConfig.config_file` in `transport/zenoh.py`,
`RemoteInferenceConfig.zenoh_config_path` in `inference/factory.py`). The flags follow from them the way LeRobot's
other flags do.

## Tested

- **Verified:** Zenoh 1.10.2 takes the setting. With the file, an idle link between two Zenoh peers on one machine sent
  a frame about every 100 ms each way; without it, a frame every 1.5 s or less often. `tests/network/test_lerobot.py`
  repeats this when Zenoh is installed.
- **From source inspection:** LeRobot loads `config_file` first and then sets only its own keys over it (the
  mode, the endpoints, discovery, shared memory and the connect timeout), so a transport setting written in the file
  stays.
- **Measured on a real link:** two Zenoh peers on two cloud machines, one answering the
  other's requests. A 100 KB request 0.53 s after each response took 38.1, 44.8, 82.8 and 164.3 ms at round-trip times
  (RTTs) of 11, 21, 41 and 81 ms, and 13.9, 23.2, 42.3 and 82.2 ms with the file on both ends: two to three round trips
  became one, on 40 of 40 requests. [What do the network components save?](measurements.md) has the table.

## Limitations

- The file has not been tested with LeRobot's own policy server and robot: the measurement used two Zenoh peers.
- On some links the idle restart costs nothing: see [the README's limitations](../README.md#limitations).

## Try it

[`examples/quickstart_lerobot.py`](../examples/quickstart_lerobot.py) writes the file and prints where to set it on each
side:

```bash
python examples/quickstart_lerobot.py
```

```text
wrote zenoh-keep-alive.json5:
{
  "transport": {
    "link": {
      "tx": {
        "keep_alive": 100
      }
    }
  }
}
on the GPU, the policy server's setting:   zenoh.config_file = zenoh-keep-alive.json5
on the robot, remote inference's setting:  zenoh_config_path = zenoh-keep-alive.json5
```

`keep_alive` is how many keep-alive frames a Zenoh link sends in one lease. The lease is 10 seconds unless you
changed it, so 100 frames per lease is one every 100 ms. The lease itself is unchanged: it is how long a silent peer
is waited for, and a short one drops a robot whose Wi-Fi stalls for a moment.

**You already have a Zenoh file**, for TLS or access control. LeRobot takes one file, so put both in it:

```python
mine = {"transport": {"link": {"tls": {"root_ca_certificate": "ca.pem"}}}}
path = lerobot.keep_alive_file("zenoh.json5", base=mine)     # your settings, with the keep-alive beside them
```

Zenoh 1.10.2 loaded that file with both settings in it.

## Why an idle link costs anything

On Linux, a TCP connection that has sent nothing for longer than its retransmission timeout, which is at least its
RTT plus 200 ms, has its congestion window restarted. The next observation can then take more round trips to upload
than it would on a warm connection. A link that sends a small frame every 100 ms is never idle that long.
[Why does my robot's upload get slow after a pause?](why-does-my-robots-upload-get-slow-after-a-pause.md) has the
measurements, including the links where the pause cost nothing.

## The other things that set the latency

**The distance.** The RTT between robot and GPU is incurred on every request, whatever else is done.
[Measure it](how-do-i-measure-the-link-between-my-robot-and-my-gpu.md), and if the GPU is far,
[get one nearer](how-do-i-get-a-gpu-near-my-robot.md).

**LeRobot's own settings.** These are LeRobot's, from the `RemoteInferenceConfig` of 0.7.0, and were not measured
here:

| Setting | Default | What it is |
|---|---|---|
| `encoding` | `"raw"` | `"jpeg"` sends camera frames as JPEG, at `jpeg_quality` (90), which makes each upload smaller |
| `refill_seconds` | `0.5` | Request the next chunk when this much playback is left; never less than the recent longest turnaround plus one action interval |
| `chunk_merge` | `"auto"` | How a new chunk is merged with the one being executed: `"append"` or `"aligned"` |
| `max_observation_age_s` | `5.0` | The oldest observation that is still acted on |

**The robot's machine.**

- `window.keep()` sets BBR on one socket. It cannot reach LeRobot's sockets, because Zenoh opens its own.
  The machine-wide settings do reach them, and both need root:
  `sudo sysctl -w net.ipv4.tcp_congestion_control=bbr` (after `sudo modprobe tcp_bbr`), or
  `sudo sysctl -w net.ipv4.tcp_slow_start_after_idle=0`.
- On Wi-Fi, check power saving. This reads the machine's settings and changes nothing:

  ```python
  from sequence_ai.network import wifi_power

  wifi_power.check()    # the interface whose power saving is on, having warned how to turn it off; else None
  ```

## What this package does not add to LeRobot

LeRobot's remote inference has its own transport, message format and client, so `connect()`, `msgpack_io`,
`compress` and `one_write` do not apply to it. LeRobot already requests the next chunk ahead of time based on
the turnaround it measured, aligns a new chunk with the one being executed, and can send frames as JPEG.
[The comparison page](comparison.md) describes what each does, from source.
