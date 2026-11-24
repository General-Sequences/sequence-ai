# How many robots can one GPU serve?

As many as it can answer in time. A policy that responds with a chunk of actions leaves its robot some steps still to
execute when it requests the next chunk: that buffer, less the robot's round-trip time (RTT) and a margin, is how long a
batch may take. `slack` computes, on a clock, how many robots a replica of the model answers in time from each robot's
buffer and the model's batch times, and names the robots it cannot serve; `batcher` then sends their requests as
batches.

```python
from sequence_ai.engine import slack

table_user = slack.measure(forward_of_user, sizes=[1, 2, 4])  # forward_of_user(n): one forward pass of n requests
play_user = slack.playback(hz=20, valid_len=15, replan_at=8)  # 15 valid steps, the next requested at the 8th
seating_user = slack.Seating(table_user, play_user["buffer_s"] * 1000, sizes=[1, 2, 4],
                             cycle_ms=(play_user["play_s"] - play_user["buffer_s"]) * 1000)
seats_user, over_user = seating_user.seat([("near-1", 30.0), ("near-2", 30.0), ("near-3", 30.0), ("near-4", 30.0),
                                           ("far", 60.0)])                        # (robot, its RTT in ms)
```

With batches of 1, 2 and 4 that take 80, 150 and 250 ms, this gives `(4, ["far"])`: the replica has capacity for
four robots, and the one 60 ms away is the one to move to another replica. `python examples/components/slack.py`
shows the same rules on a case of its own, with no model and no GPU.

## The limits

- **Latency.** A robot fits the largest listed batch size whose batch time, plus the robot's own RTT and 50 ms, is
  within the buffer. A robot further away has less of its buffer left for the server, so it fits a smaller batch.
- **The batches a request arrives behind.** A replica runs one batch at a time, and robots send on their own clocks:
  a request that arrives as another batch runs waits for it, and is in time or not depending on how far apart the
  robots' requests fall. No formula has that, so `slack` runs 2, 3 and more robots on a simulated clock, as its
  server loop would serve them, from starts spread evenly over every way their requests can fall in a cycle (64 for
  two robots, 128 for more). A number of robots is within capacity when no response is late in any start: each robot's
  loop starts again from the moment it is called, so a fleet can spend a whole episode in any of the ways its requests
  can fall. `worst=False` asks instead for 95% of the responses back before their robot's buffer runs out, each start
  counting the same.
- **Load.** In each of those starts the robots' batches may fill at most 0.9 of a cycle, the time between one robot's
  requests. Past that, responses queue behind each other however the requests are batched.

The first number of robots that fails ends the count: robots are admitted one at a time, so a replica reaches a
number of robots through every smaller one. In the example above a batch of four fits the buffer of a robot 30 ms
away (250 + 30 + 50 ms, within 350), and the capacity is four: on the clock the four near robots have every response
in time, and their batches fill no more than 0.78 of a cycle.

Each robot is evaluated by its own RTT. When every robot was evaluated by the furthest robot's RTT, one far robot
lowered the batch size, and with it the capacity, of every near robot on the same replica.

## Using it

The policy says its buffer: `@seq.policy(hz=15, action_chunk_valid_len=8, action_chunk_replan_at=5)` is 15 steps a
second, the first 8 of each chunk played, the next asked for at the 5th. With `seq serve`, batching is one decorator:
`@seq.concurrent(sizes=[1, 2, 4, 8])` on the `@seq.infer` method hands it the observations waiting, stacked, through the
`Batcher` below, and each robot gets its own row back ([How do I serve a robot policy from my own
GPU?](how-do-i-serve-a-robot-policy-from-my-own-gpu.md#your-model-in-a-class)). In a server of your own:

```python
from sequence_ai.engine.batcher import Batcher

# model_on_a_list_user(inputs: list) -> list
batcher_user = Batcher(model_on_a_list_user, sizes=[1, 2, 4, 8], wait_ms=2.0)
output_user = await batcher_user(observation_user)  # in each request's handler
```

- A batch is dispatched as soon as it is full, or `wait_ms` after its first call arrived while the model is free.
- While the model is busy, waiting calls keep accumulating and are dispatched together as soon as it is free.
- Only the sizes you list are sent: with sizes `[1, 2, 4]`, three waiting calls are dispatched as a batch of 2, then
  a batch of 1.
- A batch in which the model raised is re-run call by call, so only the call that fails by itself fails.

`Batcher` waits one fixed time for every caller and has no notion of a robot. A server that holds a batch by each
robot's slack asks `slack.Aligner` how long the open batch may wait and which requests are dispatched first:
[`docs/slack.md`](slack.md) has the rules and a loop.

## Measured

- On an A100-40GB, GR00T N1.6 with CUDA graphs answered 24 robots in time where answering one at a time answered 9:
  each robot 20 ms away, on its own clock, under 5% of responses late.
- π0.5 DROID on an A100-40GB answered 12 robots in time, where one at a time answered 6.
- Seating only the robots it can answer in time kept every seated robot on time; a server that seats every robot that
  asks had none on time (π0.5's batch times on an A100, 16 robots asking).

One replica per GPU: one copy of the model answers all its robots, their requests batched.

## When it applies

- A policy whose robots wait for each response leaves no slack, so its replica serves one robot.
- Each robot reports its RTT; without one, a fixed RTT is assumed.
