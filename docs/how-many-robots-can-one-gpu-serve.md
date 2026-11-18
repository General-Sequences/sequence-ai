# How many robots can one GPU serve?

As many as can be answered within the buffer their action chunks leave. A policy that responds with a chunk of
actions leaves its robot some steps still to execute when it requests the next chunk: that buffer, less the robot's
round-trip time (RTT) and a margin, is how long a batch may take. `slack.Seating` turns a timing table and three
policy parameters into a capacity, and names the robots a replica of the model cannot serve. `batcher` then sends
their requests as batches.

```python
from sequence_ai.engine import slack

table_user = slack.measure(forward_of_user, sizes=[1, 2, 4])  # forward_of_user(n): one forward pass of n requests
play_user = slack.playback(hz=20, valid_len=15, replan_at=8)  # 15 valid steps, the next requested at the 8th
seating_user = slack.Seating(table_user, play_user["buffer_s"] * 1000, sizes=[1, 2, 4],
                             cycle_ms=(play_user["play_s"] - play_user["buffer_s"]) * 1000)
seats_user, over_user = seating_user.seat([("near", 30.0), ("far", 200.0)])     # (robot, its RTT in ms)
```

With batches of 1, 2 and 4 that take 80, 110 and 130 ms, this gives `(4, ["far"])`: the replica has capacity for
four robots, and the one 200 ms away is the one to move to another replica. `python examples/components/slack.py`
prints that case with no model and no GPU.

## The limits

- **Latency.** A robot fits the largest listed batch size whose batch time, plus the robot's own RTT and 50 ms, is
  within the buffer. A robot further away has less of its buffer left for the server, so it fits a smaller batch.
- **The batches a request arrives behind.** A replica runs one batch at a time, and robots send on their own clocks:
  a request that arrives as another batch runs waits for it, and is in time or not depending on how far apart the
  robots' requests fall. No formula has that, so `slack` runs 2, 3 and more robots on a simulated clock, as its
  server loop would serve them, from starts spread evenly over every way their requests can fall in a cycle (64 for
  two robots, 128 for more). A number of robots is within capacity when at least 95% of their responses are back
  before their robot's buffer runs out, each start counting the same.
- **Load.** In each of those starts the robots' batches may fill at most 0.9 of a cycle, the time between one robot's
  requests. Past that, responses queue behind each other however the requests are batched.

The first number of robots that fails ends the count: robots are admitted one at a time, so a replica reaches a
number of robots through every smaller one. In the example above a batch of four fits the buffer of a robot 100 ms
away (130 + 100 + 50 ms, within 350), and the capacity is four: on the clock four such robots have every response in
time, and their batches fill no more than 0.68 of a cycle.

Each robot is evaluated by its own RTT. When every robot was evaluated by the furthest robot's RTT, one far robot
lowered the batch size, and with it the capacity, of every near robot on the same replica.

## What batching itself gains

It depends on where the model spends its time. Measured on π0.5 on an L4, with the model's own batched call:

| Batch size | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| Requests per second | 5.99 | 8.21 | 9.38 | 10.52 | 10.66 |

That is 1.78 times at batch 16, and most of it by batch 4. That model spends most of a call on its prefix, the
images and the prompt, and batching does not amortize the prefix: 138 ms per request at batch 1, 119 ms at batch 16.
What batching does amortize is the action head, from about 53 ms to about 2 ms per request. **A model that spends
most of its time in the prefix gains correspondingly little.** These numbers are the model's, not `batcher`'s, which
has unit tests and no measurement of its own.

A server does not reach the model's batched call directly: each observation goes through the policy's input transform
first, and each row of the output through its output transform. Timed that way, also with π0.5 on an L4, batches of 1,
2, 3, 4 and 8 took 195, 312, 429, 540 and 1,027 ms: 5.1 requests per second at batch 1 and 7.8 at batch 8, 1.52 times.
Under JAX each batch size compiled on its first call, in 32 to 43 s.

One replica per GPU: one copy of the model answers all its robots in one pass.

## Sending the batches

With `serve()`, batching is one argument: `serve(policy_on_a_list, batch=[1, 2, 4, 8])` sends the observations waiting
to `policy_on_a_list` as one list, through the `Batcher` below, and each robot gets its own output back ([How do I serve
a robot policy from my own GPU?](how-do-i-serve-a-robot-policy-from-my-own-gpu.md#your-model-as-a-function)). In a
server of your own:

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

## Checked on a real model

One run checked whether the capacity `slack` computed was the capacity a real model has. For this model it was, and
on this model and GPU batching added no capacity: openpi's own server, one request at a time, served the same robots
in time. `slack` then computed the load by a formula, which gave too high a capacity for a model with slower batches:
[Why the capacity is computed on a clock](#why-the-capacity-is-computed-on-a-clock).

openpi's π0.5 (`pi05_droid`) on one L4, in a test server of about 200 lines that speaks openpi's WebSocket protocol and
runs [the loop in `docs/slack.md`](slack.md#in-a-server) around the batched call above, with openpi's own client on
each robot. The robots were simulated on the server's machine: each sent a 301 KB observation at the 8th of a
chunk's 15 valid steps, and a relay added an RTT of 20, 50 or 100 ms. A response is late when it arrives after the
buffer has run out. One run per cell, of 60 s, and of 120 s for 4 robots at 20 Hz.

| Policy rate | Buffer | Capacity computed | Robots connected | Late responses |
|---|---|---|---|---|
| 15 Hz | 467 ms | 2 | 1 | 0 of 112, at each RTT |
| | | | 2 | 0 of 225, at each RTT |
| | | | 3 | 21.0% and 25.7%, at 20 and 50 ms |
| | | | 4 | 99.2 to 99.7% |
| | | | 8 | 99.5 to 99.8% |
| 20 Hz | 350 ms | 1 | 2 | 39.8% and 89.5%, at 20 and 50 ms |
| | | | 4 | 99.9% |

`slack.per_replica` computed each capacity from the timing table, before any robot connected. At 15 Hz it held: no
response was late with 2 robots, and 21 to 26% were with one more. At 20 Hz one robot was not run, and two, one more
than the capacity, had 40% and 90% of their responses late. The test server admitted every robot that connected:
`Seating.seat()` is what limits a replica to its capacity, and this server did not call it.

Computed on a clock, as it is now, the capacity for these batch times is the same at each RTT. Two robots 100 ms
away are in time unless their requests fall 5 to 25 ms apart, where the second arrives too late to share the first
one's batch and its response, behind that batch, is back after the buffer ran out: 3.9% of their responses over the
starts, within the 5% the rule allows. The cell run with two robots at 100 ms had no late response.

Batching added no capacity here. openpi's own policy server, which runs one request at a time, also served 2 robots
with no late response, and a third would need 586 ms of forward passes in each 533 ms. A batch of 2 takes 312 ms where
two requests one by one take 390 ms, and a batch of 3 takes 429 ms, which with a robot's RTT and the 50 ms margin is
past the 467 ms buffer. Whether batching adds capacity depends on how a model's forward pass grows with its batch
size on a given GPU: [What batching itself gains](#what-batching-itself-gains).

- **At or below the capacity the loop costs nothing.** With 1 and 2 robots the median latency was within 1 ms of
  openpi's own policy server at every RTT, 216 to 304 ms.
- **Above it, batching completes more requests and does not make them arrive in time.** openpi's own server runs one
  request at a time: with 4 and 8 robots it completed 300 to 305 requests in 60 s, at a median latency of 800 to 807 ms
  and 1,615 to 1,627 ms, and 99% of responses were late. The test server completed 360 to 368 and 415 to 419, at 653
  to 667 ms and 1,160 to 1,173 ms, and 99% were late as well.
- **Above it, however a batch waits, responses do not arrive in time.** At 20 Hz two robots need two forward passes,
  about 201 ms each under load, in the 400 ms between a robot's requests, and no batch of two formed. Each run was in
  time until a queue built up, and 85 to 97% late from then on. That took 33.5 s and 4.7 s with `hold_s` (RTTs of 20
  and 50 ms), 7.1 s and 4.3 s with a fixed wait of 2 ms in its place, and 3.9 s and 0.3 s with one of 50 ms. The late
  share of a 60 s run, 39.8 to 89.5%, therefore measures when the queue formed, in one run per cell, and says nothing
  about the waiting rule.

One GPU, one model and one run per cell.

## Why the capacity is computed on a clock

The same check with three models, each on one L4 with simulated robots of a 15 Hz policy (a 467 ms buffer, a request
every 533 ms) 20 ms away, 60 s per cell, and two draws of when each robot starts. `slack` then computed the load by a
formula:

| Model | A batch of 1, 2, 4, 8 | Capacity by the formula | Measured | Capacity on a clock |
|---|---|---|---|---|
| π0.5 | 198, 320, 565 ms (no 8) | 2 | 2 robots: no late response in 4 cells | 2 |
| GR00T N1.6 | 223, 273, 399 ms (no 8) | 2 | 2 robots: no late response in 3 cells, 0.4% late in a fourth | 2 |
| SmolVLA | 256, 273, 303, 377 ms | 8 | 4 robots: 95.4% and 96.8% late. 8 robots: 99.2% and 99.6% late | 1 |

The formula assumed that robots whose batches of one size do not fit in a cycle share larger batches: four SmolVLA
robots as one batch of four, 303 ms of each 533. They did not. Robots send on their own clocks, and nothing brings
their requests together. The four ran as two fixed pairs in turn, each request behind the other pair's batch: 273 ms,
273 ms and the RTT, a median of 553 ms against a buffer of 467. A round of two pairs takes 546 ms, more than the
533 ms between a robot's requests, so after the first 4 to 6 s every response was late. The formula also returned the
largest number of robots that passed it, and with these times two failed where three to eight passed.

`slack` now runs the robots on a clock, which has both.
[`examples/fleet_on_a_clock.py`](../examples/fleet_on_a_clock.py) prints what it runs, here with SmolVLA's times:

```bash
python examples/fleet_on_a_clock.py --times 1:256,2:273,4:303,8:377
```

```text
  robots   responses in time   starts with a late response   load
       1             100.0%                0 of 1             0.48
       2             100.0%                0 of 64            0.53 to 0.96
       3              51.5%              125 of 128           1.00 to 1.01
       4               0.8%              127 of 128           0.57 to 1.05
       5               0.0%              128 of 128           1.04 to 1.31
       6               0.0%              128 of 128           1.09
       7               0.0%              128 of 128           1.28
       8               3.7%              128 of 128           1.05 to 1.14

slack.per_replica computes a capacity of 1 for these times (8 by the buffer alone): the first number of robots
with fewer than 95% of its responses in time over the starts, or a load above 0.9 in one of them, ends the count.
```

Four robots have nearly every response late, as on the L4. Two are in time in every start, at a load of 0.53 where
their requests fall close enough to share a batch and 0.96 where they do not. 0.96 is above the 0.9 limit, so the
capacity is 1. That limit was set on one stand-in model ([below](#what-the-load-limit-was-measured-on)).

It is a clock, not a model: every batch takes its timed p95, rounded up by 3% at most, nothing else runs beside it,
and the starts are a sample. Rerun on the model, SmolVLA on an L4 with simulated robots about 30 ms away, it gave a
capacity of 1 again, and the robots did this:

| Robots connected | Their requests | Run | Late responses |
|---|---|---|---|
| 2 | 150 ms apart | 60 s | 0 of 226 |
| 2 | 270 ms apart | 300 s | 0 of 1,125 |
| 3 | together | 60 s | 18 of 339 (5.3%), all in the first 5.4 s |
| 3 | evenly spread | 60 s | 97 of 338 (28.7%) |
| 3, and the server asked `Seating.seat()` as each connected | drawn from a seed | 30 s | one admitted, 0 of 57 late; two refused |

Three robots were late, as on the clock. Two were not, in either run: the clock has them in time as well, and
refuses the second for its load. The 0.9 is kept: at 0.96 of a cycle little is left for a network or a GPU that
varies, and these robots ran on the server's own machine. One run per cell.

## What the load limit was measured on

Four robots, 300 s and about 2,100 requests per run, on a stand-in model on a CPU paced like a 15 Hz policy (15 valid
steps, the next requested at the 8th: a 467 ms buffer, 533 ms between a robot's requests; batches of 1, 2 and 4 in
234, 254 and 299 ms). All four on one replica, a load of 0.95, had 85.7% of their responses back within the 467 ms.
Two per replica on two replicas, a load of 0.88, had 94.1%, on the slower network of the two runs. The 0.9 is set from
those two runs, in the serving stack `slack` was taken from, not with the file itself.

## When it applies

- A policy whose robots wait for each response leaves no slack, so its replica serves one robot.
- Each robot reports its RTT; without one, 100 ms is assumed.
