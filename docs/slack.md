# `slack`

**Scheduling by the slack an action chunk leaves, for one replica of a model: how many robots the replica can serve,
how long a request may wait for a batch, and which requests are dispatched first.**

One file, standard library only: [`sequence_ai/engine/slack.py`](../sequence_ai/engine/slack.py). Apart from `measure`,
which times a forward pass you provide, it runs nothing and sets no timer: the server that owns the batches calls it,
and passes it the current time.

```python
from sequence_ai.engine import slack

table = slack.measure(forward_of, sizes=[1, 2, 4])               # forward_of(n): one forward pass of n requests
play = slack.playback(hz=20, valid_len=15, replan_at=8)          # 15 valid steps, the next requested at the 8th
seating = slack.Seating(table, play["buffer_s"] * 1000, sizes=[1, 2, 4],
                        cycle_ms=(play["play_s"] - play["buffer_s"]) * 1000)
seats, over = seating.seat([("near", 30.0), ("far", 200.0)])     # (robot, its RTT in ms)
```

With forward passes of 80, 110 and 130 ms this gives `(4, ["far"])`: the replica has capacity for four robots, and the
one 200 ms away is the one to move to another replica: its request may arrive as the near robot's batch runs, and
would then miss its deadline. A batch of four fits the buffer of a robot 100 ms away, and four such robots, run on a
clock, have every response in time.

```bash
python examples/components/slack.py        # no model and no GPU: the rules on one pattern of arrivals
```

**Terms.** A *request* is one robot's request for its next action chunk. A *replica* is one loaded model, which runs
one batch at a time. A robot's *round-trip time (RTT)* is the time its requests spend outside the server: uplink,
downlink, and the client's own encoding and decoding (`outside_ms` in the code). The *buffer* is the playback time left
in the chunk a robot is executing at the moment it sends its next request. A request's *slack* is what is left of the
buffer after everything the request spends before its response arrives: its robot's RTT, its wait for a batch, its
batch's time and a margin. A response is *late* when it arrives after the buffer has run out. The rules below keep
the margin before that.

## The rules

- **The buffer comes from the policy.** A response has N valid steps at `hz`, and its robot sends the next request at
  step k, counted from when the previous request was sent. At that moment (N - k) / hz of playback is left, and the
  robot sends a request every k / hz (the cycle). Nothing is declared per request and nothing is read from a response.
  If `replan_at` is not given, k = N: no buffer, and no slack.
- **A request's deadline is its buffer, less its robot's RTT, less 50 ms.** The RTT is what the
  robot reports with its request (`outside_ms`, an integer number of ms from 0 to 5000), and 100 ms for a robot that
  reports nothing. A robot that sent its request late reports the playback time it has left (`left_ms`), and that
  value is used as its deadline.
- **Batch times are measured, not estimated.** `measure` runs each listed size once untimed, then three times. `Table`
  keeps the latest 64 forward passes of each size: their p95 is what a batch is taken to last, for waiting decisions
  and on the clock, and their p50 is what it counts for in the load.
  `Served` keeps what a batch took in the server once dispatched, including the handling around the forward pass; a
  size with 8 answered requests is timed by that.
- **Capacity, by latency.** A robot fits the largest listed batch size whose batch answers it in time: the batch's
  time, the robot's RTT and 50 ms, within the buffer.
- **Capacity, on a clock.** A replica runs one batch at a time and robots send on their own clocks, so whether a
  request is in time also depends on the batches it arrives behind. `slack` runs 2, 3 and more robots on a simulated
  clock, as the loop below serves them: from starts spread evenly over every way their requests can fall in a cycle
  (64 for two robots, 128 for more), each until the run repeats or for 48 cycles. A number of robots is within
  capacity when at least 95% of their responses are back before their robot's buffer runs out, each start counting
  the same, and in no start do the batches fill more than 0.9 of a cycle (the load: what the batches typically take
  for each request, with one request from each robot in a cycle). Robots' clocks drift, so over a long run a fleet
  meets every way its requests can fall, and the share over the starts is the share of that run. The first number
  that fails ends the count: robots are admitted one at a time, so a replica reaches a number of robots through
  every smaller one. The clock rounds each batch time up by 3% at most and each RTT up to 10 ms, and keeps each
  verdict by them: a server's timings move by a few ms from one request to the next, and rounded they ask a question
  already answered.
- **Each robot by its own RTT.** Robots are admitted nearest first, and robots that fit the same batch
  size are batched together. A replica runs one batch at a time, so each group's batch must still be in time behind
  the longest batch of another group, and the robots are run on the clock each with its own RTT. The robots beyond
  that capacity, which are the furthest ones, are returned as `over`. A far robot never lowers the capacity available
  to nearer ones.
- **The open batch waits for the replica's other robots** until its most urgent request can wait no longer: the
  latest time that request's response may leave, less the batch's time with one more request in it, less 50 ms. It
  waits only for a robot that is admitted to the replica, whose next request is due safely before then (a robot
  sends its next request k steps after its last one; one whose requests have arrived early and late is counted as
  that much later), and whose own deadline allows the batch. It is dispatched immediately when every admitted robot
  has a request in it, or when one of its robots has no chunk executing yet.
- **A far robot's request is dispatched separately, first**, where the whole batch would miss its deadline: in the
  largest listed size its deadline allows, and only where the requests behind it still meet theirs.
- **A far robot's batch waits one control step at most** for a near robot's request that is due as it would run.
  Dispatched immediately, it would make the near request wait behind it for a full batch time.
- **A robot that sends nothing for four of its periods is dropped from the schedule**: no batch waits for it.
- **Where a policy sets `max_wait_s`** (up to 60 s), that wait replaces the buffer, for capacity and for the batching
  delay. Its robot may run out of actions, and waits.

## In a server

```python
aligner = slack.Aligner(table, play, seated=lambda: robots_on_this_replica)    # one per replica

aligner.arrived(robot, now, outside_ms=slack.outside_ms(reported))     # each request as it arrives
wait_s = aligner.hold_s(open_requests, now)                            # the open batch may wait this long; 0.0: run it
first = aligner.split(open_requests, now, [1, 2, 4], free=False)       # indexes to dispatch first; None: all as one
aligner.answered(batch, now)                                           # as a batch's responses leave
table.add(len(batch), forward_ms)                                      # each forward pass's time, as it is served
```

`open_requests` and `batch` are lists of robots, by the names their requests arrive under (strings; a request with
no name is not scheduled). `now` is one monotonic clock,
in seconds. Call `split` only when a replica is free to run the batch.
`seating.seat(list(aligner.outsides_ms(now).items()))` computes the capacity of a replica from the RTT each robot last
reported; `Seating(..., most=n)` caps a replica's robots (64 unless given; the serving stack this was taken from passes
its largest listed size). [`examples/components/slack.py`](../examples/components/slack.py) has such a loop in about
fifty lines, on a simulated clock.

| In the file | What it provides |
|---|---|
| `measure`, `Table`, `Served`, `SizeFailed` | the time a batch of each listed size takes |
| `playback` | a policy's `hz`, N and k as `play_s`, `buffer_s`, `step_s` (and `wait_s`) |
| `per_replica` | the capacity of a replica whose robots all have the same RTT |
| `in_time`, `on_a_clock`, `starts` | whether robots with these RTTs are answered in time, over the starts; one run on the clock; the starts a number of robots is run from |
| `Seating` | `fit`: the largest batch that answers one robot in time; `seat`: the capacity, and which robots are over it |
| `Aligner` | `arrived`, `hold_s`, `split`, `answered`, `left`, `due`, `outsides_ms`, `replay` |
| `outside_ms`, `left_ms` | what a robot reported, parsed as an integer number of ms, or `None` |

## Relation to `batcher`

[`batcher`](../sequence_ai/engine/batcher.py) dispatches a batch when it is full, or `wait_ms` after its first call: one
wait for every caller, and no notion of a robot. `slack` answers the two questions that leaves open: how many robots
a replica can serve at all, and, for the requests waiting now, how long they may still wait and which are dispatched
first. They are two components, each used alone, and `Batcher` does not call `Aligner`: a server that batches by
slack runs its own loop, as the example does.

## Measured

The first five were measured with this file, the fifth on a stand-in for a model. The others were measured in the
serving stack it was taken from.

- **The capacity it computes, on a real model.** π0.5 on one L4, in a test server built from this file around the
  model's batched call, with simulated robots: 15 valid steps, the next request at the 8th, 60 s per cell. At 15 Hz
  (a 467 ms buffer; batches of 1, 2 and 3 in 195, 312 and 429 ms) `per_replica` computed a capacity of 2: 2 robots
  had no late response, 3 had 21% and 26%, and 4 had 99%. At 20 Hz (350 ms) it computed 1, and 2 robots had 40% and
  90% late. The server admitted every robot, so the cells above the capacity show what `seat()` prevents. Batching
  added no capacity on this model and GPU: openpi's own server, one request at a time, served the same 2 robots.
  `per_replica` then computed the load by a formula. On a clock, as now, those capacities are the same, for robots
  20, 50 or 100 ms away.
  [How many robots can one GPU serve?](how-many-robots-can-one-gpu-serve.md#checked-on-a-real-model) has every cell.
- **The capacity, on three models.** One L4, simulated robots of a 15 Hz policy 20 ms away, 60 s per cell, with the
  load computed by that formula. For π0.5 and GR00T N1.6 it computed 2, and 2 robots had no late response (0.4% in
  one cell of eight). For SmolVLA (batches of 1, 2, 4 and 8 in 256, 273, 303 and 377 ms) it computed 8, and 4 robots
  had 95% and 97% of their responses late, 8 had 99%. Computed on a clock, the three capacities are 2, 2 and 1:
  [Why the capacity is computed on a clock](how-many-robots-can-one-gpu-serve.md#why-the-capacity-is-computed-on-a-clock).
- **The capacity on a clock, on a model.** SmolVLA on one L4 again, the same policy and test server, simulated robots
  about 30 ms away. The clock gave a capacity of 1, by the load: two robots' batches fill more than 0.9 of a cycle
  where each request runs alone. Two robots, one more than the capacity, had no late response in 226 (their
  requests 150 ms apart, 60 s) and none in 1,125 (270 ms apart, 300 s; a median latency of 278 ms in the first third
  of that run and 279 ms in the last). Three robots had 5.3% of their responses late where they started together
  and 28.7% where they started evenly spread. Where the server asked `Seating.seat()` as each robot connected, it
  admitted one and refused the next two. One run per cell.
- **A batch held for a robot that arrived just after it had to go.** The same run, SmolVLA, two robots whose requests
  were 150 ms apart: the first request was held 115 ms for the second, which arrived 35 ms after the batch had to go,
  and 55 of the second robot's 56 responses were late. None was late in a server that held no batch. A batch waited
  for any robot that could arrive in time at the earliest, one control step sooner than it does. It now waits for a
  robot whose request is due in time. Rerun on the model after the change, two robots 150 ms apart had no late
  response in 226, and the least any response had to spare was 79 ms.
- **A batch held for a robot whose requests arrive a little early or late.** A stand-in for the same model on a laptop
  (each batch sleeps the model's time), two robots, 120 s per cell: 2 of 512 responses were late where their requests
  were up to 119 ms apart, none of 267 where they were 150 ms or more apart, and 31 of the second robot's 113 where
  they were 120 to 149 ms apart, just past what the first request could wait. The rule then took a robot's next
  request to be due three times the spread of its arrivals earlier, so a batch was held for a request that came just
  after it had to go. It now takes it that much later. On a clock with requests arriving 5 ms early or late (standard
  deviation), two such robots had 6.4% of their responses late before, and 0.02% after. Two robots further apart ran
  120 s at a load of 0.95 with no late response in 445.
- **Four robots on one replica for two minutes**: 20 Hz, 15 valid steps, the next request at the 8th (a 350 ms
  buffer), a stand-in model on a CPU. With every robot assumed to have a fixed 100 ms RTT and a batch timed
  by its forward pass alone, 459 of 1,114 responses arrived after their robot had run out of actions. A request
  waited a median 58 ms for a batch, took 197 ms more in the server and about 131 ms outside it: 386 ms against 350.
  Per-robot RTT and batch times as served were added because of that. The late share after those
  changes was not measured at the same settings.
- **Load.** Four robots, 300 s and about 2,100 requests per run: a stand-in on a CPU paced like a 15 Hz policy (15
  valid steps, the next request at the 8th: a 467 ms buffer, 533 ms between a robot's requests; batches of 1, 2 and 4
  in 234, 254 and 299 ms). All four on one replica, two batches of two in each 533 ms and a load of 0.95, had 85.7%
  of their responses back within the 467 ms. Two per replica on two replicas, two batches of one and a load of 0.88,
  had 94.1%, on the slower network of the two runs. `LOAD_MAX`, 0.9, is set from those two runs.
- **Capacity set by the furthest robot.** A replica that had capacity for four robots reported a capacity of one once
  a far robot was on it, and the three robots on it were not answered in time. Per-robot admission was added because
  of that.
- **Dispatching a far robot separately**, a simulation of one replica (15 Hz, a 467 ms buffer, a batch of one
  234 ms, 120 s per run): a robot 60 ms away, beside one 300 ms away, was answered in time 80% and 83% of the time in
  two runs, and 88% and 92% once the far robot's request was dispatched separately. In a third run the replica was
  too loaded for the two and neither was answered in time, either way.
- **A near robot behind a far robot's batch.** On a replica with two robots (15 Hz, a 467 ms buffer, batches of 1
  and 2 in 230 and 250 ms), a robot 23 ms away beside one 250 ms further ran out of actions 31 times in 240 s, and
  not once alone. The one-step wait was added because of that; its effect afterwards is not recorded here.

## Where it helps

- **Several robots on one replica, and a policy that leaves slack** (k < N). The slack is spent on fuller batches: in
  the example, two near robots are answered as one batch of two instead of two forward passes, and still have time
  to spare.
- **Robots at different distances.** A far robot is batched by its own deadline, is dispatched ahead of a batch that
  would make it late, and is the one named to move. The near ones keep their batch size and their capacity.
- **Before any robot connects.** A timing table and a policy's three numbers give a replica's capacity without a
  trial: `slack` runs the fleet on a simulated clock, and
  [`examples/fleet_on_a_clock.py`](../examples/fleet_on_a_clock.py) prints what each number of robots gets.

## Limitations

- **No slack, nothing to wait in.** A policy whose robot sends a request only when a chunk is fully executed (k = N)
  has no request held, and a replica serves one of its robots. `max_wait_s` changes that only by making the robot
  wait.
- **A robot that does not report its RTT** is assumed to have 100 ms. If it is further away, its
  responses arrive late and nothing here detects it.
- **One robot per replica** has nothing to batch with.
- **A replica near full load.** Past about 0.9 of a cycle, responses queue behind each other however the requests are
  held (the load measurement above). Capacity limits are what keep a replica out of that state, not the batching
  delay.
- **The capacity comes from a simulated clock, not from your model under load.** Every batch takes its timed p95,
  rounded up, nothing else runs on the GPU, each robot has one request in flight and an RTT that does not vary, and a
  robot that sent a request late reports the playback time it has left. The starts are a sample: on 150 fleets the
  share of late responses they gave was within 2 points of what 1,024 starts gave, and an arrangement that fills
  more than 0.9 of a cycle can fall between them. For three models' timing tables the clock gives what was measured
  with them where it was measured: 2 robots for π0.5 and GR00T N1.6, and 1 for SmolVLA, where three or more robots
  had late responses. All of that is one rate and one replan step (15 Hz, the next request at the 8th of 15 steps),
  on one kind of GPU.
- **Within capacity is not never late.** A fleet is within capacity with up to 5% of its responses late over the
  starts, and those fall where the robots' requests are a particular distance apart: 5 to 25 ms for two π0.5 robots
  100 ms away, with that model's batch times on an L4. Robots whose clocks keep them at such a distance are late for
  as long as they stay there. The share is what a long run of drifting robots gets, not a bound on any stretch of
  it.
- **A request due at the very instant its batch must go.** Batch times, RTTs and a cycle that add up exactly leave
  that choice to the last bit of a sum. The clock then takes one of the two ways a real server, whose timings vary,
  would alternate between, and the capacity is that of the way it took.
- **Where the load ends the count, the capacity may be low.** For SmolVLA on an L4 the clock has two robots in time
  in every start, at a load of 0.53 where their requests fall close enough to share a batch and 0.96 where they do
  not. 0.96 is above the 0.9 limit, so the capacity is 1, and on the GPU two such robots had no late response in 226
  and in 1,125 (above). The limit was set on one stand-in model (the load measurement above) and is kept: at 0.96 of
  a cycle little is left for a network or a GPU that varies, and those two runs had their robots on the server's own
  machine.
- **A policy that requests its next chunk early** (k < N / 2) has more buffer than time between requests, so its
  robots can be in time while they wait for each response and send less often than every k steps. The clock runs
  them that way, and counts the load at the rate they ask for. No capacity for such a policy has been checked on a
  model.
- **The first computation takes time.** About 5 ms of one core for a capacity of 2, 35 ms for 4, 0.2 s for 8, 1 s
  for 16 and 7 s for 32, and more for a policy that requests early. Each verdict is kept, so asking again is a
  lookup. Compute it when a timing table or a robot's RTT changes, not on the path of a request.
- **Robots at different distances are admitted for the worst arrival pattern.** A far robot is `over` beside nearer
  ones wherever its request could arrive behind their batch and miss its deadline, even where, as its requests
  happen to arrive, it is answered in time. The example prints exactly that case.
- **It moves no robot and starts no replica.** `over` names which robots to move; where to is the caller's decision.
- **A forward pass is timed by its batch size alone.** A model whose time depends on its input is timed by its p95.

## Related work

Scheduling several robots' requests by the slack of their action chunks is also the subject of published work, among
it *Action Chunk Scheduling for Batched Robot Policy Serving* (the Armory system, arXiv:2608.00337) and *FirmVLA:
Sequence-SLO Serving for Heterogeneous VLA Fleets* (a NeurIPS 2026 workshop paper). This file is the version General
Sequences serves with. It has not been compared with either.

## Constants

| Constant | Value | What it is |
|---|---|---|
| `MARGIN_MS` | 50 | the margin kept between a response and its robot's deadline |
| `ROUND_TRIP_MS` | 100 | the RTT assumed for a robot that has not reported its own |
| `LOAD_MAX` | 0.9 | the largest share of a cycle a replica's batches may fill |
| `MAX_ROBOTS_PER_REPLICA` | 64 | the most robots a replica serves, however fast it is |
| `CLOCK_CYCLES`, `CLOCK_WARM` | 48, 8 | the cycles a start is run for on the clock, and how many of the first are not counted |
| `CLOCK_STARTS` | 128 | the starts a number of robots is run from on the clock; half as many for two robots |
| `IN_TIME_MIN` | 0.95 | the share of a fleet's responses that must be back before their buffer runs out, over the starts |
| `CLOCK_BITS`, `CLOCK_RTT_MS` | 6, 10 | the clock's resolution: a batch time rounded up to 6 significant bits (3% at most), an RTT up to 10 ms |
| `WINDOW` | 64 | the runs kept for each batch size |
| `RUNS` | 3 | the timed runs of each size, after one untimed |
| `SERVED_MIN` | 8 | the answered requests after which a size is timed by what its batches took |
| `OUTSIDE_MAX_MS`, `LEFT_MAX_MS` | 5000, 600000 | the largest values a robot may report for its RTT, and for the playback time it has left |
| `WAIT_MAX_S` | 60 | the longest `max_wait_s` |
| `GONE_PERIODS` | 4 | the periods of silence after which a robot is dropped from the schedule |
| `PERIOD_WEIGHT` | 0.2 | the weight of a new gap between a robot's requests in its estimated period |
