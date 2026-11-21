# Changelog

What changed in each release of SequenceLink's package, `sequence-ai`, newest first. It starts at 0.34.10. The same notes are public at
https://generalsequences.com/docs/changelog/.

## 0.66.0 — 2026-11-21

Teleoperation is open source and takes a headset, a gamepad or the keys, `seq serve` runs a template by its name,
`seq data export` turns recordings into LeRobot v3.0 datasets, and `connect()` runs on Python 3.9 again.

Needs `pip install 'sequence-ai[teleop]'`, on Python 3.11 or later, for teleoperation, and `'sequence-ai[data]'` for a
LeRobot dataset.

### Added

- `seq teleop serve` pairs a robot with consoles on a machine of your own, and a person drives it from its page or
  `sequence_ai.teleop()`.
- `seq serve --cfg-parallel` splits a model's classifier-free guidance across two GPUs, kept where acts get faster
  and the answers stay within your `--tolerance`.

### Removed

- `@stable`, `@beta` and `sequence_ai.serve` are gone: a template has one version, and `seq serve` serves a
  `@seq.policy` class.

## 0.65.0 — 2026-11-20

`sequence-ai` is the open-source framework (a `@seq.policy` class served by `seq serve`, and `connect()` on the
robot), and `seq deploy`, evals and the platform's other commands move to a separate package, `sequence-cloud`.

Needs `pip install sequence-cloud --extra-index-url "https://$SEQUENCES_API_KEY@api.generalsequences.com/simple"` for
`seq deploy`, evals and the platform's other commands.

### Added

- `seq serve` keeps each speed-up that leaves the outputs exactly the same and makes acts at least 10% faster, and runs
  `--torch-compile`, `--fused-kernels`, `--tensorrt` and `--precision` only with the `--tolerance` you give.
- `sequence_ai.engine.load` packs a checkpoint (safetensors, .pth or orbax) into one file that loads onto the GPU in
  parallel reads.

### Changed

- `connect()` to your own server streams each observation with `observe()` as it is made, `act()` carries none, and a
  camera picture goes as a video keyframe unless `pictures="exact"`.

## 0.64.1 — 2026-11-19

A benchmark that types a setting no eval can set is refused when you deploy it, and a `--param` value reaches a text
parameter as you typed it.

### Changed

- `seq deploy` refuses a benchmark whose constructor types a setting no eval can set, such as
  `scale: float | str | None`, and says how to type it.

### Fixed

- `--param benchmark.scale=1.10` reaches a `str` parameter as `"1.10"`, as typed, not as `"1.1"`.

## 0.64.0 — 2026-11-19

A template's policy runs on your own GPU, `chat()` takes your own OpenRouter key, a policy names its robot arm, and an
act after a pause goes in halves where that saves a round trip.

### Added

- Off the platform, `seq serve` serves a template's policy from your GPU (`seq weights get`), and `chat()` takes
  `server="openrouter"` with your own OpenRouter key.
- `@seq.policy(robot=)` names the arm a policy is for, which a benchmark's `"{policy.robot}"` argument takes unless an
  eval's `--param` or `--sweep` sets another.

### Fixed

- `openpi.keep()` and `openpi.patch()` keep a connection through a long first inference, and `openpi.connect()`
  waits for a server still loading.

## 0.63.1 — 2026-11-18

A robot whose address changes while it runs is let back into its GPU by itself, and on a line whose round trip
varies, fewer acts wait for a ping.

### Added

- `network.connect(url, on_unreachable=fn)` calls `fn(link)` when an act cannot reach its server, and sends the act
  once more if it returns True.
- With `follow=`, a line that cannot reach its GPU because its address changed is let in again, and the act is sent
  once more.

### Fixed

- On a line whose round trip varies, a ping is left out if its answer could still be coming when the next act
  is due.

## 0.63.0 — 2026-11-18

`sequence_ai.serve(policy)` serves a model in one call with the defaults on at both ends, and
`python -m sequence_ai.check <url>` says which are on.

### Added

- `connect(url, lines=["eth0", "wwan0"])` sends each act over two uplinks, and the robot uses the first answer.
- A rented GPU's ports take IPv6 ranges, and `connect(url, follow=token)` lets a robot's changing address in, each
  line's with `lines=`.

### Fixed

- A robot with no standby whose worker stops answering reconnects within a few control periods, not after its act's
  timeout.

## 0.62.3 — 2026-11-17

`pip install sequence-ai` now installs everything the quick start needs, and `sequence-ai` is called an open-source
framework.

### Changed

- `pip install sequence-ai` installs uvicorn, numpy and zstandard too: the `server` and `arrays` extras are gone, and
  naming them only draws pip's warning.
- The README, its banner and the citation call `sequence-ai` an open-source framework, not an SDK.
- The README's results fit in short tables, and its limitations are five plain points, with every number in
  `docs/measurements.md`.

## 0.62.2 — 2026-11-17

The documentation now says what `openpi.keep()` does with openpi's own server: its pings wait while the server runs
inference, so on Linux without BBR the window can still be halved.

### Fixed

- `openpi.keep()`'s and `window.Warm`'s docstrings no longer say a pinged connection is never idle: a server that
  answers no ping while it works leaves it silent.
- `docs/measurements.md` adds a run that counts where the window is cut: with `keep()`, openpi's server and no BBR,
  Linux halved it before most requests.
- The openpi how-to says that BBR on the robot keeps the window where openpi's server cannot answer the pings.

## 0.62.1 — 2026-11-17

A robot that runs openpi's own client against openpi's own server on a GPU of your own sends its pictures as JPEG,
with one changed line on each side.

### Added

- `image_tools` from `sequence_ai.network.openpi`, imported on the robot in place of `openpi_client`'s, fits a frame as
  openpi's `resize_with_pad` does and sends it as JPEG.
- `python -m sequence_ai.network.openpi scripts/serve_policy.py`, or `openpi.patch()` in a server script of your own,
  makes openpi's server ping each robot every 100 ms and decode its JPEG pictures.
- A websockets server of your own does the same with `frames.decode`, which turns a JPEG or PNG picture into its array,
  and `ping.serve`.

## 0.62.0 — 2026-11-17

`sequence-ai` is open source and carries the components that make remote inference fast on any GPU, yours or one of
ours.

### Added

- `sequence_ai.engine` serves a policy fast on a GPU of your own: CUDA graphs, batching, replicas and scheduling by
  slack.
- `sequence_ai.network` keeps a robot's connection to its GPU fast across the robot's pauses.
- `sequence_ai.connect("http://your-gpu:8000")` reaches a policy server of your own with `observe()` and `act()`.

## 0.61.5 — 2026-11-16

`seq gpu ssh` makes no SSH key for a GPU that is not there or not running, and `seq gpu add` says a refused `--count`
once.

### Changed

- `seq gpu ssh` with no key in `~/.ssh` makes one only once the platform says that machine runs, and never writes over
  a key there.
- `seq gpu add` says a refused `--count` in the platform's one sentence, which names the counts, where a line under it
  said them again.

## 0.61.4 — 2026-11-16

`seq policy status` says why a copy seats one robot where the policy's declarations make it so, in the words `seq
deploy` uses, and `sequence-ai[teleop]` needs Python 3.11.

### Changed

- `seq policy status` says it on the line under `robots per worker`: an episode that holds its copy, no
  `@seq.concurrent`, no valid length, or asking at the last valid step.
- `seq deploy` says why a policy without `@seq.concurrent` seats one robot a copy: infer takes one act at a time.
- `sequence-ai[teleop]` installs on Python 3.11 or later, as aiortc 1.16 does, and teleop on 3.10 says it needs 3.11.

## 0.61.3 — 2026-11-16

A copy's seats are counted by running its robots on a simulated clock, as the worker batches them, so a policy with
slow batches may seat fewer robots than before.

### Changed

- A copy seats robots while at least 95% of their answers come in time and batches fill at most 90% of the time between
  asks.
- A batch waits only for an act that would still make it if it came a little late, so fewer answers come late.
- `seq account` names all five holders of your GPU cards, serving, evals, build checks, shells and GPUs, 0 among them,
  as the console does.

## 0.61.2 — 2026-11-16

A GPU no place has a card for fails in minutes and is not billed, and `seq gpu add` says on one line which counts a
machine holds when it refuses a count.

### Changed

- A GPU fails as `gpu_no_card`, unbilled, once every place that could open it has said it has no card, where it waited
  out 20 minutes.
- `seq gpu add` follows a refused `--count` with one line, `cards a machine holds: 1, 2, 4 or 8`, where each count had a
  line of its own.
- `seq` and the SDK's errors say your GPU limit in cards, `GPU limit 4 / 15 cards` in `seq account`, where they said
  GPUs.

## 0.61.1 — 2026-11-15

`seq gpu add --count` is how many cards its one machine has, 1, 2, 4 or 8, and for another number you add several GPUs.

### Changed

- `--count` and `count=` open one machine and refuse any count but 1, 2, 4 or 8, where 0.61.0 split other counts over several.

## 0.61.0 — 2026-11-15

`seq gpu add --count` is now how many cards, 8 at most, on one machine where one holds them all.

### Added

- Each GPU machine says its `cards`, the `request` that opened it and that request's `request_cards`, and `seq gpu` says them as `L4 × 4`.

### Changed

- `--count` and `count=` are cards, split over the fewest machines, which run together, when one cannot hold them, and your GPU limit counts each card.

### Fixed

- `seq robot record` and `sequence_ai.record` run as your `seq login` sign-in where no key is set, where they asked for a key.

## 0.60.0 — 2026-11-15

A robot's Record is now a switch: you record its steps to a file of yours, and the platform keeps none.

### Added

- `seq robot record <name>` writes a robot's steps to a file that `sequence_ai.read_record` reads, and a robot on teleop alone writes its own.

### Changed

- `sequence_ai.record` takes a robot's name with no record key, and `seq record` and `record.retention_days=` are gone.
- Saving a version's image or a GPU's snapshot is no longer billed, though a snapshot's storage still is, by the GB-month.

## 0.59.9 — 2026-11-14

`seq gpu add` now removes what it added when stopped before it answers, and says a GPU's price as its card's plus CPU and memory.

### Changed

- `seq gpu add` says an L4 costs `$0.80/h + CPU and memory`, where it said one total, `$1.22/h`.

### Fixed

- Ctrl-C while `seq gpu add` is still being answered removes what it added, where it said "cancelled" and left them to run.
- Pressed twice, it says the GPUs may have been added and how to see them.

## 0.59.8 — 2026-11-14

`seq gpu add` now says how a GPU that had its card and did not start ended, and exits 1 for it.

### Changed

- `seq gpu add` exits 1 (was 7), and `sequence_ai.gpu.add` raises `InternalError`, when a GPU had its card and did not start.

### Fixed

- `seq gpu add` says a GPU's container stopped before it ran, and points to `seq gpu logs`, where it said it did not start within 20 minutes.

## 0.59.7 — 2026-11-13

`seq gpu` now says what a GPU it cancelled was billed, and exits with the same codes as every other command.

### Changed

- `seq gpu` exits as the error it prints says: no card 7 (was 4), limits 5, credit 6 (were 3), snapshot refusals 4 or 5 (were 2).

### Fixed

- `seq gpu add` says what a GPU it cancelled or that never came up was billed, where it always said "not billed".

## 0.59.6 — 2026-11-13

A GPU can now serve your robots straight at its own address, on ports you open to the addresses you allow.

### Added

- `seq gpu add --port 8000 --allow CIDR --run CMD --name NAME` opens your server's ports to those addresses, keeps its command running and gives it an address your robots keep.
- `seq gpu allow` changes who reaches the ports, `seq gpu logs` shows what happened to a GPU and what its command printed, and `seq gpu save` keeps a snapshot that `seq gpu add --from` starts from.

### Changed

- `seq gpu add` now waits until each GPU runs, Ctrl-C cancels it unbilled, and `sequence_ai.gpu` raises `GpuBadRequest`, `GpuOverLimit` or `GpuNoCard`.

## 0.59.5 — 2026-11-12

GPUs now show the place you asked for, which must be in the United States, and ended GPUs move to a history.

### Changed

- `seq gpu add --place` refuses a place outside the United States, and `seq gpu places` lists only US cities.
- `seq gpu list` shows your running GPUs by the place you asked for, and `seq gpu history` lists the ended ones.

### Fixed

- A GPU's spent amount now includes the seconds another place held its card, as billed, and `seq gpu ssh` leaves
  `~/.ssh` untouched.

## 0.59.4 — 2026-11-12

A robot connected to `"teleop"` is ready about three seconds sooner.

### Changed

- `wait_until_ready()` on a robot connected to `"teleop"` returns about three seconds sooner than it did.

## 0.59.3 — 2026-11-12

`seq gpu places` now says the other name a city was found by, as Warsaw is for "paris", and an airport code finds
its city only when typed in full.

### Added

- `seq gpu places paris` ends Warsaw's line with `(as Paris of the North)`, and each city `sequence_ai.gpu.places`
  returns has `also`: the other name it was found by, or `None` where what you typed starts its own name.

### Fixed

- A robot can connect to a policy the moment it is put on the robot's list, not up to 30 seconds later.
- The console's Add GPUs sends the place it shows, so a city typed but not chosen clears the latitude and longitude.

## 0.59.2 — 2026-11-11

A GPU machine's place can now be chosen from a list of cities, and a robot's key has no spending limit.

### Added

- `seq gpu places san fr` and `sequence_ai.gpu.places("san fr")` list the cities whose name starts with what you
  type, each with the latitude and longitude `--place` takes.

### Changed

- A robot's key has no spending limit, including a key from before that had one.
- The console's Teleoperation tab has no action box, so a robot's actions come from a console in Python.

## 0.59.1 — 2026-11-11

`seq robot show` now lists a robot's teleop connections, as the console's Robots page does.

### Fixed

- `seq robot show arm_user` prints `teleop · connection <id>` for each teleop connection, so a robot connected to
  `"teleop"` alone no longer reads as having no connection.

## 0.59.0 — 2026-11-11

A robot is its own thing now, with its own key and a list of names it connects to, `"teleop"` among them, and you can
open GPU machines of your own.

Needs `sequence-ai[teleop]` (Python 3.10 or later) on a robot that connects to `"teleop"` and on a console in Python:
`pip install 'sequence-ai[teleop]'`. `connect(teleop=True)`, `teleop("<policy>", ...)`, `seq robot add`,
`seq robot revoke` and `seq policy set <name> capture=on` now raise, saying the new form.

### Changed

- `seq robot create arm_user` prints the robot's key, and `seq robot set arm_user policies=pick_user,teleop record=on`
  sets its list and keeps what it does.
- `connect("pick_user", "teleop", api_key=ROBOT_KEY_user)` returns one handle per name, and a console connects straight
  to the robot, its cameras as video.
- `sequence_ai.gpu.add(place="San Francisco, CA", gpu="L4", count=2)` opens GPU machines near that place, and
  `seq gpu ssh <id>` gives you a root shell on one.

## 0.58.3 — 2026-11-10

Closing a connection now frees its policy copy for the next robot, `seq gpus` says which machines pay which CPU and memory
rates, and `seq account delete --help` says what deleting charges first.

### Fixed

- A connection closing ends the episodes it only observed, so the next robot no longer waits up to 300 seconds for a policy
  copy.
- `seq gpus` says which machines pay the serving CPU and memory rates, and `seq billing plan` lists Custom as agreed per
  account.
- `seq account delete --help` says a due of $0.50 or more is charged first, and the account is kept if it cannot be
  paid.

## 0.58.2 — 2026-11-10

Prices are written as the price list does with billing days in UTC, `seq/eval.md` gives `seq eval report`'s flags and
postpaid billing, and recorder items name each robot.

### Fixed

- `seq gpus` and `seq deploy --dry-run` write a price of whole cents to the cent, name the card's price and list
  several-card machines.
- `seq gpus` says which machines are billed at least 60 s, and `seq account` and `seq billing` say billing days in UTC.
- Recorder items name their robot by its key's name and last four, and an episode under way when record comes on starts
  with a `gap`.

## 0.58.0 — 2026-11-10

Your own recorder of a policy's steps, teleop episodes kept only when you say, workers priced as billed, and benchmarks on
L4 or RTX-PRO-6000.

Needs sequence-base 0.8.10 or later; `pip install -U sequence-ai` brings it.

### Added

- `sequence_ai.record()` and `seq record add` let a recorder of yours read each step from the workers, and the platform
  keeps none.
- `teleop.keep=on` keeps the episodes of robots connected with `teleop=True`, now off unless set.

### Changed

- A worker is priced as billed, its card plus its card's standard machine, and a shared benchmark machine at the
  greater of request and use.

## 0.57.5 — 2026-11-09

`seq deploy` says which step each line belongs to when two steps run at once, `seq usage` says all a deleted policy
cost, and `seq account` names a Link payment method by its email.

### Fixed

- A step that ends under another step's line names itself: `upload done in 15 s · 8 files, 53.9 KB`.
- `seq usage --policy` of a deleted policy says its spend with its storage and image saves, as `billing()` does, and
  `seq usage --group deployment` shows each name whole.
- `seq account` names a Link payment method by its email, not `ending ····`.

## 0.57.4 — 2026-11-09

A failed build's lines are said once, and `seq policy rm` lists every version it deletes.

### Fixed

- When a build fails, `seq deploy` shows its last lines once, under the step that printed them.
- `seq policy rm` and its `--dry-run` list the failed versions it deletes too, marked as failed.

## 0.57.3 — 2026-11-09

Errors and messages now say only what is so: no worker's address, no card on CPU, a cut line marked "…".

### Fixed

- A robot's errors no longer show its worker's address, even in the traceback under `NewVersion`.
- `seq deploy` no longer asks you to fix a build that the platform failed.
- `seq usage --policy` says the acts no listed robot reported, so its robots add up to its acts.

## 0.57.2 — 2026-11-09

A `breakpoint()` left in your code now does nothing in the build's check and on serving workers, and stops only on
`seq policy debug`'s machine, which plays your policy again each run.

### Fixed

- When `--timeout` stops a command, `seq policy shell` and `seq benchmark shell` say so on stderr, and still exit 124.
- `connect()` with no key names `seq robot add --policy NAME`, and an upgrade given `--to v7` names the
  `TEMPLATE@v7` to write.
- `seq usage --policy` names a policy's image saves beside its storage, so its spend adds up to the use above it.

## 0.57.1 — 2026-11-08

`seq usage --policy NAME` now gives what a deleted policy cost.

### Fixed

- `seq usage --policy NAME` gives a deleted policy's use under its deleted name, where it said $0.00.
- Two deleted policies of one name each get a line of their own.

## 0.57.0 — 2026-11-08

You can now open a terminal on the machines your code runs on, and stop at a `breakpoint()` on a machine set up as your
workers are.

### Added

- `seq policy shell NAME` and `seq benchmark shell NAME` open a terminal on their machine, and a command typed after
  `--` runs in it.
- `seq policy debug NAME` plays your policy's sample episode on a machine of its own, and any `breakpoint()` stops in
  your terminal.

### Changed

- `seq usage` calls a debug machine and a benchmark's shell machine shells.

## 0.56.10 — 2026-11-08

A dry run on an A100, A100-40GB or H100 now gives the price you are billed: the card's price, which includes its CPU
and memory.

### Fixed

- `seq deploy --dry-run` no longer adds CPU and memory to an A100, A100-40GB or H100's price.
- `seq gpus`, `seq policy status` and `seq deploy` say those cards' CPU and memory are in the card's price.

## 0.56.9 — 2026-11-06

You now set how long a deploy may run, whatever happens to your terminal, and a benchmark that crashes as it starts
fails with its own error instead of timing out.

### Changed

- `--timeout S` on `seq deploy`, `seq policy upgrade` and `seq benchmark upgrade` sets how long it may run, then the
  platform stops the build and bills what ran.
- Without `--timeout`, `seq deploy` follows the build until it ends, however long that takes.
- A failed `seq policy upgrade` or `seq benchmark upgrade` now exits non-zero.

## 0.56.8 — 2026-11-05

Starter's free credit is now $5 a month, from December 1 for existing accounts, policies always run nearest their
robots, and GPUs start faster when cards are short.

### Changed

- GPU requests start faster when cards are short.
- A connected robot holds its worker, billed, and when a new version goes live its next call raises `NewVersion`:
  `connect()` again (older SDKs reconnect by themselves).

### Removed

- `seq deploy --region`, `seq policy set <name> region=...` and the console's Region field are gone, and
  `@seq.policy(region=...)` no longer does anything.

## 0.56.7 — 2026-11-05

`seq key revoke`, `seq benchmark cancel` and `seq benchmark shell` now say what happens and when.

### Changed

- `seq key revoke` says the key opens nothing within 30 seconds, and a robot connected with it is cut off within a
  minute.
- `seq benchmark cancel --help` says its machines end, billed for what they ran until then.
- `seq benchmark shell` says it runs on the newest version alone and `--stop` names the benchmark, and `seq account`
  counts its machine in your GPUs.

## 0.56.6 — 2026-11-05

An idle worker now ends, so a robot that comes back after its idle timeout starts on a new machine, a cold start every
time.

## 0.56.5 — 2026-11-05

Planning is part of the policy now, in place of `@seq.planner`, and every policy and key is one kind, with no
`seq prod`.

Needs sequence-base 0.8.7 or later; `pip install -U sequence-ai` brings it.

### Changed

- To move a planner, write `def plan(self, robot)` under `@seq.plan` in its policy's class, read
  `episode.state.get("seq.plans")` in the act, and connect with `connect("<policy>")`.
- `seq key add` takes no `--production`, nothing needs promoting, and the keys and robot keys you hold go on working.
- A benchmark's image is built where it first runs, by its first eval or `seq benchmark shell` there, and billed as a
  build.

## 0.56.4 — 2026-11-04

An idle worker stops after a minute unless set, a benchmark's registration is priced before it registers, an eval the
platform stops before its episodes costs nothing, and `seq account` says what you may spend now.

Needs sequence-base 0.8.6 or later; `pip install -U sequence-ai` brings it.

### Changed

- `idle_timeout` is 60 seconds unless set, 2 seconds to 20 minutes, and on a version's first worker it counts from when
  its image is saved.
- `seq deploy` of a benchmark, and its `--dry-run`, say the hourly price of each machine registering it runs, before it
  registers.
- An eval the platform stopped before its episodes began costs nothing, as when no card was free within 10 minutes.

## 0.56.3 — 2026-11-04

A policy's build is ready sooner, a policy that fails to start tells its robots why, and how long an idle worker stays
up is a deployment's setting.

Needs sequence-base 0.8.5 or later; `pip install -U sequence-ai` brings it.

### Added

- A deployment's `idle_timeout` setting: how long a worker with no robot acting on it stays up before it stops.

### Changed

- A policy's build is ready sooner, as soon as its check passes.
- A policy that fails to start tells its robots why, with its traceback.

## 0.56.2 — 2026-11-03

`seq benchmark shell` runs one command on a machine of your benchmark, as `seq policy shell` does on a deployment.

### Added

- `seq benchmark shell NAME -- COMMAND` runs one command on a machine of your benchmark's newest version, billed as its
  check, and prints its output.

### Changed

- The console's billing card shows the credits added to your account and what is left of them.
- An eval's warning that its policy is slow no longer reads a batched policy as several times slower.

## 0.56.1 — 2026-11-03

Upgrade to 0.56.1, as the platform refuses older versions: usage past your monthly free credit is charged to your card
each month, and a policy's time limits are the platform's.

Needs sequence-base 0.8.4 or later; `pip install -U sequence-ai` brings it.

### Added

- At the start of each month, usage past your monthly free credit is charged to your default card, first on 1 December.
- `seq policy shell NAME -- COMMAND` runs one command in a worker serving your deployment and prints its output.

### Removed

- `seq.Timeouts`, `seq.Tensors`, `timeouts=` and `tensors=` on `@seq.policy`, and the settings `timeouts.request`,
  `timeouts.episode_idle`, `timeouts.startup`, `alerts.error_rate` and `alerts.p95_ms` are gone.

## 0.56.0 — 2026-11-03

Upgrade to 0.56.0, as the platform refuses older versions: a robot's worker starts near it when it connects, every
version is kept until you delete it, and billing is postpaid.

Needs sequence-base 0.8.3 or later; `pip install -U sequence-ai` brings it.

### Added

- `max_wait_for_batch=` on `@seq.policy` sets how long a robot may wait for its batch.

### Changed

- A `@seq.concurrent` policy's infer is handed its batch stacked, arrays along a new first axis and text as a list,
  unless it sets `collate=False`.

### Removed

- `@seq.reset`, `seq.collate`, `seq.uncollate`, `group_by=`, `seq policy freeze`, `seq benchmark freeze`,
  `seq.policy_freeze`, `seq.benchmark_freeze`, `freeze=`, `deps=`, `io=`, `family=`, `engine=`,
  `seq account delete --confirm`, the `seq deploy` flags `--freeze`, `--hours-per-day`, `--starts-per-day`, `--robots`,
  `--max-gpus` and `--timeout`, and the `seq eval run` flags `--policy-gpu` and `--timeout` are gone.

## 0.55.1 — 2026-11-02

What you deleted stays deleted: an eval, a list or a report says so when what it ran is gone, and `seq eval report
--records` saves each episode's record.

### Changed

- `seq eval report --records DIR` saves every episode's record - what its policy was sent, its cameras as video, and
  each answer - where `--videos` and `--trajectories` said every eval kept none: an eval keeps that one record, and
  those two flags are gone; `seq eval replay` plays an eval as the console's Replay does now, with its Plan line for an
  eval through a planner.
- A policy, benchmark or adapter deleted since an eval ran is said so beside its name - "(deleted)" - in `seq eval
  report`, `seq eval ls`, `seq benchmark evals`, `seq adapter evals` and `seq adapter ls`, and such an eval offers no
  `--rerun-failed`.
- `seq volume put` resumes a stopped upload only into the volume it began for: one deleted meanwhile, or deleted and
  made again, starts over.

## 0.55.0 — 2026-11-01

An eval can run through a planner, a delete is permanent, and every version of a policy or a benchmark is kept until you
delete it.

### Added

- `seq eval run <planner>` runs the policy its planner plans for, with the planner beside it on the same machine, and
  `--policy` picks one when the planner plans for more than one; the report says the plans it wrote and what its model
  calls cost. A benchmark may also run on a T4 or an A100.

### Changed

- `seq policy rm`, `seq benchmark rm`, `seq adapter rm` and `seq volume rm` delete at once and for good, and the name is
  free, so `--purge` and `seq storage purge` are gone; what a policy captured stays as a dataset until you delete it with
  `seq volume rm`. Every version of a policy or a benchmark is kept: `seq policy rm <name>@vN` deletes one, and
  `seq deploy --override` (in place of `--replace`) deletes the version it replaces once the new one serves.

### Fixed

- `add_local_python_source` no longer runs the module on your machine, so what it imports only needs to be in the
  image, and weights in named parts show every part with its size in `seq deploy --dry-run`, `seq policy ls` and
  `seq policy versions`.

## 0.54.2 — 2026-11-01

A robot no longer waits at an episode's end for a new version's machine, an act no longer waits behind your robot's own
busy threads, new machines are ready sooner, a stuck worker is replaced, and one storage limit you can raise yourself
covers everything you keep.

### Added

- `seq account set --storage-limit 500GB`, or your Account page, sets your own storage limit, any size, and you are told
  as you near it. `seq policy rm <name> --purge --yes` and `seq benchmark rm <name> --purge --yes` delete one for good now;
  `seq storage purge --yes`, or Delete on the console's Storage page, does all you deleted that is still kept. Otherwise
  `seq volume ls` says the day its files are removed.

### Changed

- A robot keeps its worker while a new version's machine starts, so `end_episode()` no longer waits for one. An eval's
  policy and benchmark share one machine when they fit, it bills no machine it gave no work, and a benchmark's image is
  built once as it is registered, so the eval never waits for a build. `seq eval run` and `seq eval status` say what an
  eval waits on before its first episode and about how long is left, and the dry run's time counts its machines' start,
  which is not billed. Each deploy's image save is now billed per GB sent, and `seq deploy --dry-run` says the price.

### Fixed

- `seq volume rm deps` deleted the rest of your storage with it, your evals' recordings among them: it now deletes only
  your version dependencies. `seq eval rm` now deletes the eval's recorded files as it says, and recordings whose files
  are gone stop being billed.

## 0.54.1 — 2026-10-31

A robot can connect through a planner, a new version's check machine stays on as its first worker, and deploys are
faster; 0.54.0 and older are refused, so upgrade with `pip install -U sequence-ai`.

### Added

- `sequence_ai.connect("<planner>", policy="<policy>")` connects a robot through a planner, which reads its prompts and
  tells its policy what to do; `act()` raises `WaitingForPlan` until the first plan.

### Changed

- A new version's check machine stays on as its first worker, so its first robot's acts are warm, and stops after its
  idle time with no robot connected.
- A version whose weights did not change reuses its earlier version's copy and saves only what changed in its image,
  and `weights="hf://org/repo@<rev>/<path>"` copies only that folder or file.

## 0.54.0 — 2026-10-30

A robot now talks to its policy over one stream only, and `seq deploy --dry-run` tells you when a policy may not fit
in the memory it asks for.

### Changed

- A robot's observations, acts and episode ends all go to its worker over one stream; the request-by-request path is
  gone. When something between the robot and the platform blocks WebSocket, `connect()` says so and what to allow.
  An older SDK is refused with a message to upgrade, and a policy deployed before this release is told to deploy again.

### Added

- `seq deploy --dry-run` warns when a policy's `memory=` is smaller than its weights, before the deploy rather than at
  load.

## 0.53.1 — 2026-10-30

Stopping a robot or taking it over now covers the act on its way, an eval's report says when its machines were
reclaimed, and the messages about your robot say what actually happened.

### Added

- `seq eval report` says how many of an eval's machines were reclaimed mid-run and started again, and how many episodes
  ran again; the lost part is not charged.

### Fixed

- A robot stopped or taken over while an act is on its way is handed its hold or the person's chunk, never the policy's
  answer, and `on_resume` waits until the arm has something to play; a console also reaches a robot that starts
  observing right after `connect(teleop=True)`.
- A robot's notices and `seq robot status` say what happened: no notice when it gave its seat back itself to connect
  again or in `wait_until_ready()`, whose next poll seats it again; no lost keepalive blamed while the platform hears
  it; a far robot told to move hears why, not that its worker is leaving; and the stretches with nothing to play that
  already ended are counted apart from the one under way.

## 0.53.0 — 2026-10-30

The platform now hears your robot between acts, tells your code the moment the arm has nothing left to play, and shows
how each robot's connection is doing.

### Added

- A Policy sends its worker a heartbeat from its first `observe()` or `act()` on, so the platform can tell a robot that
  is alive from one whose loop is stuck; `connect(on_stale=, on_resume=)` calls your code when the arm runs out of
  valid answers to play and again when it has one.
- `seq robot status` shows each connection's link - its round trip and upload time as its worker measured them - and
  `seq policy status` its robots' connects of the last day: how many, how fast to the first answer, and where the
  others stopped.
- `seq models` shows each model's context and the inputs it takes, `seq eval ls --adapter NAME[@vN] --status STATE`
  lists an adapter's evals in one state, and the `seq.*_logs` functions take `since=0` as the log's start.

## 0.52.1 — 2026-10-29

The CLI shows times in your account's time zone and clock, says what the platform did when you sign in or out, and
checks that a robot key's limit and expiry took.

### Fixed

- `seq account set --time-zone/--clock` sets the times every `seq` command prints, not only the console's; `seq logout`
  and a new `seq login` end your old sign-in on the platform or say they could not, and the sign-in and payment links
  print before a browser opens - none opens over SSH.
- `seq robot add --limit/--expires` checks the key came back with them, `seq robot set --limit 0` is refused (`none`
  lifts a limit), `seq billing top-up` waits for its own payment, and a command the platform does not take yet says so
  rather than "Method Not Allowed".
- A robot already connected gets its standby within a minute once its deployment keeps a spare worker, and
  `wait_until_ready()` waits on through a ready poll the platform did not answer in time; `seq` writes each line as it
  prints it in a script or a pipe, and a chat answer spent on reasoning alone warns why it is empty; `seq policy ls`
  shows dev or production, `seq account` its id and `seq eval ls --adapter NAME@vN` one version's evals; and the
  `seq.*_logs` functions take a time as epoch seconds or a datetime.

## 0.52.0 — 2026-10-28

Sign in with `seq login` and do from the command line what the console's pages do; an eval's replays open in a 3D
player on your own machine.

### Added

- `seq login`, `seq key`, `seq billing` and `seq account` do what the console's Keys, Billing and Account pages do - a
  promotion code too, with `seq billing redeem CODE`; a robot key takes an expiry and a spending limit (`seq robot add
  --expires/--limit`, `seq robot set`), `seq robot status` and `seq planner ls|status` show what the Machines page
  shows, and every log pages back with `--before` - each with its `seq.*` function.
- `seq eval replay EVAL_ID|FILE` opens an eval's replays in the console's 3D player on your own machine - turn, zoom
  and scrub - and `seq.read_replay(path)` reads a replay file as arrays.

### Fixed

- A robot whose worker is lost goes on with its episode on its standby; `seq policy status` says why a worker of your
  warm hours, your minimum or a spare runs and stopped, `seq policy rm` names each key it stops by its kind, and
  `seq --json eval cancel` prints the platform's answer.

## 0.51.11 — 2026-10-28

A robot says where its worker runs and why, and `seq usage` and `seq eval report` show how each robot's link went.

### Added

- When a robot connects or moves to another worker, it says about how far away that worker is when it is far;
  `seq policy status` says under each robot why it sits on its worker.
- `seq usage --policy` lists each robot by the key it connects with - its acts, errors, round trips and on-time - and
  `seq eval report` shows the link between the benchmark's and the policy's machines.

### Fixed

- A run a robot leaves open when its process exits still ends, so `seq usage` shows its on-time; a read told to wait
  (429) tries again as asked, up to three times; and timing reports go every 5 s, so they no longer use up the key's
  requests per minute.

## 0.51.10 — 2026-10-28

A teleoperated robot's audit says who drove it.

### Fixed

- `seq audit` names the operator of a teleop session beside its robot key, and `sequence_ai.teleop()` sends its
  operator when it connects.
- `sequence_ai.teleop()` needs no `robot=`: the key already names its robot.

## 0.51.9 — 2026-10-27

`seq policy status` counts the workers running or starting, and says when a spare worker was stopped at no cost to you.

### Fixed

- `seq policy status`'s fleet line counts only the workers running or starting; a stopped one is still listed below
  with why it stopped.
- A second worker started for the same place is stopped and not billed, and `seq policy status` says so.

## 0.51.8 — 2026-10-27

A robot that reconnects keeps its episode, and robots move to a new version sooner.

### Fixed

- A robot that reconnects to the worker it was on keeps its episode: its acts no longer stop for up to a minute after
  its connection drops.
- A robot moves to a new version, or back to an earlier one in a rollback, about two seconds sooner: as soon as its
  worker says it is leaving.
- `seq robot revoke <id>` names the revoked key by its own last four characters, and `help(seq.planner)` shows the
  sample a planner's build checks.

## 0.51.7 — 2026-10-27

A person can watch a robot and drive it from a console of their own, and you can see, download and delete what you store.

### Added

- `sequence_ai.teleop(policy, robot=..., key=...)` watches a robot's cameras, takes it over, sends it actions, stops it
  and hands it back to its policy; a robot that `connect(teleop=True)` takes them, and `seq.ask_human(reason=...)` lets
  a planner ask a person for help.
- `seq storage` and `seq volume files`, `get` and `rm`: what you store against your plan's limit (Starter 100 GB, Team
  2 TB), and each file to download or delete.

### Fixed

- `seq policy data export --format lerobot` writes a dataset LeRobot reads and trains on: at the policy's hz, with
  LeRobot's column names, each frame's task, and who drove it.

## 0.51.6 — 2026-10-27

A robot's acts and observations share one connection, a policy can run in a region you choose, and robots reach a new version within seconds.

### Added

- `@seq.policy(region=...)` and `seq deploy --region`: `us-west` or `us-east` starts a policy's workers in that
  region alone, a robot waiting while no card there is free; `auto`, the default, starts each robot's worker on the
  closest GPU available.

### Changed

- `seq secret rm` refuses a secret a deployment still uses, naming the deployment; `--force` deletes it, and that
  deployment's workers stop serving. A secret set again reaches every worker, the ones that start later too.

### Fixed

- A `@seq.policy`, `@seq.benchmark` or `@seq.adapter` class written in a script run as `python file.py`, in a
  notebook or inside a function runs there as it did before the decorator; `seq deploy` still refuses one it cannot
  import, saying why.

## 0.51.5 — 2026-10-27

Where a class keeps its episodes follows what its methods take, as the docs say.

### Changed

- A class whose `@seq.observe` and `@seq.infer` methods both take `episode` keeps its episodes in `episode.state`, as
  the docs' example does: `seq deploy` says they are handed to observe and infer, no longer kept on the class.
- A class with an `@seq.reset` method keeps its episodes on itself even when its infer takes `episode`: a reset is
  handed nothing, so what it clears is the class's own.

## 0.51.4 — 2026-10-27

A policy says how its robots play its answers with two numbers, and a copy's robots fall into one batch when there is
time to wait for them.

### Changed

- `@seq.policy(action_chunk_valid_len=N, action_chunk_replan_at=k)` in place of `action_chunk_len` and
  `action_chunk_exec_len`: an answer's first N steps are valid, and its robot asks for the next one k steps after it sent
  the act, playing the rest meanwhile; the old names are refused saying the new ones. A copy seats more than one robot
  only when k is below N, and `seq deploy` says how many from the time between them.
- `--latency-ms` and `latency_ms=` are gone: an eval asks for the next answer at step k and switches to it at step N,
  the world held still until it comes.
- Requires sequence-base 0.7.0 or later.

## 0.51.3 — 2026-10-26

A planner is a class of its own that plans for the policies and planners it names, teams and `observations=` are gone,
and a robot's observations keep moving on a slow or idle line.

### Added

- `@seq.planner(members=[...])` with one `@seq.plan` method, `def plan(self, group)`, deployed with `seq deploy` as a
  policy is; `sequence_ai.planner(name).observe(...)` for what only a planner reads, such as a site's iPad; and
  `seq planner key add <planner> --member <name>` (and `list`, `revoke`): one key for each member of a planner, a
  deployed member's held by the platform.

### Changed

- `@seq.team`, `sequence_ai.team(...)`, `seq team key`, `@seq.plan` on a policy, and `@seq.policy(observations=...)`
  with `seq.Observations` and `seq.Stream` are gone; each says what to write instead. Deploying needs sequence-ai
  0.51.3; a robot of an older one connects and acts as before. `seq robot add | list | revoke` name a robot key by its
  policy's tier: a robot develop key or a robot production key.

### Fixed

- A big observation on a slow line is no longer cut and sent again, a robot's connection stays warm while it acts, the observations' connection
  opens on the first observe() so the acts after it find it open, an observation's arrays go compressed when that
  saves time, and an act that had to send its observations again says how long it waited. On Linux, the first large
  observation says once how to keep the connection's window after a pause. A CPU worker's estimate says it is billed
  from when it starts, a build error in a list names its step once, a build check that called no act says only loading
  was measured, and an observation past 1 MB points to `sequence_ai.Video`.

## 0.51.2 — 2026-10-26

`seq benchmark ls` shows the driver a benchmark runs with, an eval adds no latency unless you set one, and a robot takes
its spare worker sooner.

### Added

- `seq benchmark ls` and `seq benchmark status` show the driver version of a benchmark that names one, as
  "L4 · driver 580.178.04".

### Changed

- An eval's latency is 0 unless you set `--latency-ms` or `latency_ms=`; the platform estimates none.

### Fixed

- A robot whose policy keeps spare workers takes one soon after it is ready, where it could wait over a minute. An
  official template's `seq deploy` no longer says its lock is not read, a team's says its version, template notes read
  as sentences, a policy that does not batch is no longer said to serve "0 batch(es)", and `seq deploy --dry-run` says
  "kept on one copy" where it said "none kept" of an episode held on one copy.

## 0.51.1 — 2026-10-25

A team plans for the robots of several policies at once, a class builds itself in its constructor, and a robot's video
plays on through a lost frame.

Needs sequence-base 0.5.1 or later; `pip install -U sequence-ai` brings it.

### Added

- Teams: `@seq.team(policies=[...])` with one `@seq.plan` plans for the robots of several policies, a group of them at a
  time; a robot joins a group with `sequence_ai.team(name, group=...)`, and `seq team key add | list | revoke` issues the
  team key a site's iPad observes its group with.

### Changed

- `@seq.load` and `@seq.setup` are removed: a policy loads its model and a benchmark builds its worlds in `__init__`, and
  `init=` hands it what the platform fills in - `"{weights}"`, `"{num_worlds}"`, `"{policy.hz}"`. Each refusal writes the
  method as the constructor it becomes; a benchmark registered with `@seq.setup` is registered again with `seq deploy`.
- A robot's video reaches its policy sooner, several frames on their way at once, and a frame lost on the way no longer
  ends its episode: the policy reads on from the stream's next keyframe. A robot whose policy keeps a spare worker gets
  it as soon as one is up.

## 0.51.0 — 2026-10-25

A policy declares its control loop and the batch sizes it serves, a benchmark is the environment as it is, a
deployment's settings change without a redeploy and `seq deploy --dry-run --robots N` prices N robots at once, a
robot whose copy of the policy stops answering goes on at once, its episode with it, on a spare copy the deployment
keeps (`buffer_workers`), each robot
connects with a key of its own, and a robot names nothing of the
policy's.

Needs sequence-base 0.5.0 or later; `pip install -U sequence-ai` brings it.

### Changed

- `@seq.policy(hz=..., action_chunk_len=..., action_chunk_exec_len=...)`: the control rate, and how many steps an answer
  holds and how many of them play, are the policy's to declare - `hz` to deploy - and `@seq.concurrent(sizes=[1, 2, 4])`
  lists the batch sizes its act function is called with; `max_concurrent=` and `dynamic=` are refused. A benchmark is
  `@seq.benchmark(tasks=..., seeds=..., max_seconds=..., driver=...)` with `@seq.setup`, `@seq.reset` and `@seq.step`,
  the environment as it is; one registered before this release is registered again with `seq deploy`.

### Added

- `seq policy settings <name>` and `seq policy set <name> KEY=VALUE ...`: every setting of a deployment, where its value
  comes from and when a change takes effect, changed at once, all of them or none; `seq policy scale`, `params`,
  `budget`, `alerts` and `capture` are settings now, each refused in words that give the new form
  (`seq policy set <name> max_gpus=4`, `params.<name>=VALUE`, `budget.monthly_usd=USD`, `alerts.webhook=URL`,
  `capture=on`). `seq robot add --policy <name>` gives a robot a key of its own, printed once (`seq robot list`,
  `seq robot revoke <id or last four>`); `seq prod promote <name>`, run with a production key, makes a dev policy a
  production one; `seq policy fork <name> <name>-next` copies a policy - its code, settings and weights - into one of
  its own, the way to try a version first now that `seq policy traffic` is refused. From Python:
  `seq.policy_settings()`, `seq.policy_set()`, `seq.robot_add()`, `seq.robot_list()`, `seq.robot_revoke()`,
  `seq.prod_promote()` and `seq.policy_fork()`; `sequence_ai.scale()` points to `seq.policy_set()`.

### Removed

- The robot-arm preset, and every keyword a robot gave of the policy's: `sequence_ai.observation()`, `Observation`,
  `ImageFrame`, `Proprioception`, `ActionChunk`, `policy.prepare_image()`, `to_command()`, `to_commands()`,
  `policy.contract`, `policy.set_instruction()`, `policy.instruction`, binary frames (`policy.act_format`,
  `SEQ_ACT_FRAMES`), `act(steps_left=, action_horizon=, return_world_prediction=, robot=, params=)` and
  `connect(control_hz=, params=, validate=, expect=)`. A robot hands what it observes, its instruction too, to
  `policy.observe(x)` as it is, `act()` returns what the policy's act function returned, and
  `sequence_ai.resize_image()` fits a frame; each removed name is refused saying what to write instead. A robot on
  0.50.0 or earlier that connects with `expect=` is refused by its own SDK before its first act: upgrade to 0.51.0 and
  leave `expect=` out.

## 0.50.0 — 2026-10-24

A policy's planner runs beside its act function, a robot's episode keeps its last minute, and capture is one switch that
keeps every episode whole.

### Added

- `episode.state["seq.observations"]` and `episode.state["seq.actions"]`: each observation a robot sent, at its step,
  and each return of the act function, at the step of the newest observation before it, kept in the robot's episode for
  about its last minute.

### Changed

- `@seq.plan(every_s=, wake_on=)` marks `def plan(self, episode)`, plain or async: it is called every `every_s` seconds
  after a robot's last plan began, once the robot has sent something new, and at once for an observation that holds a
  key `wake_on` lists. What it returns is kept in `episode.state["seq.plans"]`; the act function takes `episode` and
  reads the newest as `episode.state["seq.plans"][-1]["value"]`. `plan=`, `robot=`, `max_concurrency=` and
  `max_batch=` are gone, refused in words that say what to write instead.
- `seq policy capture <name> on | off`: on keeps every episode whole, its frames too, and `seq policy data export`
  includes what the policy kept in `episode.state` at each episode's end; `--rate`, `--images` and `--off` are gone,
  refused in words that give the new form.

## 0.49.0 — 2026-10-24

A robot plays its actions in a loop of its own, and a policy marks its methods by decorator alone: `run()`, the `Player`
and the method names on `@seq.policy` are gone.

### Removed

- `policy.run()`, `sequence_ai.Player`, `RunOutcome` and `Answer`: `act()` returns what the policy's act function
  returned as it returned it; a robot-arm policy's `act()` sends the newest observation `policy.observe(obs)` kept and
  returns its `ActionChunk`, and `act(obs)` is refused.
- `infer=`, `observe=`, `reset=`, `end=`, `sample=` and `make=` on `@seq.policy`: mark the methods `@seq.infer`,
  `@seq.observe`, `@seq.reset`, `@seq.on_episode` and `@seq.sample` - for a class of your repository, in a subclass of
  it - and build what `make=` built in a class's constructor; `seq deploy` writes the subclass out for you. A method
  takes `episode` where it took `episode_id` (`episode.id`), and `@seq.on_episode`'s is `(self, episode)`.
- `connect(embodiment=)`, `@seq.policy(embodiments=)` and execution feedback (`act(feedback=)`, `seq.Feedback`): what a
  robot is, and what it was playing, go in its observations.

## 0.48.1 — 2026-10-24

`Video`, `Depth` and `Tactile` reach a class that names its `observe=` method, and `seq deploy --dry-run` answers as the
deploy would in two more cases.

### Fixed

- `policy.observe()` of a `sequence_ai.Video`, `Depth` or `Tactile` item works for a policy that names its `observe=`
  method or declares `observations=seq.Observations(every=1)`; before, it was refused. A policy that reads some steps
  and not others still refuses them, saying why.
- `seq deploy --dry-run` under the name of a deployment you just deleted ends as the deploy would, with a new deployment,
  instead of failing after saying the platform accepted it.
- `init=` naming a part that `weights=` does not have, as `"{weights.nope}"`, is refused by `seq deploy` and its dry run,
  naming the part and the parts there are; before, the build failed on the platform.

## 0.48.0 — 2026-10-24

Every input goes through `observe()` and `act()` carries none, and a robot can send its camera, depth and touch data
compressed.

### Added

- `sequence_ai.Video`, `Depth` and `Tactile`: wrap an item of what you observe, and it travels compressed and reaches
  your policy as it was sent (`pip install "sequence-ai[observe]"`, Python 3.10 or later).
- A sample episode (`sample=`) is the robot's own calls in order, `("observe", x)` and `("act",)`: `seq deploy` plays it
  on the card and says how long the first act and the ones after took, and the peak GPU memory.

### Changed

- For a policy that names its methods, or one written with `@seq.infer` that takes one act at a time and plans nothing,
  `policy.act()` carries no data: hand inputs to `policy.observe(x)` first. With no `episode_id` the calls go under the
  handle's current episode, and what the act function returns comes back as it is, arrays with their dtype and shape.
  A policy deployed before this takes `act(x)` until `seq deploy` builds it again.

## 0.47.0 — 2026-10-24

A class you already have deploys as it is: its factory function, its methods' named inputs and its torch tensors.

### Added

- `make="package.module:function"` names the function that builds a policy, and `seq.policy(...)(Cls)` in a file of
  your own declares a class imported from elsewhere, so the class's own files stay as they are.
- A named method that takes several required inputs is handed an act's dict by key; a missing or extra key is refused,
  naming it.

### Changed

- An answer's torch tensors reach the robot as numpy arrays of the same dtype (bfloat16 and float8 as float32), and
  `seq deploy` warns when `infer=` names a method that answers one action a step, such as `select_action`.

## 0.46.0 — 2026-10-24

`seq deploy` says what a policy's declarations make of it before anything is built, and installs a project's named
requirements file.

### Added

- `seq deploy` and its `--dry-run` say what a policy reads of each observation, how its answers play, whose its
  episodes are, what its build check acts on and how many robots a copy seats - and, once, what to add when a policy
  that could seat several robots a copy seats one because nothing times it: `sample=`, or an observation contract.
- `seq.Image.from_project(requirements="requirements-gpu.txt")` names the requirements file a project installs from.

### Changed

- `seq deploy` installs a folder's one `requirements-<name>.txt` when it has no other dependency file, and of several
  asks you to name one.

## 0.45.1 — 2026-10-24

A policy that declares `observations=` is handed the same inputs on a robot as in an eval, and `seq deploy` says how long
its `sample=` episode took.

### Fixed

- A policy that declares `observations=` and reads each act's own step alone - `seq.Observations({"head":
  seq.Stream(...)})` - is handed `{"observations": [...], "steps": [...]}` on a robot, with only the keys it declares,
  as an eval hands it; before, a robot's act reached `infer` as it was sent, every key in it. A robot on an older
  `sequence-ai` gets the same, its acts' steps counted 0, 1, 2, ... each episode; a policy deployed before this gets it
  once `seq deploy` builds it again.
- `seq deploy` says how long a policy's `sample=` episode took, where it said no act was called, and the platform's
  hints no longer ask for a `sample=` the policy has.
- A policy's own inputs that hold numpy scalars - a `bool_`, a complex, a `timedelta64`, a `longdouble` - reach it as
  such instead of failing the act.

## 0.45.0 — 2026-10-24

A policy's own inputs keep their numpy arrays' dtype and shape, and its `observe=` method can be handed each frame's step.

### Added

- `@seq.policy(observe="...")` naming a method that takes `step` is handed each frame's step in its episode, from 0 each
  episode; `seq deploy` checks the method takes it after the observation or by name.

### Fixed

- A policy's own inputs that hold numpy arrays reach it with each array's dtype and shape, as bytes do, and an answer's
  arrays come back as arrays - before, `act()` could not send them; to a policy deployed before this, arrays go as JSON
  lists, with one warning.
- A policy that names its `observe=` method and declares a key read every few steps, or a window with gaps, has its acts
  served instead of refused; after the copy holding an episode stops, the new episode counts its steps from 0 and is
  handed none of the old one's observations.

## 0.44.2 — 2026-10-24

`observe()` as a robot's first call waits for a worker that is starting, as `act()` does.

### Fixed

- `policy.observe(...)` made before any act - the call that connects - waits out a worker that is starting, or a place
  in line, within `startup_timeout_s` (0 still raises at once) instead of raising `Unavailable`; once connected, a frame
  never waits.
- `policy.observe(...)` while its worker restarts mid-episode keeps the observation for the episode's next act, which
  waits for the worker and sends it, instead of raising `Unavailable`.

## 0.44.1 — 2026-10-24

A policy of its own inputs plays as its `seq.Actions` declares, with no `actions_of=`.

### Fixed

- `run()` and a `Player` play the rows each answer's `seq.Actions(key=...)` names, up to its `open_loop_horizon`, at
  its `hz` - else the default: a dict or a list of numbers is one action, a list of rows of one length plays row by
  row; `control_hz=` and `absolute=` no longer need `actions_of=`.
- `answer.rows` and `answer.spare` read such an answer's rows the same way, and a `Player` plays one you hand it.

## 0.44.0 — 2026-10-24

`seq eval report` says what an eval's own measurements say is worth declaring.

### Added

- `seq eval report` prints the platform's hints for an eval as `hint:` lines under the policy's load.
- A run's report says the step its chunks take within one and how long one plays, so the platform's hints about a
  jump between two chunks and about chunks running out are said only where they apply.

### Fixed

- A `Player` on a policy that reads steps before an act's own no longer sends, near its run's end, a frame that no act
  reads.

## 0.43.0 — 2026-10-24

`seq policy status` and `seq deploy` say what a deployment's own measurements say is worth adding to it next, under the
figure each comes from.

### Added

- `seq policy status` prints each of the platform's hints as a `hint:` line under the figure it comes from - batching,
  the build check, the robots waiting - and an `on time:` line with the stalls, its hints under it; `seq deploy` prints
  its build's after its check.
- A run's report says the settings it played its chunks by (`lookahead`, `smooth_seam`, `max_age_s`, `aligned`), so a
  hint never asks for one a run had set.

### Changed

- A `Player` on a policy that reads steps before an act's own sends each of those frames once, as it is made, also
  when its acts come at a changing spacing.

## 0.42.0 — 2026-10-23

A policy can be handed what the robot was playing when it sent each act, and a robot plays a chunk that starts at its
observation's moment from the row that is now.

### Added

- An `@seq.infer` that takes `feedback: seq.Feedback` is handed, with each act, the rows the robot had not played,
  the steps it expects to play before the answer is back, how many it had played and the last one; a `Player`,
  `run()` and an eval send it, and `act(obs, feedback=...)` sends your own.
- `aligned=True` on `seq.ActionContract` or `seq.Actions`: a `Player` and `run()` take each answer as it lands and
  play it from the row that is now, the rows that went by while it was inferred skipped, and `Player(policy,
  aligned=...)` and `run(aligned=...)` say it for one loop.

## 0.41.0 — 2026-10-23

A policy can say what it reads and answers, a robot sends only the steps its acts read, and a policy that keeps its
episodes on the platform can run its acts together.

### Changed

- `@seq.policy(observations=, actions=, tensors=)`, with `seq.Observations`, `seq.Stream`, `seq.Actions` and
  `seq.Tensors`, replace `own_observations=` and `own_actions=`: `observations=seq.Observations(every=1)` takes every
  observation since the last act, and `actions=seq.Actions()` plays an answer a row a step.
- A policy whose observations read a window of past steps (`at=`) or keys fed every few steps (`every=`) is handed
  those steps by its worker, and `policy.observe()`, `run()` and a `Player` send only the steps its acts read.
- `@seq.concurrent` beside an infer that takes `episode` hands it `episode: list[seq.Episode]`, one for each
  observation; when a batch raises, acts at their episode's start run again alone and the other episodes end, and a
  robot on 0.41.0 goes on in a new one.

## 0.40.0 — 2026-10-23

The control loop comes apart into `run()`, a `Player` for a loop of your own and `act()` for one request, and a policy
can keep each robot's episode apart and batch its acts.

### Changed

- `predict()`, `next_action()` and `act("instruction", observe=, act=)` are gone: write `act(obs)`, which returns an
  `Answer`, `sequence_ai.Player(policy)` for a loop of your own, or `run(observe, step, instruction=...)`, which now
  needs `max_seconds=` or `max_actions=`.
- `@seq.policy(streaming=)` is gone, worked out from how the policy is written: an `@seq.infer` that takes `episode`
  is handed that episode's `seq.Episode`, so one copy keeps several robots' episodes apart, and
  `policy.observe(obs, episode_id=...)` sends each observation as the robot makes it.
- `seq.collate` and `seq.uncollate` turn a `@seq.concurrent` policy's acts into one batch and its answer back, and
  when a batch raises, only an act that fails on its own gets an error.

## 0.39.0 — 2026-10-23

`seq deploy` takes a folder and makes its image from the project's own dependencies, a robot goes on in a new
episode when its policy's copy stops, and an eval's estimate says what it will likely cost.

### Changed

- `seq deploy [folder]` (`.` by default) deploys the folder's one `@seq.policy`, `@seq.benchmark` or `@seq.adapter`
  class (`--name` picks one of several) and replaces `seq deploy <file>.py`, `-m` and `--entry-only`; a `.env` stays
  on your machine unless a `!.env` line uploads it.
- A policy that writes neither `image=` nor `deps=` gets an image made from its project's own dependency files.
- When the copy of the policy holding an episode stops, the robot goes on in a new episode with a warning;
  `policy.end_episode()` ends an open episode, and `close()` ends one left open.

## 0.38.17 — 2026-10-21

Freeze a version to rebuild it from your own storage, and deploy a class that runs locally by naming its own methods.

### Added

- `@seq.policy(freeze=True)`, `seq deploy --freeze` and `seq policy freeze NAME[@vN] [--off]` (and the benchmark
  ones) keep a version's base and what its build installed in your storage, at $0.12/GB-month, so it can be built
  again exactly as it is.
- `@seq.policy` can name a class's own methods (`infer=`, `observe=`, `reset=`, `end=`) and its constructor's
  arguments (`init=`, `make=`, `seq.config`): a class that runs locally deploys with one decorator.

### Changed

- `connect()` and `wait_until_ready()` wait as long as the platform holds the start (1800 s when it says nothing) and
  say how far it got; a start whose machine stops while it loads ends at once, unbilled.

## 0.38.16 — 2026-10-21

Framework shortcuts are gone, a version no longer keeps a copy of what it depends on in your storage, and a worker's
start, its warmup included, runs within your own startup time.

### Removed

- `@seq.xpolicylab`, `seq port scaffold --robodojo` and `seq port scaffold --isaaclab`: wrap a policy class you already
  have with a plain `@seq.policy` instead.

### Changed

- A version no longer keeps a copy of what it depends on in your storage (0.38.15 did); your image can install a
  package too large to upload from where it is published, by its URL.
- A worker's start, its `@seq.warmup` included, has your `seq.Timeouts(startup=...)` (1800 s unless set) and ends at
  once when `@seq.load` raises; start time past what the default allows is billed.

## 0.38.15 — 2026-10-21

A version keeps what it depends on in your own storage, `seq port scaffold` writes a benchmark from your RoboDojo or
Isaac Lab checkout, and a policy of its own inputs and outputs is evaluated the way it declares.

### Added

- Each version of a deployment or benchmark keeps its own copy of what it depends on in your storage, so it can be
  rebuilt as it was; `seq policy versions` and `seq benchmark versions` say whether the copy is being kept or kept whole,
  and `seq volume rm deps` asks for `--yes` while versions hold copies there.
- `seq port scaffold --robodojo` (with `--envs N` for a batch) and `seq port scaffold --isaaclab <task>` write a benchmark
  folder from your own checkout; `@seq.benchmark(reset_together=True)` resets a batch of environments at once.

### Fixed

- `own_observations` and `own_actions`, and the ones `@seq.xpolicylab` declares for you, now reach your evals (0.38.14 did
  not send them); a client without numpy reads an answer's arrays as lists; a robot that waits past its time is told
  whether its GPU is held or still awaited; tables keep their columns apart.

## 0.38.14 — 2026-10-21

A policy of its own inputs and outputs takes bytes and arrays and is evaluated the way its own eval ran it, and an
XPolicyLab policy deploys as it is.

### Added

- `@seq.xpolicylab("XPolicyLab/policy/<name>", ...)` deploys a policy from an XPolicyLab checkout, with no code of your
  own around it.
- Inputs of a policy's own may hold bytes and numpy arrays (they travel as msgpack), and `@seq.policy(own_observations=,
  own_actions=)` has an eval send every observation since the last act and play the answer one action a step.

### Fixed

- An act waits as long as its deployment's request timeout (it waited 30 s), is not sent again while its worker still
  runs it, and a busy worker or a briefly busy service is retried instead of failing the call.

## 0.38.13 — 2026-10-21

A benchmark's Dockerfile builds with its context, a subclass keeps its parent's hooks, and `seq volume put` sends only
what changed.

### Added

- `from_registry` and `from_dockerfile` take `clear_entrypoint=True`, which drops the base image's `ENTRYPOINT`; and
  `seq deploy --dry-run` says how long the name's last version took to start and what its check cost.

### Fixed

- A benchmark's Dockerfile context ships with its registration, so its `COPY` finds what it names; a subclass of a
  benchmark or a policy has the hooks of the classes it is built on; and a class whose declaration is refused no longer
  stops the other classes of its file from deploying.
- `seq volume put` run again sends only the files that changed, and says the volume is already up to date when none did.

## 0.38.12 — 2026-10-21

`seq deploy --dry-run` checks a benchmark or an adapter on the platform, as it already checked a policy.

### Fixed

- A benchmark's or an adapter's dry run is checked by the platform as registering it would be: a missing assets
  volume, a name another class holds or a limit is said before anything is built, with the same error and exit code.
- `seq.Image.from_dockerfile` reads its Dockerfile and context next to the file that declares them, as `add_local_file`
  and `add_local_dir` already did, so `seq deploy <folder>/benchmark.py` works from any directory.

## 0.38.11 — 2026-10-21

`seq policy rm` and `seq policy storage` show small sizes as they are.

### Fixed

- Sizes under a gigabyte show to a tenth of a megabyte, as `seq volume ls` shows them: a 2.6 MB compile cache no longer
  reads "0 MB".

## 0.38.10 — 2026-10-20

`seq deploy` shows every line a build step prints, including the ones that arrive after the step ended.

### Fixed

- Lines a step printed late, such as what `@seq.load` printed, now show in `seq deploy`'s live view under
  "load, continued" instead of being left out.

## 0.38.9 — 2026-10-20

`seq policy ls` and `seq policy status` say when deploying the same code again brings a platform update.

### Changed

- A policy whose serving version was built before a platform update that reaches it only by a new build (on 2026-10-20,
  faster acts) is told so in `seq policy ls` and `seq policy status`, with the `seq deploy` command that brings it.

## 0.38.8 — 2026-10-20

`seq policy versions` and `seq benchmark versions` say more about each build.

### Changed

- `seq policy versions` says what changed in an image it rebuilt: its base image, Python, packages, environment
  variable names, commands or files.
- `seq benchmark versions` shows a failed build that a newer registration replaced, for a while, and the version that
  replaced it.

## 0.38.7 — 2026-10-20

`seq policy versions` and `seq policy logs` say more about each build.

### Changed

- `seq policy versions` shows a failed build that a newer deploy replaced, for a while, with the step it stopped in and
  the version that replaced it.
- In `seq policy logs`, a build's check machine points to that build: `seq policy logs <policy>@vN --build`.
- Every command's `--help` description starts with a capital letter.

## 0.38.6 — 2026-10-20

Every command's help says what it does, `seq volume` takes the names it lists, and a deploy and its logs say more.

### Fixed

- `seq volume show` and `seq volume rm` take a volume by the name `seq volume ls` prints; a policy's own copy of its
  weights is deleted with the policy, by `seq policy rm`.

### Changed

- Every command's `--help` starts with what the command does, and `seq policy rm` says "delete its version vN" where it
  said "take its version vN". Needs sequence-base 0.4.21.
- `seq deploy` says how long each build step usually takes and how many robots a worker seats, and `seq policy logs`
  says when a machine's bill ended on your own clock.

## 0.38.5 — 2026-10-20

Every command that reads now rides out a connection that fails, and `seq deploy` follows a build more briefly.

### Fixed

- `seq models`, `seq gpus`, `seq account` and every other command that reads now also ask again when the connection
  fails.

### Changed

- `seq deploy` no longer lists each uploaded file again while it follows the build, since it listed them before the
  upload; `seq policy logs --build` still shows every line. Needs sequence-base 0.4.20.
- `seq eval status` on an eval waiting for compute to free up says when it is expected back, or that the time is not
  known yet.

## 0.38.4 — 2026-10-20

Old argument names and the eval schedule are gone, and `seq policy rm` says first what goes with a policy.

### Removed

- `seq eval schedule` is gone, and `min_containers=`, `buffer_containers=`, `scaledown_window=`, `max_containers=` and
  `region=` are no longer arguments: write `min_workers=`, `buffer_workers=`, `idle_timeout=` and `max_gpus=`.

### Added

- `seq policy rm` lists the workers, robot keys, versions and storage it deletes before it deletes them (`--dry-run`
  lists them alone), and a deleted policy's captured data stays as a dataset that `seq policy data export` downloads.
- `seq policy logs` starts with the machines it covers and what an hour of each costs, `seq eval run --dry-run` splits
  its estimate into GPU, CPU and memory, `seq volume show <name>` says what uses a volume, and
  `seq benchmark source <name>[@vN]` downloads the code a benchmark version runs.

## 0.38.3 — 2026-10-20

A benchmark can load what all its evals share in `@seq.load`, apart from each eval's own `@seq.setup`.

### Added

- `@seq.load` on a benchmark takes only `self` and runs once on each machine, before `@seq.setup`; it opens no camera
  or window. Needs sequence-base 0.4.19.

## 0.38.2 — 2026-10-20

Commands that read, and a robot's first connection to its policy, ride out a connection that fails.

### Changed

- Every `seq` command that only reads asks again, up to three tries, when the connection to the platform fails or does
  not answer, and says what failed when it gives up; commands that change something are sent once. `seq policy logs
  --since <time>` reads back to that time, up to 5,000 lines.
- A robot opens its first connection to its policy in up to three tries, each on a new connection; when all fail, the
  warning says how many tries were made and the step the last one stopped in.
- `seq eval report` no longer prints the adapter's hash on its `ran:` line.

## 0.38.1 — 2026-10-19

Commands say more plainly what an object is made from, what an eval will cost and why something was refused.

### Changed

- `seq eval run --dry-run` adds a composite policy's planner cost and estimates a stopped policy too; `seq eval
  episodes` shows each metric in its own unit; `seq gpus` states the 60-second minimum.
- `seq benchmark upgrade`, `versions`, `status` and `cancel` say what changed in a sentence, that evals run the version
  that last passed its check until the new one passes, and a canceled registration as canceled; a folder you edited
  after `seq init` shows as the template it came from, "toy-policy v3 (edited)".
- `seq deploy` refuses a composite policy that would span machines and answers `instruction=` for each kind of policy;
  every `seq` command sends `User-Agent: sequence-ai-python/<version> (seq)`. Needs sequence-base 0.4.18.

## 0.38.0 — 2026-10-19

A composite policy plans each robot on a clock of its own and its plan can be anything, and `seq eval run --dry-run`
names the fastest way to run an eval within your GPU limit.

### Changed

- `@seq.plan` plans each robot `every_s` after its own last answer, the robots concurrently (`max_concurrency=`, 4 by
  default); `max_batch=N` plans the robots due together in one call, a list in and a list out. A policy with
  `@seq.plan` runs one copy on a worker: a `gpus_per_replica` that splits its cards into several copies is refused.
- `plan` returns any object, which `@seq.infer` receives as `plan=` (`instruction=` is refused, and the error says
  what to write); both may take `robot=`: the robot's own `robot.state`, and `robot.replan()` to plan it again at once.
- The dry run prints `fastest: <B> benchmark machines x <E> environments + <P> policy machines, <G> of your <L> GPUs:
  ~<time>, ~$<cost>` and, on the next line, the flags that run it (`--benchmark-machines`, `--envs`,
  `--policy-machines`).

## 0.37.19 — 2026-10-19

An adapter says which benchmark → policy pairs it converts between and an eval runs it only on those, and a run's
result is shown as its value alone.

### Changed

- `@seq.adapter(pairs=[(benchmark, policy), ...])`: each side a name in your account, `template:<name>` for anything
  made from that template, or `*`; `pairs="any"` for every pair. `seq deploy` refuses an adapter that declares none,
  and an eval whose benchmark and policy its adapter does not declare is refused before anything runs, the error
  listing the pairs it does declare.
- `seq adapter ls` lists each adapter's pairs, `seq adapter check --benchmark B --policy P` says whether it declares
  that pair, and an adapter prints without its class.
- `seq eval ls` and `seq benchmark ls` show a run's result as its value alone, in its metric's unit, under a heading
  named by the benchmark's metric; a run that failed reads `Fail`.

## 0.37.18 — 2026-10-19

Everything is written the newest way, and what only an older version needed is gone.

### Changed

- Removed: `@seq.policy(volumes=...)`, the retired `keep_warm=`, `container_idle_timeout=`, `allow_concurrent_inputs=`
  and `frames_per_chunk=` (now unknown arguments), an eval result's `clamped_by`, a decide result's `provider`, and the
  `seq robot` and `seq fleet` commands with `seq.robots`, `seq.robot`, `seq.robot_logs` and `seq.fleet`.
- `gpu=` takes one card: `gpu="L4"`, `gpu="H100:2"` for two of one type, or `gpu="cpu"`. A list is refused, and the
  error says what to write. A worker's price always counts its CPU and memory.
- The platform serves sequence-ai 0.37.16 and later; an older one is told to `pip install -U sequence-ai`.

## 0.37.17 — 2026-10-19

Registrations and evals stop the moment you cancel them, times print on this machine's clock again, and each setting
has one flag.

### Changed

- `seq benchmark cancel NAME` stops a benchmark's registration in flight, and `seq eval cancel` takes effect at once,
  a machine that is moving or waiting for its card included; nothing is billed after either. A GPU benchmark's machine
  that cannot draw its cameras moves to another three times at most.
- Every time the CLI prints is on this machine's clock, with the zone's name (`14:05 EDT`); `seq usage`, a policy's
  monthly budget and your plan's credit go by the billing period, the UTC month, and say UTC.
- The step limit is `--max-steps N` (`max_steps=`) and an idle worker's timeout is `--idle-timeout`;
  `--param eval.max_steps`, `--scaledown` and `seq eval run --compare` (`seq.eval(compare=)`: another policy is an
  eval of its own) are gone, and `seq.port.compare` is the function behind `seq port compare` (it was
  `seq.port.replay`).

## 0.37.16 — 2026-10-19

Each thing goes by one name, the one the platform and the console use, and each flag has one meaning; the old names
and spellings are gone.

### Changed

- Evals: `--policy-machines` and `--benchmark-machines` (`policy_machines=`, `benchmark_machines=`), `seq eval logs
  --machine`, `--spend-cap` on `seq eval run` and `seq eval schedule` (`spend_cap_usd=`), the new `--max-steps N` for
  the step limit (`max_steps=`), and `seq eval run POLICY` (`seq.eval(policy, ...)`).
- A version is always `NAME@vN`: `seq policy rollback NAME@vN`, `seq benchmark tasks NAME@vN`, `seq policy traffic
  NAME@v5=10 NAME@v4=90` and `NAME@vN --pin KEY`, and `upgrade --to TEMPLATE@vN`. `--csv` prints CSV and `-o PATH`
  writes it; `seq port export` and `convert` take `--trials N`, and `seq port compare` takes its scenes as arguments.
- Robot keys, max GPUs, the monthly budget and the spend cap are said by those names, every time the CLI prints is in
  UTC, every amount and price is written one way (a price by the hour, `$0.9612/h`), `seq usage --policy` gives the
  policy's spend with its storage as its page and its budget count it, a policy's model family is the one
  `@seq.policy(family=)` declares, and a new deployment's name may not contain a `.`.

## 0.37.15 — 2026-10-19

Each task has one CLI command: the duplicates are gone, with no aliases.

### Changed

- `seq usage` shows what your machines used and cost (this billing period, UTC; `--policy NAME`, `--since` / `--until`,
  `--csv`), in place of `seq billing export` and `seq policy usage`.
- `seq deploy --dry-run` checks the files and prices the policy, in place of `seq validate` and `seq policy estimate`.
- `seq policy versions` lists each version with its build, `seq benchmark versions` is new, `seq adapter source`
  replaces `seq adapter pull`, `seq init --list` and `seq eval ls --benchmark` replace `seq models --kind template` and
  `seq benchmark evals`, and a version is always written `NAME@vN`.

## 0.37.14 — 2026-10-18

Adapters and robot contracts drop their lossy, binarize and on_violation settings, a benchmark names the metric its
episodes succeed by, and `run()` can drive a policy with its own inputs and outputs.

### Changed

- Removed: `seq.AdapterKind`, `seq.AdapterTransform`, `seq.LossyTransform` and the transforms' `lossless=` and `lossy=`;
  a robot contract's `binarize` and `on_violation` (an action out of bounds is clipped); `seq benchmark compare`; and
  `--metric` on `seq eval run` and `seq eval episodes`.
- Added: `@seq.benchmark(success_metric=, metric_units=)`, `seq.tf.crop`, `seq.GpuRenderError`, `run()`'s
  `actions_of=`, `control_hz=` and `absolute=` for a policy with its own I/O, and `seq port` for the GPU porting
  commands that were under `seq benchmark` (`export`, `check-model`, `scaffold`, `compare --replay`, `convert`).
- `joint_positions` is optional, only absolute action spaces are smoothed between chunks, `seq.tf.action_project` only
  reorders or picks within one action space, values show in their declared units, and `seq eval report --csv` saves the
  console's CSV.

## 0.37.13 — 2026-10-18

A robot warns when its acts are much bigger than what its policy reads, `seq policy logs` names each machine by its
worker, and `seq policy estimate` shows a template's shared weights as parts that cost no storage.

### Changed

- A robot's act much bigger than what its policy reads (twice what its declared cameras read, or 1 MB for a policy with
  its own inputs) is warned about once a session; the act still goes as it is.
- `seq policy logs` names each machine by its worker, as `seq policy status` lists it (a0001, a0002, …), and
  `--worker` takes that name.
- `seq policy estimate` lists weights in named parts; a part an official template shares costs no storage.

## 0.37.12 — 2026-10-18

An eval's output gives no confidence interval or act latency, suites are gone, and `seq volume ls` shows a template's
shared weights at their size.

### Changed

- `seq eval report` and `seq eval status` print no confidence interval or act latency; `seq benchmark compare` gives no
  verdict and has no `--threshold`.
- `seq.Condition(suite=)`, `seq.conditions(suite_of=)` and `seq.SuiteAgg` are removed.
- `seq volume ls` lists a deployment that reads a template's shared weights at their size, "shared · not billed".

## 0.37.11 — 2026-10-18

`seq policy logs --build`, `seq policy builds` and `seq benchmark logs` say in their help that a build's log is kept as long
as its version is, and `seq benchmark logs --help` says what it shows.

## 0.37.10 — 2026-10-18

`seq policy logs` shows a policy's log as events: a line each for when, which machine, and what happened.

### Changed

- Each robot's acts in a run, and a print repeated many times, fold into one line that says when they ended; `--all`
  shows every act and print.
- What the policy printed while it loaded is marked `(setup)`, and what it printed on stderr `(stderr)`.
- `--worker` takes the machine's number as the log shows it, or the worker as `seq policy status` lists it.

## 0.37.9 — 2026-10-18

An eval runs on the card its benchmark was registered on, and `seq policy stop` says when a version is still building.

### Changed

- `seq eval run --benchmark-gpu` and `benchmark_gpu=` are gone; `--benchmark-instances` sets how many machines, each on
  the benchmark's card.
- `seq policy stop` says when a version is still building: it finishes and is checked as usual (its check billed), and
  it is the version that serves once you start the policy.

## 0.37.8 — 2026-10-18

A policy's access token can read its own policy's status, and a run that is stopped part-way is still counted in usage.

### Changed

- `seq policy status <name>` works with the policy's access token; listing, scaling, starting and stopping still need a
  full key.
- A run ended by SIGTERM, SIGHUP, the program's exit or `close()` reports its stalls; one ended by SIGKILL cannot.

## 0.37.7 — 2026-10-18

`seq policy builds` lists a deployment's builds, and `seq policy logs --build --version N` shows any version's build
step by step.

### Changed

- Each build says how long it took and why it was built: the files that changed, a new checkpoint, a template upgrade.
- Each step says how long it usually takes on this deployment.

## 0.37.6 — 2026-10-18

`seq deploy` shows each step of the build as it happens, with its time.

### Changed

- Each step (image, weights, waiting for a card, load, compile, snapshot, check) appears as it begins, with its own
  output and how long it took.
- A failed build says once which step failed, its last lines and, when the cause is certain, what to change.
- A ready build ends with the time of each step.

## 0.37.5 — 2026-10-18

`seq deploy` no longer fails when the platform has taken a new version while robots are connected: it says the build
has started and follows it.

### Changed

- When the platform takes a deployment but answers in a way this version cannot read, `seq deploy` says the build has
  started and how to follow it (`seq policy status <name> --wait`) instead of exiting with an error.
- `seq policy estimate` no longer bills storage for an official template's weights the platform reads from its own
  copy, or for weights in a volume you uploaded (billed as that volume); `seq gpus` names those templates.
- A robot waiting in line for a seat is told it is no longer in line when its deployment stops, instead of being told
  its seat was released.

## 0.37.4 — 2026-10-18

Adapters are no longer labelled lossy or lossless, and a new policy cannot declare `binarize` or
`on_violation="refuse"`.

### Changed

- `seq adapter ls`, `show` and `check`, and `seq eval report`, no longer show lossy or lossless; `lossy=` and
  `lossless=` are still accepted.
- A new `@seq.policy` that declares `binarize` or `on_violation="refuse"` is refused with how to do the same inside the
  policy; deployments made before keep working.
- Requires `sequence-base` 0.4.9 or later.

## 0.37.3 — 2026-10-17

`seq deploy` says that a new deployment's first worker stays on until its idle timeout, and is billed while it does.

### Changed

- `seq eval run --envs` says what a CPU benchmark runs when it is not given: one environment per core it declares, 16
  when it declares none.
- `sequence-ai` requires `sequence-base` below 0.5.

## 0.37.2 — 2026-10-17

`seq eval report` and `seq benchmark compare` read every episode of a large eval, not only the first 5000.

### Changed

- `seq eval report --csv`, `--trajectories` and `--replays`, and `seq benchmark compare`, page through all of an eval's
  episodes when it ran more than 5000 (with a platform from 2026-10-17).
- A benchmark's registration carries `task_descriptions` and `seeds_repeat` only when the benchmark declares them.

## 0.37.1 — 2026-10-17

`seq eval run` warns when a benchmark and a policy may not fit, and `seq policy logs` says why a refused act failed.

### Changed

- `seq eval run` and `--dry-run` print the platform's warnings, with what the machines' start costs when a run would stop
  at its first act; an act the platform refused is listed as `failed: r-b0381edd (refused: stopped, request
  r-48e71246)`, and `--rid` with the act's id shows the refusal.
- A robot moving to another worker no longer waits for its old worker to acknowledge the end of its episode.
- Warnings point at the line of your program that led to them; `seq adapter check` compares the action space and
  control rate too; `seq policy estimate` counts the billed time a worker runs after each use (`--starts-per-day`), and
  `--hours-per-day` takes 0 to 24.

## 0.37.0 — 2026-10-17

An eval says which of a benchmark's tasks it runs, how many times each, and on which seeds; there are no suites.

### Changed

- `seq eval run` and `seq.eval()` take `--tasks`, then `--trials 10` or per task (`--trials libero_spatial/3=20,*=5`;
  past a task's seeds they run again as repeated starts, unless `seeds_repeat=False`) and `--seeds 10-19`;
  `--param eval.max_steps=300` sets the most steps an episode runs. `--suite` is gone.
- `seq benchmark tasks <name>` lists each task with its seeds, episodes and description, and `seq eval episodes <id>`
  reads an eval's episodes a page at a time, by task, outcome or arm.
- `seq benchmark ls` and `status` say which template a benchmark comes from, and whether it is stable or beta.

## 0.36.14 — 2026-10-17

`seq policy usage` shows how often your robots' actions were on time, and their stalls.

### Changed

- A robot driven by `run()` reports how many of its action chunks acted on time and how many stalls it had, and the
  policy page and `seq policy usage <name>` show the on-time rate and the stalls.
- A robot moves to a newer version at its next episode boundary, and the line that says it will move names a latest time
  only when that is within two hours.

## 0.36.13 — 2026-10-17

Every request the platform refuses a robot is listed on its own line, with its time, request id, reason and hint.

### Changed

- `seq policy logs <name>` lists each refused connect and ready check of the deployment's robots, one line each (with a
  platform from 2026-10-17).
- `seq policy logs <name> --rid <id>` finds a refused request, or a connect, by its request id.
- An error the platform answered ends with why and its request id, for example `[stopped, request r-5a5a5a5a]`.

## 0.36.12 — 2026-10-17

`seq deploy` shows one more step of a version's first start, between loading the policy and checking one act.

## 0.36.11 — 2026-10-17

A robot whose worker is replaced no longer starts its next episode late.

### Changed

- The robot moves at the end of the first episode after the new worker has a place for it (with a platform from
  2026-10-17).
- Why it moves is said once, in one `[sequences]` line, with the time on your machine's clock.
- `seq policy status` and `seq deploy` say how often a worker is replaced, and name a version deployed before 2026-10-17
  with "deploy it again" to have its workers replaced about once a day.

## 0.36.10 — 2026-10-17

A robot waiting for a seat is told it is in line, and its place.

### Changed

- A robot that waits because every seat on its deployment's workers is taken is told so, with its place in line, rather
  than that the platform's GPUs are busy.

## 0.36.9 — 2026-10-17

A robot whose worker is being replaced moves to the new worker sooner.

### Changed

- A robot hears that its worker is being replaced within seconds, and moves at the end of its current episode, or at its
  next act when no episode is open.

## 0.36.8 — 2026-10-17

A worker is replaced about once a day by one started beside it, unless you turn that off.

### Changed

- `seq policy scale <name> --replace-beside on|off`: on, the default, a worker's replacement starts beside it, one
  worker past `max_gpus` billed as any worker, so no robot waits; off, it is replaced in its place, its robot waiting
  between two episodes.
- `seq policy status`, `seq deploy` and `seq policy estimate` say how a worker is replaced, when the next replacement
  is, and what a worker up all day costs with it.
- A lease renewal the platform cannot answer for a moment no longer ends a robot's run: it keeps acting and asks again.

## 0.36.7 — 2026-10-16

A robot cut off from the platform for half a minute keeps its slot, and one cut off for longer is told its slot went.

### Changed

- A robot cut off from the platform for up to about 30 s keeps its slot, as does a machine that slept or a process that
  was paused.
- When the platform says the slot went, one line on stderr says so, and the next act takes a slot again or waits in
  line.
- `ready()`'s help says how often to poll it, and a poll refused for your key's requests a minute (`429`) says so.

## 0.36.6 — 2026-10-16

A robot that gives up waiting frees its place for the robots behind it, and a robot whose worker is replaced at the end
of its episode waits for the new worker.

### Changed

- What a `ready()` poll holds for a robot — its slot, the worker starting for it, its place in line — is given back at
  `close()`, the end of a `with`, garbage collection and process exit, and within a minute when the robot is killed
  while it waits.
- A process stopped by `SIGTERM` or `SIGHUP` gives back what its open policies hold before it ends, and a handler of
  your own for either signal is kept.
- A robot whose worker was replaced in place at the end of its episode waits for the new worker, instead of being told
  its episode was reset.

## 0.36.5 — 2026-10-16

`ready()` says why a start that takes longer than usual has no estimate.

### Changed

- Past the time a start should take, `ready()`'s `eta_s` is `None` and `eta_unknown` is `"past_expected"`, where it read
  5 s for as long as the start took.
- `wait_until_ready()` timing out on such a start says it was still starting, slower than usual.

## 0.36.4 — 2026-10-16

A robot is told why it moves to another worker before it does.

### Changed

- A robot whose worker is leaving is warned once: why (its worker has served its hours, or a newer version serves) and
  the latest it is moved by.
- `seq policy status` lists each worker's copies of the model and the robots it seats apart (`copies`, `seats`,
  `sessions`), where one `slots` column read as seats.
- `seq deploy` says how its first machine's start went once it is ready, step by step.

## 0.36.3 — 2026-10-16

What a worker costs is estimated as it is billed: its card, then its CPU cores and GiB of memory apart.

### Fixed

- `seq policy estimate` prices a worker's CPU and memory apart from its card, at the cores and GiB the policy declares,
  and a CPU policy at its cores and GiB alone.
- `seq gpus` says a worker that declares no GPU pays its CPU and memory alone.
- sequence-ai 0.36.2 and before price a CPU policy at one figure for a small CPU worker: upgrade for the policy's own
  price.

## 0.36.2 — 2026-10-16

What a robot's acts go up as, binary frames or JSON, is yours to read, and `seq policy status` says whether a
deployment's workers take binary frames.

### Added

- `Policy.act_format` is `"frames"` while the connection sends a robot's frames as binary frames, `"json"` once its
  worker said it does not read them (or with `SEQ_ACT_FRAMES=0`), and `None` until the worker has answered.
- The debug log of `sequence_ai` says once per connection why its acts go up as they do.
- `seq policy status` says whether a version's workers take a robot's frames as binary frames, and for one built before,
  that a redeploy brings them.

## 0.36.1 — 2026-10-16

A robot's camera frames go to its worker as their own bytes rather than base64, a quarter less to upload on every act,
and the 3D replay is on by default.

### Changed

- A direct act sends its frames as binary frames when its worker reads them, and the policy sees exactly what it saw
  before; a policy deployed before 2026-10-16 takes JSON until you `seq deploy` it again.
- `SEQ_ACT_FRAMES=0` sends every act as JSON.
- An eval of a benchmark that declares `replay=seq.Replay(...)` records its 3D replay unless it says
  `seq eval run --replay none`.

## 0.36.0 — 2026-10-15

An eval can record its 3D replay, drawn in the console after the eval, when its benchmark declares
`replay=seq.Replay(...)`.

Needs sequence-base 0.4.6 or later.

### Added

- A benchmark declares it with `@seq.benchmark(..., replay=seq.Replay(engine="mujoco", state="engine_state"))`, beside
  `render=seq.Record(...)`, or hands its own frames with `engine="own"` through `seq.replay.scene` and
  `seq.replay.frame`.
- `seq eval run --replay 3d|none` and `seq.eval(..., replay=...)` choose whether an eval records it.
- `seq eval report <id> --replays DIR` saves each episode's replay and the scenes they are drawn in.

## 0.35.3 — 2026-10-15

`seq policy upgrade` and `seq policy status` say why an upgrade goes to a version other than the template's stable.

### Changed

- When the platform cannot deploy the stable, an upgrade goes to the newest version it can, and `--to` a version it
  cannot deploy is refused, naming one it can.

## 0.35.2 — 2026-10-15

An act's camera frames go up in one round trip on Linux, and `seq deploy` says what its build check did with a policy
that declares no observation contract.

### Changed

- `seq deploy` and `seq policy status` say what the build check did with such a policy: a robot-arm policy is acted on
  once with a guessed observation, and a failure is a warning, never a failed build.
- On Linux, the connection to the worker keeps its speed between acts, so an act carrying camera frames goes up in one
  round trip.
- With a slower policy, loading BBR on the robot (`sudo modprobe tcp_bbr`) keeps every act to one round trip.

## 0.35.1 — 2026-10-15

`seq init` writes only a version this sequence-ai can deploy, and `predict()` waits for a worker that is starting.

### Changed

- `seq init <template>` writes the template's stable when this sequence-ai can deploy it, else the newest version it
  can, saying which and why; a channel or version you name is written as named, with a warning when it does not deploy.
- `predict()` waits for a worker that is starting on every call, within `startup_timeout_s`, and `interrupt()` from
  another thread ends the wait.

## 0.35.0 — 2026-10-15

One declaration says how a worker serves robots, `@seq.concurrent(max_concurrent=N)` on infer, and a slot follows the
open policy.

### Changed

- `@seq.concurrent(max_concurrent=N, dynamic=True)` on the `@seq.infer` method replaces `@seq.batched` and its waits,
  `gpus_per_replica=` and `replica_start=` are `@seq.policy(...)` arguments, and `seq deploy` refuses the old ones,
  saying what to write instead.
- While a policy is open the SDK keeps its slot alive, so a robot that pauses between acts keeps it, and the slot is
  given back the moment the policy closes or its process exits.
- `run()` and `act()` drop what a run computed when it ends, so the next run never plays chunks computed for the one
  before.

## 0.34.43 — 2026-10-15

A robot far from its worker waits again when the worker says to, as before 0.34.42.

### Fixed

- `run()` and `act()`: a robot whose round trip is longer than a chunk's playback no longer sends an act that reaches
  its worker while another robot's runs.

## 0.34.42 — 2026-10-15

A robot far from its worker asks for its next chunk as soon as one lands.

### Changed

- `run()` and `act()`: a robot whose round trip is longer than a chunk's playback asks for the next chunk the moment one
  lands, and tells its worker its network time from its first act.

## 0.34.41 — 2026-10-15

A robot waiting in line is told its place and about how long, and a late chunk is bridged by the steps it carries past
its horizon.

### Changed

- `ready()` returns a `Ready` whose `queued` is True, with `queue_position` and `eta_s` (`None` when the wait cannot be
  estimated), and `wait_until_ready()` and `seq policy status` say where each queued robot stands.
- `run()` and `act()` play the steps a chunk carries past its `open_loop_horizon` while the next chunk is late, and
  `RunOutcome.spare_steps_used` counts them.
- `@seq.batched(serve_wait_ms=...)` is removed, and a policy that still names it is refused when `seq deploy` reads it.

## 0.34.40 — 2026-10-15

One copy of the model a card: several copies on one card are removed.

### Removed

- `@seq.concurrent(replicas_per_gpu=...)` is refused when `seq deploy` reads it, with what to write instead; a worker of
  several cards still runs one copy on each (`gpus_per_replica=`).
- The capacity line of `seq deploy`, `seq policy status` and `seq policy estimate` says whether one copy fits on the
  card, and `seq policy estimate --gpu CARD` says it for each card.
- `seq policy status` says a worker that ran out of GPU memory needs a card with more memory.

## 0.34.39 — 2026-10-15

A worker's price is what it is billed.

### Changed

- `seq policy status` and `seq deploy` price a worker's CPU and memory at what its workers use now when that is more
  than its build check measured, as it is billed.

## 0.34.38 — 2026-10-14

`seq policy estimate` says that copies on one card make each act take nearly twice as long when two act at once.

## 0.34.37 — 2026-10-14

A robot's first acts on a new connection go up in one trip each.

### Changed

- For a deployment that declares cameras, opening the connection takes a few more round trips, before
  `wait_until_ready()` returns or `run()` starts, so that each act goes up in one trip from the first.

## 0.34.36 — 2026-10-14

A failed act's report says what the robot got, so the console can say why each act failed.

### Changed

- A failed act's latency report carries the error's code, message and hint, each cut at 500 characters, and never the
  request.

## 0.34.35 — 2026-10-14

`seq deploy` prices the version it made.

### Fixed

- `seq deploy` prints the price and capacity of the version it made, where a version on its way in was priced at the
  serving version's requests.
- The capacity line suggests more copies on a card only when they would fit, and says that copies on one card make each
  act slower.

## 0.34.34 — 2026-10-14

`seq account` with an access token shows only that token's policy, and `wait_until_ready()` returns once the robot's
connection to its worker is open.

### Changed

- `seq account` and `sequence_ai.account()` with an access token (a robot's key) show the one policy the token is for,
  and say so on the first line.
- `wait_until_ready()` returns once the connection to the worker is open, so an act sent the moment it returns no longer
  waits, and `run()` does the same before its first observation.
- An access token reads nothing of its account but its own policy: the rest answers `403` with "this operation requires
  a full key".

## 0.34.33 — 2026-10-14

`seq deploy` and `seq policy status` say ready only once robots can connect, a session refused for `expect=` says how to
fix it, and an API key's credit limit is a total.

### Changed

- While a version's quick start is still being prepared, `seq deploy` ends "preparing", and
  `seq policy status <name> --wait` returns once robots can connect.
- A session refused for `expect=` says what the robot expects, what the policy sends, and how to fix it.
- An API key's credit limit, and a policy's access token's, is what the key may spend in all, never reset, and the key
  is refused at it until the limit is raised or removed on the console.

## 0.34.32 — 2026-10-14

A robot behind an HTTPS proxy checks its worker's certificate again, and the connection to the worker opens before the
first act.

### Changed

- A robot that reaches its worker through an HTTPS proxy checks the worker's certificate there too, and refuses a wrong
  one with `CertPinError`.
- The client opens its connection to the worker as soon as it connects, and once `wait_until_ready()` sees the policy
  ready, so the first act waits for neither.
- Where the platform offers it, the client reaches its worker through an endpoint whose certificate it checks against
  the certificate authorities it trusts (`SSL_CERT_FILE` / `SSL_CERT_DIR`, else certifi's).

## 0.34.31 — 2026-10-13

`seq policy status` says how long a start waits for the card, and the rest of the SDK says one card a deployment.

### Changed

- `seq policy status` says how long a start waits for the deployment's card, under its GPU line.
- `seq gpus`, `seq eval --dry-run` and the README say a deployment runs on its one card and waits for it to come free,
  10 min unless `seq policy scale <name> --start-wait` sets another.

## 0.34.30 — 2026-10-13

One card a deployment: a start waits for it up to `--start-wait`, says which card and how long when none came free, and
`seq policy status` names it.

### Changed

- `seq policy scale <name> --start-wait SECONDS` (and `sequence_ai.scale(start_wait_s=)`) sets how long a robot's start
  waits for the deployment's card: 60 to 3600 s, 600 unless set.
- A robot whose start found no free card is told which card, how long it waited and how to let it wait longer, and a
  start that ends this way is not billed.
- `seq policy status` names the one card a deployment runs on, with one worker's price an hour, and `policy.ready()`'s
  `first_choice` is always True now.

## 0.34.29 — 2026-10-13

`seq policy logs --rid` says where a first act's time went, and an unchanged official pi05-droid deployment keeps no
weights of its own to bill.

### Changed

- `seq policy logs --rid` splits an act's time into the connect, TCP, TLS, the transfer and the worker, and
  `seq policy logs` gives the p50 of each over first acts.
- `seq policy estimate` says a deployed official template's weights are not billed when it was deployed unchanged.
- A deployment that declares more than one card runs on the first alone, and a start waits for that card.

## 0.34.28 — 2026-10-13

`ready()` and `wait_until_ready()` answer for this robot, not ready while it waits in line for a slot, and the robot
first in line hears of a free slot in about a second.

### Changed

- While its deployment holds its `max_gpus` with every slot taken, `ready()` is `(False, None)` though a worker is up,
  and `wait_until_ready()` waits in line, saying its place.
- A robot first in line for a slot hears of a free one about a second after another robot frees it.

## 0.34.27 — 2026-10-13

`policy.ready()` and `seq policy status` say the card a worker runs on and its price an hour, and an observation that
does not match says how to fix it.

### Changed

- `seq policy usage` splits each day's p50 and p95 into inference and other, as `seq policy logs` and the console do.
- An observation that does not match what the deployment declares says how to fix each problem, with the call that sends
  it right.
- `policy.ready()` and `seq policy status` say the card a worker runs on and what one worker costs an hour on it.

## 0.34.26 — 2026-10-13

`region=` is retired: workers run wherever a card is free.

### Changed

- `@seq.policy(region=...)` imports with a warning, and `seq deploy`, `seq validate` and `seq deploy --dry-run` refuse
  it, saying to remove it.
- `seq eval run --region` and `seq.eval(region=)` are gone, and prices name no region.
- After a reconnect that places the robot on no worker yet, the next act asks the platform first rather than waiting on
  a worker that had failed.

## 0.34.25 — 2026-10-13

`seq policy logs` splits a round trip into the policy's inference and the rest, and a worker is billed from the second
it was up.

### Changed

- `seq policy logs` gives the round trip's p50 and p95 with inference and the rest apart.
- `seq policy estimate` says a policy that declares no weights keeps none, and gives a size under a tenth of a GB in MB.

## 0.34.24 — 2026-10-13

What keeping a deployment costs, said where you read prices.

### Changed

- `seq gpus` and `sequence_ai.gpus()` say the storage price, a GB a month of a deployment's weights, compile cache or a
  volume, billed every day it is kept.
- `seq policy estimate` says what keeping the deployment costs in storage beside its worker time.
- The [Pricing](https://generalsequences.com/pricing/) page lists the storage price beside the compute rates.

## 0.34.23 — 2026-10-13

`seq policy logs` sums up the acts your robots reported, and `seq account`'s balance line adds up to the cent.

### Changed

- `seq account` says each amount of its balance line to the cent, so the three add up, as the console does.
- `seq policy logs` sums up the acts your robots reported: how many, over what stretch, the round trip's p50 and p95,
  and the ones that failed.
- A worker's `idle_timeout` counts from its last act, whether or not a robot still holds it.

## 0.34.22 — 2026-10-13

`seq account` says what running work holds of your balance, and `seq policy source` checks what it downloads against the
version's tree hash.

### Changed

- `seq account` says what running workers and evals hold of the balance until they stop, and what is available now; the
  console says the same.
- `seq policy source` checks the files it unpacks against the tree sha256 the version recorded when it deployed, and
  stops if they differ.

## 0.34.21 — 2026-10-13

A worker's state says where it is and a sentence says why, in `seq policy status` and on the console alike: Failed only
for one that failed.

### Changed

- `seq policy status` and the console's Workers list say why a worker stopped when no robot took it within a minute of
  starting, and that nothing was billed for it.

## 0.34.20 — 2026-10-13

`seq deploy` uploads the folder your policy is in, whole, and says what it uploaded, and `seq policy source` downloads
the code a version runs.

### Changed

- `seq deploy` uploads the folder the entry file is in, less version control, caches, a virtualenv or build output at
  its root and what a `.seqignore` beside the entry file lists, and `--entry-only` uploads the entry file alone.
- `seq policy source <name> [--version N] [-o DIR]` downloads the code a version runs, as it deployed it, and the
  console's Versions has the same Download.
- `seq init` downloads the large upstream sources a template's image installs into `third_party/`, and
  `seq init --no-source` leaves them out.

## 0.34.19 — 2026-10-13

Pillow comes with sequence-ai: `pip install -U sequence-ai` is all a robot needs to fit and encode its camera frames.

### Changed

- `sequence-ai[image]` still installs, and adds nothing more.

## 0.34.18 — 2026-10-13

`seq usage` says what its dollars cover, `seq audit` the template versions an upgrade moved between, and an older
sequence-ai is told which version a template needs.

### Changed

- `seq usage` counts a deploy's checks as build checks, not evals, and each bucket's dollars are everything its machines
  cost: GPU, CPU and memory.
- `seq audit` says under each upgrade the template versions it moved between, and `seq policy logs` says "stop not
  recorded" for a machine whose stop was not recorded.
- A template version this sequence-ai is too old to read is answered with the version it needs, such as "toy-benchmark
  v4 needs sequence-ai 0.34.17 or later: `pip install -U sequence-ai`".

## 0.34.17 — 2026-10-13

An image can install from an official template version's frozen files, which the platform keeps.

Needs sequence-base 0.4.3 or later; `pip install -U sequence-ai` brings it.

### Added

- `seq.Image.template_files(template, version)`: each `run_commands` step after it reads that template version's frozen
  build files at `/seq/frozen`, read-only, while the image builds.
- `seq policy ls` and `seq policy versions` give times on your machine's clock with its zone's name.
- A policy class in a module that cannot be imported is refused with "the platform cannot load it".

## 0.34.16 — 2026-10-13

`seq policy` says what a redeploy would bring the version that serves, and `upgrade --help` says what `--to` takes.

### Changed

- `seq policy upgrade --help` and `seq benchmark upgrade --help` say that `--to vN` names any published version of the
  template: a beta one to try it, or an earlier one to go back.
- `seq policy ls`, `versions` and `status` say what a redeploy would bring the version that serves when its workers do
  not make a newer check yet.

## 0.34.15 — 2026-10-12

A deployment or benchmark made from an official template says which version of the template it runs and moves to a newer
one with one command, and a new version keeps the deployment's scaling.

### Added

- `seq policy upgrade <name>` and `seq benchmark upgrade <name>` move to the template's newer version, or the one
  `--to vN` names, printing what changes, what stays and one worker's price before and after (`--dry-run` deploys
  nothing).
- `seq policy status` and `seq policy versions` name the template version each version runs, and the newer stable with
  the command that moves to it.
- A new version of a running deployment keeps its scaling, and `seq deploy` sets only what its flags name and says what
  it kept.

## 0.34.14 — 2026-10-12

A camera frame that is not the size its camera declares is refused before sending only when the deployment's version
refuses it too.

### Fixed

- For a version built before the platform checked frame sizes, the warning says the frame passes on and that deploying
  it again makes it refuse such frames, and `validate="strict"` lets the frame go.
- `policy.contract["checks"]` says what the version the connection reached refuses beyond the shape it declares,
  `"frame_sizes"` for a camera frame of another size.

## 0.34.13 — 2026-10-12

A new version of a deployment that already serves takes over once its quick start is ready, and the CLI says so while it
waits.

### Changed

- `seq deploy`, `seq policy status`, `ls` and `versions` say when a new version has passed its check and waits for its
  quick start, while the version before it serves.
- `seq policy rollback` says which version it dropped, and drops a version still building or waiting for its quick
  start.

## 0.34.12 — 2026-10-12

Camera frames are fitted on the robot, by tools you call: nothing in the SDK resizes a frame behind your back.

Needs sequence-base 0.4.2 or later; `pip install -U sequence-ai` brings it.

### Changed

- `sequence_ai.resize_image(frame, width, height, *, mode, filter)` and `Policy.prepare_image(view, frame)` fit a frame
  on the robot, the second the way the deployment declares for that camera.
- `Policy.to_command(step)` and `Policy.to_commands(chunk)` give the command a robot runs: each value clipped into the
  deployment's `bounds`, then each `binarize` dimension set to 0 or 1.
- `sequence_ai.encode()` sends a frame at its own size, and a worker built from 12 October 2026 refuses a colour frame
  that is not its camera's width x height.

## 0.34.11 — 2026-10-12

`seq account` names the cards a deploy's build check holds beside serving and evals, so the parts add up to what is in
use.

## 0.34.10 — 2026-10-12

`seq audit` writes a billing line's amounts in dollars and a plan's change from the plan before it, as the console's
Audit page does.
