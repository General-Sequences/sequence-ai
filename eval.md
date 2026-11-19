# Benchmarks & eval

Score a **deployed** policy on a set of simulation tasks. You write a benchmark the same way you
write a policy — a decorated class — register it with the same `seq deploy`, bridge its observations and
actions to the policy with an **adapter** (named per eval; an official benchmark template registers the ones it pairs
with the template policies itself), then run `seq eval run` (or
`seq.eval(...)` from Python). Before
committing GPUs you run it with `--dry-run` and get back the wall-clock, the dollar cost, and how many GPUs you
will actually be granted. The sims and the policy both run on the platform; nothing runs on your machine.

`eval` is a capability on a policy you have already deployed, not a step before deployment. Deploy
serves the policy for `act()`; eval scores it. Neither gates the other, and eval only ever scores a
deployment **you own**.

> This file is the authoritative reference for the benchmark/eval surface. The inference/control-loop
> half of the platform's client (`sequence_ai.connect` / `act` / `run`) is documented in `README.md`.

---

## Install

```bash
pip install sequence-ai                        # on Python 3.10+: the client, the `seq` command, the decorators
export SEQUENCES_API_KEY=seq_dev_...           # your key identifies the account
```

The key names the account that owns your deployments and benchmarks and is billed for evals. A deployment made
under one account is invisible to another (an eval of someone else's deployment is refused).

---

## The three pieces

An eval composes three decorated classes. You already know the first from `README.md`:

| piece | decorator | what it is |
|---|---|---|
| **policy** | `@seq.policy` | the thing under test — deployed, `act()`-servable, `eval()`-scorable |
| **benchmark** | `@seq.benchmark` | the env you already have, as it is: its tasks, its seeds, its reset and step |
| **adapter** | `@seq.adapter` | two conversions: the env's observation to the policy's input, its answer to the env's actions |

All three register with `seq deploy`, but only a policy and a benchmark have machines of their own. An adapter has
none and builds no image: it belongs to neither side but to the pairing an eval runs, so the eval names it, and the
platform runs it inside the benchmark's container, between the sim and the requests it sends the policy.

Scaffold a runnable copy of each and edit in place:

```bash
seq init toy-policy                 # toy-policy/     — a closed-form @seq.policy that solves the toy benchmark
seq init toy-benchmark              # toy-benchmark/  — a minimal non-LIBERO @seq.benchmark
seq init toy-adapter                # toy-adapter/    — the @seq.adapter between them (registered with toy-benchmark)
```

---

## Writing a benchmark

A benchmark is the env you already have, as it is: a short class — your env itself, or one built on it — with three
marked methods the platform calls. **You write no loop, and the platform reads none of your observations or
actions:** each observation goes to the policy as a robot's own code would send it, and each action the policy answers
comes back to your step.

```python
import seq

TASKS_user = {"reach_left": "move the arm to the target on the left",  # {name: description} - the name your reset is
         "reach_right": "move the arm to the target on the right"}     # handed and an eval chooses by; "" for none

@seq.benchmark(
    name="toy-benchmark",
    image=seq.Image.debian_slim().uv_pip_install("numpy"),   # built on the platform when you `seq deploy` it
    gpu="cpu",                                                # numpy steps on CPU machines; a sim on a card names it,
                                                              # L4 when left out
    cpu=4,                                                    # a machine's cores (16 left out)
    memory=8,
    tasks=TASKS_user,
    seeds=range(5),                                           # trial k of a task starts at seeds[k % 5]
    max_seconds=5.0,                                          # an episode's time on the robot's clock, or {task: s}
    assets=seq.Volume.from_name("toy_scenes_user"),          # OPTIONAL: an uploaded volume → read-only at /assets
    secrets=[seq.Secret.from_name("hf")],                    # OPTIONAL: env vars for YOUR sim code (not the policy)
)
class ToyBenchmark_user:
    def __init__(self):
        # After the machine's snapshot, once a machine: build the env (init= on @seq.benchmark its arguments; none here).
        self.targets_user = {"reach_left": -0.5, "reach_right": 0.5}

    @seq.reset
    def reset_user(self, task, seed):
        # Start one episode of `task` at `seed`; return its first observation - anything, as your env gives it.
        self.env_user = _ToyEnv_user(target=self.targets_user[task], seed=seed)
        return self.env_user.observe()

    @seq.step
    def step_user(self, action):
        # One action: the row the robot would play this step. Return what came of it.
        self.env_user.step(action)
        reached_user = abs(self.env_user.x - self.env_user.target) < 0.05
        return self.env_user.observe(), reached_user, reached_user  # (observation, done, success)
```

**What the platform calls, and when.**

- Your file is imported once a machine, before its snapshot: what every eval shares and the snapshot can keep -
  kernels compiled, scene files read - goes at its top.
- Your class is built after each restore, before any episode, by its own constructor: `init=` on `@seq.benchmark` is
  its arguments, as a policy's is - `"{num_worlds}"` how many worlds to build, `"{policy.hz}"`, `"{policy.action_chunk_valid_len}"`
  or `"{policy.action_chunk_replan_at}"` what the evaluated policy declares - so your simulator's class is built as it
  is, handed what its own code takes: `init={"num_envs": "{num_worlds}", "control_hz": "{policy.hz}"}`. At registration no
  policy is evaluated, and a keyword argument that is a policy field alone is left out, its default holding. Cameras
  open here. Its other arguments with a default are settings an eval may change ([Parameters in an
  eval](#parameters-in-an-eval)). Declared `assets=` are mounted at `/assets` before it.
- `@seq.reset(self, task, seed)` starts an episode and returns its first observation.
- `@seq.step(self, action)` plays one action and returns `(observation, done, success)`, or `(observation, done,
  success, metrics)` with a dict of numbers of your own. An episode ends when `done`, or when its `max_seconds` have run
  out on the robot's clock; the success of its last step is the episode's.

**Tasks and seeds.** A task is a name to the platform: it hands it to your reset as it is, lets an eval choose by it
(`--tasks`), and groups the report's success rates by it, the description shown beside it. What a name means - its
scene, its goal, the instruction the policy reads in your observation - is your env's, and so is any variation. An
eval asking more trials than a task has seeds runs its seeds again, in order - each run after the first a repeated
start, marked so in the report, with the policy's randomness its own. Registration resets each task at its first seed,
so a name your env does not know fails there. A benchmark names no adapter: the eval names one. `gpu=` is `"L4"` (the
default) or `"RTX-PRO-6000"`, and each benchmark machine holds one such card, billed
for the time it runs, beside the policy's; registration resets the simulation on that card - each task at its first
seed, then again and again for 20 seconds - and refuses one that does not run there. It never steps it: an action is
your env's own.
`driver="580"` beside it names the NVIDIA driver's major version the card must have, for a simulator validated
against one (Isaac Sim 5.1 on 580): its machines run where the card has it, and one the card is not offered with is
refused at registration, naming the drivers it is; left out, the benchmark takes whichever driver its card has, and a
`gpu="cpu"` benchmark names none. `seq benchmark ls` and `status` write a benchmark that names one with the full
version its registration check saw: "L4 · driver 580.178.04".

**A vectorized env** - a GPU simulator stepping many worlds in one call - puts `@seq.concurrent(sizes=[N])` on its step:
N, one number, is the most worlds one call takes. Its reset is handed the worlds to restart and each one's task and
seed, its step the worlds to step and one action for each, returning one tuple a world, in that order; its constructor
builds its worlds, `init=` handing it how many - `"{num_worlds}"`, which it must name:

```python
@seq.benchmark(name="warp_reach_user", gpu="L4", tasks=TASKS_user, seeds=range(50), max_seconds=15.0,
               init={"worlds": "{num_worlds}", "control_hz": "{policy.hz}"})
class WarpReach_user:
    def __init__(self, worlds, control_hz=20.0):
        self.sim_user = MyWarpSim_user(n=worlds, control_hz=control_hz)

    @seq.reset
    def reset_user(self, worlds, tasks, seeds):
        return self.sim_user.reset(worlds, tasks, seeds)     # one first observation a world

    @seq.step
    @seq.concurrent(sizes=[64])
    def step_user(self, worlds, actions):
        return self.sim_user.step(worlds, actions)           # [(observation, done, success), ...]
```

An eval builds the fewest of N, its episodes and what the policy takes at once; registration builds N and resets each
task in them, so a size that does not fit fails there, not halfway through an eval. Without `@seq.concurrent`, each
copy of the benchmark steps one world, and the platform runs as many copies as the eval needs.

A GPU benchmark's cameras are drawn on its card too. A machine whose GPU cannot draw them (its renderer - EGL, or
SAPIEN's Vulkan - does not start on the card, or gives a software renderer) is not used: registration checks again on
another machine, and an eval's benchmark machine moves to another before its first episode: three moves at most, four
machines in all. If none of them can, the registration fails, or the eval stops (stop reason `no_gpu_render`, status
failed), naming the renderer it got and its error; none of those machines is billed. Either can be stopped at any time,
a move included (`seq eval cancel <id>`, `seq benchmark cancel <name>`): nothing is billed from then on. The official
templates say so as they set up; your own benchmark raises `seq.GpuRenderError` from its constructor, before any
episode, rather than drawing in software:

```python
raise seq.GpuRenderError(gl=gl_user, backend="egl", egl=renderer_error_user)  # backend="vulkan" for a Vulkan renderer
```

`gl` (the GL or device the renderer got), `backend` (`egl`, `osmesa`, `vulkan`) and `egl` (that backend's own error)
are all optional. It is found inside your own error too (`raise MyError(...) from seq.GpuRenderError(...)`); a
benchmark that only prints that it fell back is taken to be drawing on its card.

**A CPU benchmark** says `gpu="cpu"`: its physics runs on CPU machines, as the original simulator does, with no
porting. `cpu=N` is the cores one benchmark machine holds (16 when left out) and `memory=` is for all of them (2 GB a
core when left out). It is billed by the core-hour and the GiB-hour, holds none of your GPUs, and is the option for
tests, first runs and reproducing a paper's numbers: at the same number of worlds it costs more per world step than a
GPU version on an L4, and the more so the more worlds run — about 1 to 3 times at 16 and 5 to 11 times at 64 — because
every world holds a core while it waits for the policy, where the GPU version holds one card for all of them.
`seq eval run --dry-run` shows the cost of both.

**Success and metrics are yours.** The report gives each task's success rate - the share of its episodes whose last
step said success - overall and per task with its `n`, and beside it the mean of each metric your step returned. The
platform names no metric of its own, and units are not declared. A benchmark declares no recording either.

**A 3D replay of every episode.** Declare `replay=seq.Replay(...)` and, each step, the platform records the world
transform of everything in the scene - nothing is rendered for it - and the owner turns, zooms and scrubs it in the
console after the eval, beside the mp4. It is on by default: every eval of such a benchmark records it unless it
says `--replay none` (the console's Run eval dialog has the same switch);
`seq eval report <eval_id> --replays DIR` saves the files. `seq eval replay <eval_id>` plays them on your own machine
in the console's own player - a page on 127.0.0.1 that only the browser its link was opened in can open - and `seq eval
replay FILE` plays a saved one offline, with the scene saved beside it; `seq.read_replay(path)` reads one into numpy
arrays (`t`, `step`, `pos0`, `quat0`, `pos`, `quat`, `looks`) for your own analysis. For MuJoCo (MuJoCo Warp too) and
SAPIEN the platform reads
the simulation itself and exports the scene once, so the declaration is one line:

- `engine=` - which simulation it reads: `mujoco`, `sapien` or `isaac`; left out, the benchmark hands its frames
  itself (`"own"`). The console draws the benchmark's icon from it.
- `state=` - the name of your method that gives the engine's state of what runs: for a benchmark that steps one world
  at a time `state()`, for a vectorized one `state(worlds)`, one per world. MuJoCo: `(MjModel, MjData)`, or with MuJoCo
  Warp `(mjModel, mjw.Model, mjw.Data, world)`; SAPIEN: the `sapien.Scene` the world runs in; Isaac: `(stage, scene,
  env_index)` - the USD stage, the Isaac Lab `InteractiveScene`, and the env index.
- `looks=` (optional) - the name of your method that says what the model does not carry and an episode changes, such
  as a per-world colour: `{node: {"rgba": [r, g, b, a]}}` or `{node: {"visible": False}}`.

The replay reads the poses your simulation has computed, as they are: a reset that only writes positions (`qpos`)
leaves the old poses until the next step, so frame 0 would show the world before the reset. End your reset with the
engine's own update - `mujoco.mj_forward(m, d)`, or with MuJoCo Warp `mjw.kinematics(m, d)` - before it returns.

**Isaac only:** set `scene.step_dt = env.step_dt` on the `InteractiveScene` your `state` returns - Isaac Lab's
`env.step_dt` is `physics_dt × decimation`, the seconds one control step advances. On the current Newton image the
simulation clock is often unreachable, so the replay's timeline falls back to `steps × step_dt`; **without**
`scene.step_dt` the adapter can only use `physics_dt`, and a task with `decimation > 1` then plays back `decimation ×`
too fast (its recorded `fps` that much lower). MuJoCo and SAPIEN carry their control dt themselves and need nothing.

Another engine hands the replay over itself: `seq.Replay(engine="own")`, then `seq.replay.scene(slot, glb)` once an
episode (a glTF binary, its nodes flat children of a root `seq_root` that turns Z-up into Y-up) and
`seq.replay.frame(slot, t, nodes, pos, quat, step=None, looks=None)` after the reset and after every step - `slot` the
world (0 for a benchmark that steps one world at a time) - (seconds
since the reset, metres in the world with Z up, quaternions w x y z). Both do nothing outside an eval. A replay is kept
with the eval's recordings, billed as they are, and removed with `seq volume rm eval:<eval id>`; the dry run says about
how much an eval's replays keep. A replay holds about 20 bytes a step for each thing that moves: LIBERO's, about 45 of
its 89 parts moving, come to some 850 bytes a step, 60-120 KB an episode - about a quarter of its mp4. The scene a
replay is drawn in is kept once for all your evals that share it (libero_spatial's is 6.7 MB), outside their
recordings, and is not billed. Recording it took about 1 ms a step on LIBERO: 16 episodes with it cost what they cost
without it, $0.14. A method named that the class does not have, or `engine="mujoco"` / `"sapien"` with
no `state=`, is refused when the benchmark is defined.

**Assets are your own uploaded volumes.** Upload meshes, scenes or a dataset once with
`seq volume put toy_scenes_user ./scenes --kind dataset`, then declare `assets=seq.Volume.from_name("toy_scenes_user")` (several
volumes mount at `/assets/<name>`). The volume must exist and have finished uploading when you register and
when you eval; each run reads its current version **read-only** — your sim can never change what the next
eval sees.

**Secrets are for your sim's own code.** Store a credential once (`seq secret set hf --env HF_TOKEN`) and
declare `secrets=[seq.Secret.from_name("hf")]`: the build check and every episode sandbox get it as
environment variables, the policy under test never does, and any error text in the report has its values
replaced with `***`. Only secrets your account owns can be declared — registration refuses any other.

**No manifest files, and the folder goes up whole.** Registering a benchmark writes no `manifest.yaml`,
`Dockerfile`, or equivalent into your repo — everything comes from the decorator — and `seq deploy <folder>` uploads
the folder, whole, less what its `.seqignore` files list, and anything else the image names with `add_local_file` /
`add_local_dir` / `add_local_python_source`. A folder of several classes (a GPU and a CPU benchmark, an adapter) deploys
the one `--name` picks. The deploy prints the list.

---

## Writing an adapter

An adapter is the robot's own code, in the eval's loop: two conversions between a benchmark's env and a policy, on a
plain class.

```python
import seq

@seq.adapter(name="to_my_pi05_user", pairs=[("libero-goal", "my_pi05_user")])  # the benchmark → policy pairs it is for
class ToPi05_user:
    @seq.observation
    def to_policy_user(self, obs):       # the env's observation → what the policy's observe is sent
        return {"image": obs["agentview_image"][::-1], "state": obs["robot0_eef_pos"],
                "prompt": obs["instruction"]}

    @seq.action
    def to_env_user(self, answer):       # the policy's answer → the actions the env plays, one a step
        return answer["actions"]
```

The toy templates pair a policy with its own I/O (`seq init toy-policy`: inputs `{state, target}`, outputs `{delta}`)
with its adapter (`seq init toy-adapter`):

```python
@seq.adapter(pairs=[("template:toy-benchmark", "template:toy-policy")])
class ToyAdapter_user:
    @seq.observation
    def to_policy_user(self, obs):       # {arm_x, target} → {state, target}
        return {"state": obs["arm_x"], "target": obs["target"]}

    @seq.action
    def to_benchmark_user(self, answer):  # {delta} → one action, the step it plays
        return [[float(answer["delta"][0])]]
```

In the eval's loop each observation the env gives goes through the `@seq.observation` method to the policy, as a
robot's own code would send it; each answer comes back through the `@seq.action` method, and the platform plays its
actions on the env's step, one a step: it asks for the next answer at the policy's `action_chunk_replan_at` step and
switches to it at its `action_chunk_valid_len` step, the world held still until it arrives - no network is modeled,
so there is no latency to set.
Without an adapter the observation goes to the policy as it is, and the answer must itself be the actions. What the
policy is sent and what it answered are what its episode keeps. Both methods are required, each taking the one value
it is handed; either one raising ends that episode alone, its report saying what it raised. An `__init__` with
keyword parameters, each with a default, is what an eval may set ([Parameters in an eval](#parameters-in-an-eval)).

**It marks what it sends as a robot's code does.** A camera frame wrapped `sequence_ai.Video(frame)` (HxWx3 uint8),
a depth map `sequence_ai.Depth(depth)` and a tactile reading `sequence_ai.Tactile(touch)` go as a robot's `observe()`
sends them - the camera as low-latency video with a keyframe every second of the robot's clock, the depth
losslessly - and the policy's worker turns each back into its array before the policy reads it: the policy reads
what a robot that marks the same items would send it, and its episode keeps them compressed, as a robot's. Anything
unmarked goes as it is, and an env may mark its own observation the same way. An item that does not encode (a
camera frame that is not HxWx3 uint8, say) ends its episode, the report naming it.

**It is registered with its benchmark, or on its own, and the eval names it.** Registering an official benchmark
template registers in your account, beside it, each adapter the template pairs with a template policy - `toy-adapter`
with `toy-benchmark`; deploying a template policy brings the one a benchmark template you registered pairs with it -
and `seq deploy` says which (`also registered in your account: toy-adapter@v1`), so an eval of the two names it with
nothing deployed for it. An adapter of yours, or one you changed, registers on its own: `seq deploy toy-adapter`
registers the class as
a version — its file is sent, nothing is built or run — under its `@seq.adapter(name=…)`, else its class name in
lower case with hyphens (`ToyAdapter` → `toy-adapter`). The version is the content: the same file registered again
is the version it already is, a changed one the next (`toy-adapter@v2`). An eval names it as `--adapter
toy-adapter@v1`, or `toy-adapter` for its latest, or as a local file, `--adapter ./toy-adapter/adapter.py:ToyAdapter`,
registered first. When the eval starts, the benchmark's container fetches that version, checks the file is the one
registered — a file that is not never runs — imports it, and runs it between the sim and the policy. Changing the
adapter changes neither the benchmark's version nor the policy's; the report's first line reads `ran: benchmark
toy-benchmark v1 + adapter toy-adapter v2 + policy toy-policy v3`. A name that another
`module:Class` registered is taken over only with `seq deploy --override`.

The file runs in the benchmark's image, so it imports what that image has (numpy, say): one file, at most 256 KB.
`seq adapter ls` lists your adapters, each with its latest version and the eval that last ran it;
`seq adapter check adapter.py` reads it here - its two methods and its pairs - without sending anything (an eval checks
the pair it runs as it starts); `seq adapter rm toy-adapter` removes every version (refused while an eval that runs it
waits or runs).

**When you need one.** When the policy takes what the env gives and answers what its step plays, none. Otherwise the
policy refuses the first observation and the eval stops there, its report saying what was refused, in its own words,
and billed only for what ran; or the env's step raises on the actions it is handed, and the episodes it was stepping
end, each saying what it raised.

---

## Deploy and register

```bash
seq deploy toy-policy                                     # a @seq.policy deploys → the name you eval (toy-policy)
seq deploy toy-benchmark --name toy-benchmark             # a @seq.benchmark registers; its sim image builds
seq deploy toy-adapter                                    # only once you change it: toy-benchmark registered toy-adapter@v1
seq policy ls                                             # your deployments
seq adapter ls                                            # your adapters
seq benchmark status toy-benchmark                        # building / ready / failed
```

`--name` sets the deployment's name; without it the name is the decorator's `name=`, else the class name in lower
case with hyphens (`ToyPolicy` → `toy-policy`). A benchmark's name is its `@seq.benchmark(name=…)`. In Python, `seq.deploy` does the same and
returns the name:

```python
import seq
from policy import ToyPolicy as ToyPolicy_user       # toy-policy/policy.py's class

model_user = seq.deploy(ToyPolicy_user)              # a handle that stands for "toy-policy"
```

`seq.eval`'s first argument is that name, not the class object.

**Stopping a registration.** `seq benchmark cancel <name>` stops the registration in flight: its check's machine ends
at once, and what its machines ran until then is billed; the version it was registering is marked failed (canceled),
and evals run the version before it, when one passed its check.

**Deleting a benchmark.** `seq benchmark rm <name>` frees its name and its place in your account's quota at once and
stops a build in flight; the evals that ran it keep their reports. An eval running or waiting on it refuses the delete
unless `--force` cancels it; `--dry-run` shows what it would touch.

---

## Estimate first (`--dry-run`)

`--dry-run` returns an estimate and **stops before reserving anything** — no machine is created and no eval is
left behind. It shows the layout as you wrote it (or the defaults: one policy machine on the deployment's cards, one
benchmark machine on the benchmark's card) with the worlds each benchmark machine builds, the cards it needs against
your account's limit
and what is free now, and the estimate. The cost is what this policy's earlier evals on this benchmark were billed for
the same work (before its first one, what the benchmark's registration and the policy's deploy measured); the time
adds the usual wait for a card, and says how long it can take when cards are slow to come. A real run holds more than
its cost before it starts — every episode running to its time limit, each machine on the costliest card of its list,
plus a margin, at most its spend cap (`--spend-cap`) — and the dry run says how much (`up to $… held`;
`est.held_usd` in Python). You pay only for the time your machines run.

```bash
seq eval run toy-policy --benchmark toy-benchmark --adapter toy-adapter --dry-run
# policy: toy-policy x 1 machine(s) on CPU
# benchmark: 1 machine(s) per arm on L4, 16 worlds each
# cards: policy 1 × CPU machine (no GPU) + benchmark 1 × L4 = 1 GPUs at once
#   your account: limit 8, in use serving <n>, evals <n>, <free> available
#   fits: no wait for GPUs           (or: <n> short (why): submitted, it would wait in line up to 600 s …)
# cost: about $<cost>
#       up to $<held> held
# time: about <minutes> min
#       up to <minutes> min when GPUs are slow to come
```

```python
est_user = seq.eval("toy-policy", "toy-benchmark", adapter="toy-adapter", dry_run=True)
print(est_user.wall_clock_s, est_user.cost_usd, est_user.held_usd)
```

---

## Run it (and lay it out for speed)

```bash
seq eval run toy-policy --benchmark toy-benchmark --adapter toy-adapter
```

```python
report_user = seq.eval("toy-policy", "toy-benchmark", adapter="toy-adapter")

# each task's success rate - what @seq.step said of each episode's last step - and each metric your step returned
m_user = report_user.metrics["success"]
print(m_user.overall, m_user.n)  # overall over every episode + sample count
print(report_user.units)         # each metric's unit: {"success": "rate", ...} - a rate shown as a percentage
for t_user in m_user.per_task:
    print(t_user.task, t_user.value, f"(n={t_user.n})")
# every episode, a page at a time - by task, outcome (success / failure / error) or arm - by
# the eval's id (what `seq eval run` printed, or seq.eval(..., wait=False) returned)
page_user = seq.eval_episodes("ev-...", task="reach_left", outcome="failure")  # {"episodes": [...], "next_cursor"}
print(report_user.status)         # completed / failed / canceled (queued / running before it ends)
print(report_user.cost_usd)       # actual dollars, reconciled from real GPU-seconds
print(report_user.dashboard_url)  # the eval in the console
print(report_user.videos)         # the recordings the platform made, if any
print(report_user.incomplete, report_user.errors)  # whether every episode ran, and what went wrong if not
print(report_user.arms)        # with --sweep: a report per arm
print(report_user.timing, report_user.stop_reason)  # each phase's mean time per occurrence; why it stopped early, if it did
```

`seq.eval(..., wait=False)` returns the eval's id at once; `seq eval status <id>` says its status (queued for GPUs —
how many it needs, how many are free, how long it has waited, when it gives up — running, completed, failed or
canceled), `seq eval report
<id> --wait` reads it later, and `seq eval cancel <id>` stops it at once, a machine waiting for its card or moving
included (no new episodes; what ran is reported, and billed up to the cancel).

Before its first episode, a waiting `seq eval run` and `seq eval status` say what the eval waits on — a card, its machine's
boot, its image to be ready, its image and weights coming, its benchmark's machine — how long so far and about how long
is left (`starting`, in `GET /v1/eval/{id}/details`, each side's own in its `sides`). Its machines' start is in the dry
run's time and cost: a machine is billed from when it starts, its boot, image and weights included, and only the wait
for a card (with no card, for the machine) is not. `seq eval report` says how long the wait for a card and the machines'
start took, apart.

**Each act carries a seed of its own.** Act *n* of the episode that started at seed *s* carries a seed hashed
from both: a rerun of the episode carries the same ones, so it replays exactly, and a policy that draws its noise from
`seed` (flow matching, diffusion) starts every replan from fresh noise — not from the noise its last one started from,
which kept a stuck state stuck (RoboCasa, the same 54 episodes: 26/54 with fresh noise per act, 7/54 with one seed
for the whole episode). A trajectory's act lines record the seed each act carried.

**The layout is yours.** Each world's episode is answered by one policy machine — the one with the fewest
running episodes when the episode starts — and keeps to it to its end. A policy machine runs the requests that reach
it together, in the batch sizes its `@seq.concurrent(sizes=[...])` lists, so its largest batch is full only when
enough worlds ask at once (`seq eval report` shows how full they were). If a policy machine ends mid-episode,
its episodes go on on another one — those of a policy that keeps an episode's state run again from their reset,
since their state stayed behind — and the report counts them. Benchmark machines take episodes from the eval's queue as their
worlds free up, so a machine that is preempted loses nothing and one that finishes early takes more. More machines
finish sooner and cost more per second; the dry run shows both. The platform never derives
a number for you and never trims one: every card counts against your account's limit, a layout past it is refused,
and while your cards are busy the whole eval waits for them (up to 10 minutes, then it fails with stop reason
`queue_timeout`; nothing is held or billed while it waits).

**Where the time went.** `seq eval report` prints each phase's mean time per occurrence (`report.timing.step_ms`) —
what one occurrence cost, not its share of the whole:

| phase | one occurrence is |
|---|---|
| `setup` | building the benchmark - its constructor, `init=` filled - once per benchmark machine |
| `reset` | resetting one world for a new episode |
| `observe` | sending one world's observation to the policy |
| `round` | one policy round: every running world's request sent at once, until the last answer is back |
| `act` | one request's round trip as the benchmark saw it: network, the wait to join a batch, the forward pass |
| `forward` | the forward pass that answered it — its whole batch's, as the policy reported it |
| `queue` | the request's wait for a free replica on the policy, once its batch was sent |
| `step` | stepping one world by one action |
| `record` | reading one frame of a recorded episode |
| `encode` | encoding one recorded episode's video — per recording, so it is seconds where a step is milliseconds |

An eval's acts go from the benchmark's machines to the policy's workers as JSON: binary frames (sequence-ai 0.36.1)
are a robot's own direct connection, so `act` above is the same with or without them.

Under it, **policy load** says how the eval used each policy machine: its requests and batches, how many requests
per batch and batches per round, the share of its time spent computing, and in how many rounds the worlds
waited for it last (and by how much) — with the share of time the simulation was busy and the share it waited on the
policy. The same numbers are in `seq --json eval report <id>`, under `details.load`.

Then the **link between the benchmark and policy machines**: how long a ping takes on an open connection between
them — the network alone — at its median, 90th percentile and largest, and each request's time outside the policy's
worker (its round trip less the worker's own time: the network and the observation's upload), at its median and 90th
percentile. A millisecond or two means the two sides are close; tens of milliseconds, that they are far apart. Under
`details.link` in `seq --json eval report <id>`.

---

## Every `seq eval run` flag

| flag | what it does |
|---|---|
| `--benchmark NAME[@vN]` | your registered benchmark: its latest version that passed its check, or `NAME@vN` for any version it keeps (`seq benchmark versions`) |
| `--adapter NAME[@vN]` | the adapter: a registered one (`toy-adapter@v1`, or `toy-adapter` for its latest), or a local file, `./adapter.py:Class`, registered first; none when the policy and the benchmark already agree |
| `--tasks a,b` | only these of the benchmark's tasks (`seq benchmark tasks <name>` lists them; default every task) |
| `--trials N` / `--trials TASK=N,...,*=M` | how many times each chosen task runs - its declared seeds in order, and past the last the same seeds again as repeated starts, marked so - or a count per task (`*` for the rest; with no `--tasks` and only per-task entries, only the tasks named run) |
| `--seeds 10-19` / `--seeds TASK=10-19,...` | which seeds run, for every chosen task or per task, instead of the first N; only seeds the benchmark declares |
| `--policy-machines N` | N policy machines, sharing every benchmark machine's requests (default 1) |
| `--benchmark-machines N` | N benchmark machines per arm, each on the card the benchmark was registered on (or a CPU machine), taking episodes as they free up (default 1) |
| `--spend-cap USD` | the eval's spend cap: it stops once its machines (GPU, CPU and memory) have cost USD; the report covers what ran and says so |
| `--dry-run` | show the layout, the cards against your account and the estimate; start nothing |
| `--rerun-failed EVAL_ID` | run only that eval's errored episodes again (same adapter and selection); the new report covers both runs |
| `--rerun-unsuccessful EVAL_ID` | run the episodes that eval ran without error and did not succeed on (each one's `success`, what its last step said) again, as an eval of their own, to see why. Never merged into that eval's report: one keeping reruns of only its failures would score above what the policy does |
| `--gate "success>=0.9"` | exit 8 unless the eval completed with every episode and the metric's overall value passes (repeatable) — for CI |
| `--param [SIDE.]NAME=VALUE` | set a parameter for the whole eval: the policy's (`policy.replan_steps=3`), the adapter's (`adapter.image_size=128`), the benchmark's (`benchmark.noise_std=0`) or the eval's own `eval.max_seconds=20`, an episode's time limit on the robot's clock (a changed limit is not a standard eval, and the report says so); the bare name when only one side declares it (repeatable; see [Parameters in an eval](#parameters-in-an-eval)) |
| `--sweep [SIDE.]NAME=V1,V2` | an arm per value of any of them (every combination when repeated), run side by side in this eval on the same episodes; the report has a table per arm |
| `--replay 3d\|none` | whether the eval records its 3D replay when the benchmark declares `replay=` (default `3d`: every episode, drawn in the console after the eval) |
| `--no-wait` | return once the eval has started; wait for its report with `seq eval report <id> --wait --timeout S` (default 7200 s) |

## Every `seq eval report` flag

The report prints each metric's value - `success`, the share of episodes that succeeded, as a percentage - and the
episodes it is over (`n`), overall and per task, and the run's manifest: the policy's
version, the benchmark's version, the adapter's name, version and sha256, and the seeds — everything needed to run the
same episodes again. `seq eval run` prints it when the eval ends; `seq eval report <eval_id>` reads it at any time.

| flag | what it does |
|---|---|
| `--wait` | wait until the eval ends: completed, failed or canceled |
| `--timeout S` | seconds to wait with `--wait` (default 7200) |
| `--csv` | the eval's episodes as CSV, a row per episode - the file the console downloads - printed in place of the report, or written to `-o PATH` beside it |
| `--out PATH` / `-o PATH` | with `--csv`: the file to write it to |
| `--records DIR` | save every episode's record - what its policy was sent, its cameras as video, and each answer (`.rec.zst`) - into DIR |
| `--replays DIR` | save every episode's 3D replay (`.seqreplay`) and the scenes they are drawn in (`scenes/<sha256>.glb`) into DIR |

The CSV is the platform's one file of the eval's episodes, the one the console's Download CSV saves too: a row per
episode — `index, arm, task, seed, start, repeat, success, seconds, max_seconds, paused, steps, wall_s, terminated,
truncated, timed_out, error`, then each metric as `metrics.<name>` (`seconds` on the robot's clock, `paused` the steps
it stood still waiting for an answer) — with any cell a spreadsheet would run as a formula
starting with `'`. `seq eval episodes <eval_id> --csv [-o PATH]` gives the same file of the episodes its `--task`,
`--outcome` and `--arm` take. `seq eval run` takes neither `--csv` nor `-o`.

```bash
seq eval run my_pi05_user --benchmark libero-spatial --tasks libero_spatial/0,libero_spatial/1 --trials 5 \
    --gate "success>=0.8"
seq eval report <eval_id> --csv -o episodes.csv  # a row per episode, the file the console downloads too
seq eval report <eval_id> --records ./records    # each episode's record, its cameras as video, when the eval kept them
seq eval ls my_pi05_user                         # a deployment's evals over time
```

`seq --json eval report <id>` prints the report and its per-episode details as JSON.

---

## Parameters in an eval

Every side declares what an eval may set the way `@seq.infer` always has — keyword arguments with defaults, typed
`int`, `float`, `bool` or `str` — and registering it sends the declarations:

```python
class MyPolicy_user(seq.Policy):
    @seq.infer
    def infer_user(self, obs, *, seed=None, replan_steps: int = 5): ...   # policy.replan_steps

@seq.adapter(name="to_my_policy_user", pairs=[("my_benchmark_user", "my_policy_user")])
class ToMyPolicy_user:
    def __init__(self, *, image_size: int = 224): ...                     # adapter.image_size
    ...                                                                   # and its two methods

@seq.benchmark(..., init={"worlds": "{num_worlds}"})
class MyBench_user:
    def __init__(self, worlds, *, noise_std: float = 0.02): ...           # benchmark.noise_std (what init= hands
                                                                          # it is the platform's)
```

A benchmark's constructor keyword typed as a setting no eval can set is refused when you register it, saying how to type
it: `scale: float | str | None` as `scale: str | None`, `steps: int = "auto"` as `steps: str`. One untyped, or typed as
nothing a `--param` could be (a `dict`, a class of your own), keeps its default, and `seq deploy` warns of a typed one.

The eval has one of its own, `eval.max_seconds`: how long an episode may run on the robot's clock, the benchmark's
`max_seconds` unless it is set. Set any of them for the whole eval with `--param side.name=value` — the bare name works
when only one side declares it, so `--param replan_steps=3` still does — and sweep any of them with `--sweep` (every
combination is an arm). Each value is checked before anything is reserved: a name no side declares or a wrong type is
refused with every parameter this eval can set. `--dry-run` prints each arm's values and where each comes from.

**Who decides.** The eval's value, else the default of the side that declares it. Each eval builds its own adapter with
the values it set, and every benchmark machine's constructor is handed them. The report prints every arm's values with
their sources (`params: policy.replan_steps=5 (policy's default) · eval.max_seconds=20 (set)`) and how many steps asked
the policy of all it took; `seq --json eval report` has them under `arm_params`.

**The arm.** A benchmark that takes any arm says so with `init={"robot": "{policy.robot}"}`: an eval runs it on the
arm the policy declares (`@seq.policy(robot="Panda")`), and `--param benchmark.robot=UR5e` or
`--sweep benchmark.robot=Panda,UR5e` sets another. An arm the benchmark does not take is refused before anything is
reserved, naming the arms it does.

## Billing, in one paragraph

Billing is postpaid: you use first, and each month's usage is billed at its end. A non-dry-run eval holds the most it
can come to — each machine on the costliest card of its list, plus a margin, at most its spend cap — against what your
account may still spend this month (other evals' and workers' holds counted), **before any machine starts**. Where it
does not fit — this month's usage would pass your plan's monthly limit, or your credits with no payment method on the
account, or the key's own credit limit — the request returns **402** `insufficient_credit`, saying which, before
creating anything. When the run finishes, the hold is released and every machine that worked — each policy machine a
request reached and each benchmark machine that ran an episode — is billed per second for the time it ran, at least
60 s: its card (the one it actually got), its CPU and its memory, and
`report.cost_usd` is that reconciled amount. The eval's spend cap, `--spend-cap` (or `spend_cap_usd=`), bounds what
one eval may spend: no new
episode starts once the next minute of its machines would pass it.

---

## `seq.eval` — full parameter list

```python
seq.eval(
    policy,                   # the deployed policy's name (or what seq.deploy returned)
    benchmark,                # a registered benchmark's name, or its @seq.benchmark class
    *,
    adapter=None,             # a registered adapter, "name@vN" or "name"; or a @seq.adapter class or "./file.py:Class",
                              # registered first (the same code is the version it already is)
    tasks=None,               # only these tasks (seq.benchmark_tasks(benchmark) lists them)
    trials=None,              # first N declared seeds of each chosen task, or {task: N, "*": M}
    seeds=None,               # instead: these seeds - a list, range(10, 20), "10-19", or {task: seeds, "*": ...}
    policy_machines=1,        # policy machines, each on the deployment's own card
    benchmark_machines=1,     # benchmark machines per arm, each on the card the benchmark was registered on
    sweep=None,               # {"[side.]name": [values]} — an arm per combination
    params=None,              # {"[side.]name": value} — any side's parameters, for the whole eval
                              # ({"eval.max_seconds": 20}: an episode's time limit)
    spend_cap_usd=None,       # the spend cap: it stops once its machines (GPU, CPU and memory) have cost this much
    replay=None,              # "3d" | "none" (default 3d when the benchmark declares replay=)
    dry_run=False,            # True → an EvalEstimate (with .cards); nothing starts, nothing is reserved
    wait=True,                # False → the eval id at once
    timeout_s=None,           # how long to wait for the report
)
```

The CLI and Python post the same request; `--rerun-failed`, `--rerun-unsuccessful` and `--gate` are the CLI's alone.

### A policy that calls the platform's models

If the policy under test calls a model from the catalogue while it acts (declared in `models=`), its policy machines
are given the same short-lived credential a robot's connect gives, and each call is billed to your account like any
chat call, beside the eval's GPU time. A policy with a `@seq.plan` plans in the eval too, on the robot's clock, and the report's
`plan` says how many plans and what their model calls cost. A policy that calls a model service of its own with your key (a secret it declares) pays that service
directly; those calls never pass through the platform.

---

## Worked end-to-end

```bash
pip install sequence-ai
export SEQUENCES_API_KEY=seq_dev_...

seq init toy-policy && seq init toy-benchmark && seq init toy-adapter
seq deploy toy-policy
seq deploy toy-benchmark --name toy-benchmark
seq deploy toy-adapter
seq eval run toy-policy --benchmark toy-benchmark --adapter toy-adapter --dry-run
seq eval run toy-policy --benchmark toy-benchmark --adapter toy-adapter
```
