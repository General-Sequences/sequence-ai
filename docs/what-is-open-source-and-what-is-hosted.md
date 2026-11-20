# What is open source, and what does General Sequences host?

SequenceLink's data plane is open source: the code on the path between a robot and its GPU, which is the policy server
on the GPU, the robot's connection to it, and the components of both, is in this repository under Apache-2.0. It is
the same code wherever the GPU comes from: your own, one rented from General Sequences, or a hosted deployment. The
control plane is General Sequences' hosted service and is not open source: finding a GPU near your robot, starting and
stopping machines, accounts and billing, and building, serving and scoring a hosted policy.

## Open source

Everything a robot and its GPU run, and what a policy is written with:

- **On the GPU:** `seq serve` and the `@seq.policy` decorators a policy is written with, and the inference components
  the server uses, which a server of your own can use as well: `cuda_graphs`, `batcher`, `replicas` and `slack` (`sequence_ai/engine/`).
- **On the robot, and between the two:** `sequence_ai.connect()` and the network components on each end
  (`sequence_ai/network/`), with `openpi` and `lerobot` for the clients and servers of those projects.
- **A policy and an official template, on your own GPU:** the decorators a policy is written with, and `seq init`,
  `seq serve` and `seq weights get`, which run an official template where you choose.
- **Language models:** `sequence_ai.chat()`, with your own key or at a server of your own.
- **Checking and measuring:** `python -m sequence_ai.check`, which says which defaults are on between a robot and its
  server, and `bench/`, which measures what a pause, a ping and a camera stream cost on your link.

A hosted deployment runs the same code: the robot's side of the connection is the same `connect()`, and the GPU's side
is the same server.

## Hosted by General Sequences

What decides where a policy runs, and what it costs:

- **A GPU near your robot.** The closest GPU available to a place you give. How it is chosen is the service's.
- **Machines.** Starting, stopping and billing the GPUs you rent or deploy on.
- **Hosted deployments.** The same policy folder built and served for you, scaled, with benchmarks and evals, and a
  robot driven with it in teleoperation.
- **Your account.** Keys, limits and billing.

The service's client is not in this package: `sequence-cloud` is its own package, and lets `connect()` take a
deployment's name. [generalsequences.com/docs](https://generalsequences.com/docs/) covers the
service.

## On the robot, only what `connect()` is given changes

```python
import sequence_ai

policy_user = sequence_ai.connect("http://your-gpu:8000")   # your own GPU, or one rented from General Sequences
policy_user = sequence_ai.connect("my_policy_user")       # a hosted deployment, by its name, with sequence-cloud
```

## Versions

The version is `MAJOR.MINOR.PATCH`, after [Semantic Versioning](https://semver.org). Before 1.0, a minor version may
change or remove an interface, and the CHANGELOG lists each under Changed or Removed. A patch version changes no
interface.

**The hosted platform may need a newer client.** When the service stops accepting an older `sequence-ai`, the entry
of the version it needs starts with "Upgrade to X.Y.Z, as the platform refuses older versions". Your own `seq serve` and
`connect()` are not held to it.

[RELEASING.md](https://github.com/General-Sequences/sequence-ai/blob/main/RELEASING.md) has what a release is: one
commit, its tag, the package on PyPI and a GitHub release.
