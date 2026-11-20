# Contributing to SequenceLink

Thank you for taking the time. This page describes how to set up, what a change needs before it is merged, and how the
code here is written.

## What is developed here

Two parts of the package work with any policy server, and are developed in this repository: the inference layer
(`sequence_ai/engine/`) and the network layer (`sequence_ai/network/`), with their tests, `examples/`, `bench/` and
`docs/`. Pull requests to these are reviewed and merged here.

The rest of the package, `seq/` and the other modules of `sequence_ai/`, holds what a policy is written with, `seq init`,
`seq serve` and `seq weights get`, `chat()`, `check` and the code under the connection. It is developed in General
Sequences' own repository and published here, as it ships, at each release, without its comments or its tests. The
client of General Sequences' hosted service, `sequence-cloud`, is not in this repository. A pull request to those files cannot be merged as it is: open an issue, and
the change is made upstream and arrives with the next release.

Each release publishes the whole tree again in one commit, tagged with its version, the merged changes in it.

## Ways to help

- **Report a bug**, with a minimal script that reproduces it.
- **Propose a change.** For anything larger than a small fix, open an issue first, so that the design is agreed before
  the code is written.
- **Send a pull request**: a fix, a test, a clearer docstring.
- **Measure a link or a model.** A figure from a network or a model we have not measured, with how it was taken, is a
  contribution.

## Set up

```bash
git clone https://github.com/General-Sequences/sequence-ai.git
cd sequence-ai
uv venv --python 3.11
uv pip install -e ".[dev]"
```

Python 3.9 or later: the tests run on 3.9, 3.10, 3.11 and 3.12. We use `uv`; `python -m venv` and `pip` work as well.

## Run the tests

```bash
.venv/bin/python -m pytest -q                                  # all of them, under a minute
.venv/bin/python -m pytest tests/network/test_window.py -q     # one component's
.venv/bin/ruff check sequence_ai/engine sequence_ai/network tests examples bench tools    # the lint
```

The tests run on a machine with no GPU: the `cuda_graphs` tests substitute a fake graph for the GPU's, and the
`replicas` tests start real processes. One test reads Linux's `TCP_INFO` and skips itself elsewhere.

## Before you open a pull request

- **One change.** A fix and an unrelated clean-up are two pull requests.
- **Tests.** They pass, new behaviour has a test, and a fixed bug has a test that failed before the fix.
- **Lint.** `ruff check` on the paths above passes.
- **Changelog.** Say in the pull request what a user would notice. A maintainer writes the line in
  [CHANGELOG.md](CHANGELOG.md), whose entries are one sentence and at most three items per release.
- **Sign-off.** Every commit is signed off: `git commit -s` adds the `Signed-off-by` line that certifies the
  [Developer Certificate of Origin](https://developercertificate.org).
- **Numbers.** A figure stated in a docstring or in the README states how it was measured: the model and the GPU, or
  the two ends and the RTT between them, the size sent, and how many samples. A change in speed includes the figure
  before and after.

## How the code is written

- **A component is self-contained.** One file where possible, no import of another component unless its docstring
  states it, and an example that adds it to code outside this package.
- **A module opens with a docstring**: what it is, "In one sentence" the path a call takes through it, and why it is
  built this way, with the measurement behind the decision.
- **Comments explain why**, not what the next line does.
- **A file stays under about 1500 lines.** One that grows past it is split by responsibility.
- **No silent fallback.** A path that cannot do its work raises an error that states why.
- **Outputs do not change for speed.** A faster path is kept only when its output is identical to the slower one's,
  and a test verifies it.
- **Nothing on the robot is changed without its owner.** The client reads a machine's settings and reports what to
  change; it does not change them.
- **No names from outside the project**: no cloud provider's name, internal address or customer's name in code,
  comments or tests.

## Review

A maintainer reviews every pull request ([CODEOWNERS](.github/CODEOWNERS)). The lint, the tests and a build of the
package run on every pull request ([tests.yml](.github/workflows/tests.yml)) and must pass before a merge. Pull
requests are squashed when merged. How a version is released: [RELEASING.md](RELEASING.md).

## License

By contributing you agree that your contribution is licensed under [Apache-2.0](LICENSE). Your sign-off certifies that
you have the right to submit it.
