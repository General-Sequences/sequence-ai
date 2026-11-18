# For coding assistants

`sequence-ai` is an open-source framework for serving robot policies from remote GPUs over wide-area networks. Two parts
of it work with any policy server and are developed in this repository: the inference layer in `sequence_ai/engine/`
and the network layer in `sequence_ai/network/`, independent components of one file each. A change here is almost
always a change to one of those files. [`llms.txt`](llms.txt) maps the files; each module's docstring is its reference.

The rest of the package, `seq/` and the modules of `sequence_ai/` outside those two directories, is the client for GPUs
and deployments hosted by General Sequences. It is developed in General Sequences' own repository and published here, as
it ships, at each release: its files carry no comments and its tests are not here. Do not edit it in a pull request;
propose the change in an issue. [`platform.md`](platform.md) describes how it behaves.

## Set up and test

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"
.venv/bin/python -m pytest -q
.venv/bin/ruff check sequence_ai/engine sequence_ai/network tests examples bench tools
```

The suite is the components' own. It runs in about a minute and needs no network and no GPU: the `cuda_graphs` tests
substitute a fake graph for the GPU's, and the `replicas` tests start real processes. `examples/check_on_a_gpu.py` runs
the capture on a real GPU. One test reads Linux's `TCP_INFO` and skips itself elsewhere, and two are skipped unless
openpi-client is installed (`tests/network/test_openpi.py` and one of `tests/network/test_examples.py`): it needs numpy
below 2, which the rest of the development dependencies do not.

## Layout

- `sequence_ai/engine/<component>.py` and `sequence_ai/network/<component>.py`: one component, one file. Its tests are
  `tests/engine/test_<component>.py` or `tests/network/test_<component>.py`, and its runnable example is
  `examples/components/<component>.py`.
- `sequence_ai/network/link.py`, `sequence_ai/network/openpi.py` and `sequence_ai/engine/serve.py` combine
  components; they are the only files that import other components.
- `bench/`: the tool that measures a link. Standard library only; it imports nothing from the package.
- `examples/`: `quickstart_server.py`, `quickstart_robot.py` and `quickstart_batching.py` are the quickstarts;
  `serve_a_pytorch_model.py` serves a PyTorch model with `serve()`;
  `robot.py` and `server.py` use the network components together; `serve_with_replicas.py` uses three inference
  components together; `fleet_on_a_clock.py` prints what `slack` runs on a simulated clock to compute a capacity.
- `seq/`, `seq_launch.py` and the rest of `sequence_ai/`: the hosted platform's client, as it ships.

## Rules for a change

- **Never change a model's outputs to make it faster.** `cuda_graphs` keeps a graph only when its output equals
  eager mode's byte for byte; a change that weakens that check is not accepted. Nothing in the network layer changes
  a policy's outputs or a machine's settings.
- A component stays one file that imports none of the others. Standard library first; an optional dependency is
  imported where it is used (torch only in `cuda_graphs`).
- A number in a docstring or in the README states what it was measured with: the model and the GPU, or the payload and
  the link's RTT, and how many samples. A number you did not measure is not written. Document what a component does,
  when it applies and what it requires, next to the component; results go in the component's own documentation.
- No silent fallback: a component that cannot do its job reports it.
- No cloud provider's name, internal address or customer's name anywhere.
- A new component includes its test, its example in `examples/components/`, a row in the README's table and a line
  in `llms.txt`.
- Lines are at most 120 columns. Every commit is signed off: `git commit -s`.

[`CONTRIBUTING.md`](CONTRIBUTING.md) has the full rules for a pull request.
