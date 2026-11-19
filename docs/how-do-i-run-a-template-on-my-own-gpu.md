# How do I run a template on my own GPU?

`seq init` the template, install what its image installs, and run `seq serve` in its folder. The policy is built as a
deployment builds it, with the template's own weights, and a robot connects to your GPU's address instead of the
deployment's name.

```bash
pip install sequence-ai
export SEQUENCES_API_KEY=...          # your key: the template's weights are downloaded with it
seq init pi05-libero my_pi05_user
cd my_pi05_user
seq serve                             # on port 8000, until Ctrl-C
```

On the robot, only the address changes:

```python
import sequence_ai

link_user = sequence_ai.connect("http://my-gpu:8000")          # a deployment: sequence_ai.connect("my_pi05_user")
answer_user = link_user.act(observation_user)
```

## What `seq serve` does

- It builds the folder's `@seq.policy` class as a deployment does: its `init=` filled, `{weights}` with the folder its
  weights are in.
- It downloads a template's weights with your key into `~/.cache/sequence-ai/weights/<template>/<version>`, each file
  checked by its sha256. A second run downloads nothing; `seq weights get <template>` downloads them alone.
- It serves the class's `@seq.infer` with `sequence_ai.serve`: a `@seq.concurrent` policy's acts in its listed batch
  sizes, stacked into one batch as on the platform.
- It runs the class's `@seq.shutdown` when you stop it.

## Install what the image installs

The template's `policy.py` names its image: a base, the packages `requirements-policy.lock` pins, and the steps after
them. Install the same on your machine before `seq serve`. For pi05-libero, on a Linux machine with an NVIDIA GPU:

```bash
uv venv --python 3.11 && . .venv/bin/activate
uv pip install sequence-ai
seq init pi05-libero my_pi05_user && cd my_pi05_user
uv pip install --no-deps --require-hashes -r requirements-policy.lock       # the packages its image installs
uv pip install --no-deps -e third_party/openpi/packages/openpi-client -e third_party/openpi
uv pip install --no-deps "lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
python -c "import openpi.policies.policy_config, openpi_client, lerobot, cv2"  # the check its frozen.toml names
seq serve
```

On an L4 the packages installed in under a minute and the 12.4 GB of weights downloaded in about a minute. The server
is ready when `GET /ping` answers 200.

## Try it on a CPU

toy-policy has no weights and runs on a CPU, so the whole loop runs on one laptop:

```bash
seq init toy-policy toy_user
cd toy_user
seq serve --port 8000
```

```python
import sequence_ai

link_user = sequence_ai.connect("http://127.0.0.1:8000")
print(link_user.act({"state": [0.0], "target": [0.5]}))        # {'delta': [0.1]}
```

## Your own weights

`--weights <folder>` reads a folder instead of downloading, such as your own fine-tune of the template's checkpoint. A
policy whose weights come in named parts takes `--weights <part>=<folder>` for each.

`seq serve` serves a policy whose `@seq.infer` takes the observation. A policy that keeps each episode's state
(`@seq.observe`) runs as a deployment, which hands it each robot's episode.
