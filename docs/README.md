# SequenceLink documentation

SequenceLink is an open-source framework for serving robot policies from remote GPUs over wide-area networks: on the
GPU, a policy written as a class and `seq serve`; on the robot, `sequence_ai.connect(address)`. Its inference layer
(`sequence_ai.engine`) and network layer (`sequence_ai.network`) also work in a policy server you already run. You can
serve a policy on a GPU of your own, or deploy it to [General Sequences](https://generalsequences.com/docs/deploy/).
Each page here answers one question, starts with the answer and its code, and can be read independently.

## Start

- [What is open source, and what does General Sequences host?](what-is-open-source-and-what-is-hosted.md).
  What runs on your machines and what General Sequences runs, and how versions change.
- [Quickstart: a policy and a robot](quickstart.md). A policy as a class, `seq serve`, and the robot's own loop, every
  step's output.
- [What is on by default, and how do I turn each part off?](what-is-on-by-default.md). Both sides, every part, and the
  command that says which are on.
- [Parts for a server you already run](parts-for-a-server-you-already-run.md). `cuda_graphs`, batching, `slack` and
  replicas, and the openpi and LeRobot clients, a file each.
- [Quickstart: batching and replicas](quickstart-inference.md). Two commands, one short file, every step's output.
- [FAQ: errors you may see, and what to do next](faq.md). Each error as the code prints it.

## By model

The code is the same wherever the GPU comes from: `seq serve` on it, `connect(...)` on the robot.

| You have | Read | Runnable example |
|---|---|---|
| A model of your own, or any PyTorch policy | [How do I serve a robot policy from my own GPU?](how-do-i-serve-a-robot-policy-from-my-own-gpu.md), then [How many robots can one GPU serve?](how-many-robots-can-one-gpu-serve.md) | [`quickstart_policy/policy.py`](../examples/quickstart_policy/policy.py), [`quickstart_robot.py`](../examples/quickstart_robot.py), [`pytorch_policy/policy.py`](../examples/pytorch_policy/policy.py) |
| openpi (π0, π0.5) | [How do I run π0 (openpi) on a remote GPU?](how-do-i-run-pi0-openpi-on-a-remote-gpu.md) | [`quickstart_openpi.py`](../examples/quickstart_openpi.py) |
| LeRobot | [How do I reduce the latency of LeRobot remote inference?](how-do-i-reduce-the-latency-of-lerobot-remote-inference.md) | [`quickstart_lerobot.py`](../examples/quickstart_lerobot.py) |
| GR00T, or another PyTorch policy | [How do I make a PyTorch robot policy faster without changing its outputs?](how-do-i-make-a-pytorch-robot-policy-faster-without-changing-its-outputs.md) | [`graph_a_module.py`](../examples/graph_a_module.py) |
| Octo, or another JAX policy | [How do I serve my own model?](how-do-i-serve-my-own-model.md) | its code, on that page |
| No GPU near the robot | [How do I get a GPU near my robot?](how-do-i-get-a-gpu-near-my-robot.md) | the same files, with that GPU's address |

## Inference layer: `sequence_ai.engine`

- [How do I serve a robot policy from my own GPU?](how-do-i-serve-a-robot-policy-from-my-own-gpu.md): a policy as a
  class and `seq serve`, and any model in it
- [How do I make a policy faster without changing its outputs, and what does each switch change?](how-do-i-make-a-policy-faster-without-changing-its-outputs.md):
  what `seq serve` tries, and each switch
- [How do I load a policy's weights faster?](how-do-i-load-a-policys-weights-faster.md)
- [How do I make a PyTorch robot policy faster without changing its outputs?](how-do-i-make-a-pytorch-robot-policy-faster-without-changing-its-outputs.md)
- [How many robots can one GPU serve?](how-many-robots-can-one-gpu-serve.md)
- [Should I run two replicas of a policy on one GPU?](should-i-run-two-replicas-of-a-policy-on-one-gpu.md)
- [How do I use the inference components with openpi, LeRobot or GR00T?](how-do-i-use-the-inference-components-with-openpi-lerobot-or-gr00t.md)
- [`slack`: scheduling by the slack an action chunk leaves](slack.md)

## Network layer: `sequence_ai.network`

- [How do I serve a robot policy from my own GPU?](how-do-i-serve-a-robot-policy-from-my-own-gpu.md)
- [Why does my robot's upload get slow after a pause?](why-does-my-robots-upload-get-slow-after-a-pause.md)
- [How do I measure the link between my robot and my GPU?](how-do-i-measure-the-link-between-my-robot-and-my-gpu.md)
- [How do I keep a WebSocket policy server's connection warm?](how-do-i-keep-a-websocket-policy-servers-connection-warm.md)
- [How do I send each act over two uplinks?](how-do-i-send-each-act-over-two-uplinks.md)
- [What do the network components save, measured with the tools in this repository?](measurements.md)
- [How do the components compare with openpi's remote inference and LeRobot's?](comparison.md)
- [What does each network component do?](components.md)

## Language models: `sequence_ai.chat`

- [How do I use an LLM with my own OpenRouter key?](how-do-i-use-an-llm-with-my-own-openrouter-key.md):
  `chat(server="openrouter")` with your own OpenRouter key, or a server of your own with the OpenAI chat
  completions API

## A GPU near the robot

- [How do I get a GPU near my robot?](how-do-i-get-a-gpu-near-my-robot.md). Any GPU the robot can reach works, and
  the robot's code changes in one line.
- [How do I run a template on my own GPU?](how-do-i-run-a-template-on-my-own-gpu.md). `seq serve` in the folder `seq init`
  made: the policy built as a deployment builds it, the template's weights downloaded with your key.

## Measurements

Each measurement in these pages is given with the model, the GPU or the link it was taken on. The
[README](../README.md#results) has the results in one table, and [`bench/`](../bench/) measures your own link.

## Elsewhere

- [`llms.txt`](../llms.txt) and [`llms-full.txt`](../llms-full.txt): this repository's map and its documentation in
  one file, for a coding assistant.
