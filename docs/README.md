# sequence-ai documentation

`sequence-ai` is an open-source SDK for serving robot policies from remote GPUs over wide-area networks. It has an
inference layer (`sequence_ai.engine`) and a network layer (`sequence_ai.network`), which work with any policy server,
and the client and commands for GPUs rented from General Sequences. Each page here answers one question, starts with
the answer and its code, and can be read independently.

## Start

- [Quickstart: a policy server and a robot](quickstart.md). Three commands, two short files, every step's output.
- [Quickstart: batching and replicas](quickstart-inference.md). Two commands, one short file, every step's output.
- [FAQ: errors you may see, and what to do next](faq.md). Each error as the code prints it.

## Three ways to use it

| You have | Read | Runnable example |
|---|---|---|
| Your own GPU, and a model | [How do I serve a robot policy from my own GPU?](how-do-i-serve-a-robot-policy-from-my-own-gpu.md), then [How many robots can one GPU serve?](how-many-robots-can-one-gpu-serve.md) | [`quickstart_server.py`](../examples/quickstart_server.py), [`quickstart_robot.py`](../examples/quickstart_robot.py), [`quickstart_batching.py`](../examples/quickstart_batching.py) |
| openpi (π0, π0.5) | [How do I run π0 (openpi) on a remote GPU?](how-do-i-run-pi0-openpi-on-a-remote-gpu.md) | [`quickstart_openpi.py`](../examples/quickstart_openpi.py) |
| LeRobot | [How do I reduce the latency of LeRobot remote inference?](how-do-i-reduce-the-latency-of-lerobot-remote-inference.md) | [`quickstart_lerobot.py`](../examples/quickstart_lerobot.py) |
| GR00T, or another PyTorch policy | [How do I make a PyTorch robot policy faster without changing its outputs?](how-do-i-make-a-pytorch-robot-policy-faster-without-changing-its-outputs.md) | [`graph_a_module.py`](../examples/graph_a_module.py) |
| No GPU near the robot | [How do I get a GPU near my robot?](how-do-i-get-a-gpu-near-my-robot.md) | the same files, with that GPU's address |

## Inference layer: `sequence_ai.engine`

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
- [What do the network components save, measured with the tools in this repository?](measurements.md)
- [How do the components compare with openpi's remote inference and LeRobot's?](comparison.md)
- [What does each network component do?](components.md)

## A GPU near the robot

- [How do I get a GPU near my robot?](how-do-i-get-a-gpu-near-my-robot.md). Any GPU the robot can reach works, and
  the robot's code changes in one line.
- [The platform guide](../platform.md): a deployment hosted by General Sequences, from writing a policy class to driving
  a robot with it, with its references for [benchmarks and evals](../eval.md) and
  [debugging a deployment](../observability.md).

## Measurements

Each measurement in these pages is given with the model, the GPU or the link it was taken on. The
[README](../README.md#results) has the results in one table and the [limitations](../README.md#limitations) in one
list, and [`bench/`](../bench/) measures your own link.

## Elsewhere

- [General Sequences](https://generalsequences.com): GPUs rented by location, a paid service.
- [`llms.txt`](../llms.txt) and [`llms-full.txt`](../llms-full.txt): this repository's map and its documentation in
  one file, for a coding assistant.
