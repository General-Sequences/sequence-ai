# How do I serve my own model?

`serve()` wraps any policy whose call returns an action chunk; nothing in the model changes.

The model is loaded and called the way its own README does it, inside a function of one observation, and `serve()`
serves that function. Below is Octo (`octo-base-1.5`, under JAX), written from Octo's README alone.

## On the GPU

Install Octo as its README says, and `pip install sequence-ai` beside it. Then:

```python
import jax
import numpy as np
from PIL import Image
from octo.model.octo_model import OctoModel

import sequence_ai

model_user = OctoModel.load_pretrained("hf://rail-berkeley/octo-base-1.5")      # as Octo's README loads it
calls_user = [0]


def policy_user(observation_user):
    calls_user[0] += 1
    image_user = np.asarray(Image.fromarray(observation_user["image"]).resize((256, 256)))   # this model's size
    task_user = model_user.create_tasks(texts=[observation_user["prompt"]])
    actions_user = model_user.sample_actions(
        {"image_primary": image_user[None, None], "timestep_pad_mask": np.array([[True]])},  # a history of one
        task_user, rng=jax.random.PRNGKey(calls_user[0]))
    return {"actions": np.asarray(actions_user, dtype=np.float32)[0]}                 # a chunk: 4 steps of 7


example_user = {"image": np.zeros((224, 224, 3), np.uint8), "prompt": "pick up the black block and put it in the box"}
sequence_ai.serve(policy_user, example=example_user)
```

`example=` is run before the server answers its first robot, so JAX compiles the model then, not on a robot's first
request. The observation is the one Octo's README names: `image_primary` with a history of one, and
`timestep_pad_mask`. The actions are what `sample_actions` returns.

## On the robot

```python
import sequence_ai

robot_user = sequence_ai.connect("http://YOUR_GPU:8000")
robot_user.wait_until_ready()
prompt_user = "pick up the black block and put it in the box"
chunk_user = robot_user.act({"image": frame_user, "prompt": prompt_user})["actions"]      # frame_user: the camera's
```

`chunk_user` is the 4 x 7 chunk the policy returned. The picture goes as JPEG and comes back to the policy as the
array it reads, as [What is on by default](what-is-on-by-default.md) says.

## Return an action chunk

The robot plays a chunk while it asks for the next one, so a round trip can pass while the arm moves. A policy that
returns one step at a time leaves the arm waiting for each step:
[How do I serve a robot policy from my own GPU?](how-do-i-serve-a-robot-policy-from-my-own-gpu.md#your-model-as-a-function)

## A PyTorch model

The same function, with the model's PyTorch call inside it:
[How do I serve a robot policy from my own GPU?](how-do-i-serve-a-robot-policy-from-my-own-gpu.md#your-model-as-a-function)
has a PyTorch module, [How do I make a PyTorch robot policy faster without changing its
outputs?](how-do-i-make-a-pytorch-robot-policy-faster-without-changing-its-outputs.md) has GR00T, and [How do I
reduce the latency of LeRobot remote inference?](how-do-i-reduce-the-latency-of-lerobot-remote-inference.md) has
LeRobot's policies.
