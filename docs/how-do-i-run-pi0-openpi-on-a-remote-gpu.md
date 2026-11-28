# How do I run π0 (openpi) on a remote GPU?

Write the policy as a class in openpi's environment on the GPU, serve it there with `seq serve`, and connect the robot
with `sequence_ai.connect()`. π0 and π0.5 are the models; openpi is the repository that loads them. openpi's own policy
server and client still work as openpi's guide describes; this package adds nothing to them.

On the GPU, in openpi's environment, with sequence-ai installed beside openpi (`pip install sequence-ai`). openpi's
`download.maybe_download` fetches the `pi05_droid` checkpoint once, from the address openpi's README gives for it, into
`~/.cache/openpi` (`OPENPI_DATA_HOME` moves it); `droid_policy.make_droid_example()` makes a random observation with
DROID's keys and shapes, for the warm-up:

```python
from openpi.policies import droid_policy, policy_config
from openpi.shared import download
from openpi.training import config as openpi_config

import seq


@seq.policy(hz=15, action_chunk_valid_len=15, action_chunk_replan_at=8)
class Pi05Droid_user:
    def __init__(self):
        checkpoint_dir_user = download.maybe_download(pi05_droid_checkpoint_user)   # the address in openpi's README
        self.policy_user = policy_config.create_trained_policy(openpi_config.get_config("pi05_droid"),
                                                               checkpoint_dir_user)

    @seq.sample
    def sample_user(self):     # two 224x224 pictures, 7 joint positions, a gripper, a prompt: an act after each
        return [("observe", droid_policy.make_droid_example()), ("act",)] * 3

    @seq.infer
    def infer_user(self, observation_user):
        return self.policy_user.infer(observation_user)
```

```bash
seq serve                    # in the policy's folder
```

On the robot, the observation keeps openpi's keys, and `act()` returns what openpi's `infer()` returns:

```python
import sequence_ai

policy_user = sequence_ai.connect("http://YOUR_GPU:8000")
# observation["observation/exterior_image_1_left"]: the frame
policy_user.observe(observation_user)
action_chunk_user = policy_user.act()["actions"]
```

```bash
pip install sequence-ai          # on the robot
```

Batched, the acts that wait reach `@seq.infer` together, and `batched(policy)`, in
[`examples/serve_openpi_batched.py`](../examples/serve_openpi_batched.py), passes each observation through the policy's
transforms, as `infer()` does, and samples all of them in one call, each with its own noise:
[How do I use the inference components with openpi, LeRobot or GR00T?](how-do-i-use-the-inference-components-with-openpi-lerobot-or-gr00t.md#openpi-and-lerobot-more-than-one-robot-on-a-gpu)
has the class. The policy's JAX model compiles each listed batch size once, in the warm-up on its sample. A PyTorch
checkpoint runs its observations one by one. Every default of [What is on by
default](what-is-on-by-default.md) applies: the robot sends each 224×224 picture as a video keyframe of its own, and the
server turns it back into the array the policy reads; `connect(url, pictures="exact")` sends exact pixels.

## Tested

- **Verified:** `batched()` of `examples/serve_openpi_batched.py` against stand-ins of openpi's model and transforms:
  one call to the model per batch, each observation's own transforms and output (`tests/engine/test_openpi.py`).

## What openpi already does

openpi's client and server already do several things a remote policy needs:

- **One connection, kept open.** The client connects once and reuses the WebSocket for every `infer()`.
- **Arrays as raw bytes.** Observations and responses are sent as msgpack with numpy arrays as raw bytes.
- **Images resized on the robot.** openpi's guide resizes each image on the client with
  `image_tools.resize_with_pad(img, 224, 224)` before sending it, as an array.
- **Action chunks executed open-loop.** `ActionChunkBroker` requests a new chunk only when the current one is
  exhausted.

If you write your own server or client for an openpi model, `msgpack_io` reads openpi's array layout,
`frames.resize_image(..., mode="pad", filter="bilinear")` is openpi's `resize_with_pad`, byte for byte, and
`frames.decode` turns a picture sent as JPEG back into its array. [The comparison page](comparison.md) sets the two
side by side.

## Where the GPU comes from

Any GPU the robot can reach on port 8000. [How do I get a GPU near my robot?](how-do-i-get-a-gpu-near-my-robot.md)
covers the case where you have none nearby. The robot's code changes in one place: the address it connects to.
