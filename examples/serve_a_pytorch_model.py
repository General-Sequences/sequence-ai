# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
A PyTorch model served in one call: the function serve() is given turns the robot's observation into the model's
inputs, and the tensor the model returns reaches the robot as a numpy array. On the CPU where there is no GPU.

    pip install sequence-ai torch
    python examples/serve_a_pytorch_model.py

On a GPU, sequence_ai.serve(policy, example=example) is the whole server. Here it runs on a thread and a robot acts
on it once, so that one file holds both sides.
"""
import socket
import threading

import numpy as np
import torch

import sequence_ai

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


class Model(torch.nn.Module):
    """A stand-in for your model - LeRobot's, GR00T's, your own: a camera frame and 7 joints in, 8 actions of 7 out."""

    def __init__(self) -> None:
        super().__init__()
        self.see = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 5, stride=4), torch.nn.ReLU(),
                                       torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
        self.head = torch.nn.Linear(8 + 7, 8 * 7)

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.see(image), state], dim=1)).view(-1, 8, 7)


model = Model().to(device).eval()


@torch.inference_mode()
def policy(observation: dict) -> dict:
    """What serve() calls: one observation in, as the robot sent it (numpy arrays), and the robot's answer out."""
    image = torch.as_tensor(observation["image"], device=device).permute(2, 0, 1)[None].float() / 255   # 1x3xHxW
    state = torch.as_tensor(observation["state"], device=device)[None]
    return {"actions": model(image, state)[0]}            # a tensor: serve() sends it as a numpy array


example = {"image": np.zeros((224, 224, 3), dtype=np.uint8), "state": np.zeros(7, dtype=np.float32)}

if __name__ == "__main__":
    with socket.socket() as probe:                        # a free port on this machine, for this example
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    threading.Thread(target=sequence_ai.serve, args=(policy,), daemon=True,
                     kwargs={"host": "127.0.0.1", "port": port, "example": example, "log_level": "warning"}).start()

    with sequence_ai.connect(f"http://127.0.0.1:{port}") as link:
        link.wait_until_ready()
        frame = np.full((224, 224, 3), 128, dtype=np.uint8)
        actions = link.act({"image": frame, "state": np.zeros(7, dtype=np.float32)})["actions"]
    print(f"a {type(actions).__name__} of shape {actions.shape} and dtype {actions.dtype} from the model on the "
          f"{device}: the first {np.round(actions[0, :3], 4)}")
