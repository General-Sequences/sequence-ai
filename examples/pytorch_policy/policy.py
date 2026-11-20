# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
A PyTorch model as a policy: @seq.infer turns the robot's observation into the model's inputs, and the tensor the model
returns reaches the robot as a numpy array. On the CPU where there is no GPU.

    pip install sequence-ai torch
    seq serve examples/pytorch_policy               # then, on the robot: python examples/quickstart_robot.py
"""
import numpy as np
import torch

import seq

DEVICE_user = "cuda" if torch.cuda.is_available() else "cpu"


class Model_user(torch.nn.Module):
    """A stand-in for your model (LeRobot's, GR00T's, your own): a camera frame and 7 joints in, 8 actions of 7 out."""

    def __init__(self) -> None:
        super().__init__()
        self.see = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 5, stride=4), torch.nn.ReLU(),
                                       torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
        self.head = torch.nn.Linear(8 + 7, 8 * 7)

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.see(image), state], dim=1)).view(-1, 8, 7)


def load_model_user() -> torch.nn.Module:
    """Your model's weights loaded here."""
    torch.manual_seed(0)
    return Model_user()


observation_user = {"image": np.zeros((224, 224, 3), dtype=np.uint8), "state": np.zeros(7, dtype=np.float32)}


@seq.policy(hz=15, action_chunk_valid_len=8, action_chunk_replan_at=5)
class TorchPolicy_user:
    def __init__(self):
        self.model_user = load_model_user().to(DEVICE_user).eval()

    @seq.sample
    def sample_user(self):
        return [("observe", observation_user), ("act",)] * 3

    @seq.infer
    @torch.inference_mode()
    def infer_user(self, observation_user):
        image_user = torch.as_tensor(observation_user["image"], device=DEVICE_user).permute(2, 0, 1)[None].float() / 255
        state_user = torch.as_tensor(observation_user["state"], device=DEVICE_user)[None]
        return {"actions": self.model_user(image_user, state_user)[0]}   # a tensor: it reaches the robot as numpy
