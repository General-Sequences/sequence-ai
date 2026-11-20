# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
Quickstart, GPU side: a policy as a class. Put your model where `infer_user` returns.

    seq serve examples/quickstart_policy        # every address of this machine, port 8000

@seq.policy says how its answers play: hz, the steps a second; action_chunk_valid_len, the steps of an answer the robot
plays; action_chunk_replan_at, the step at which it asks for the next. @seq.sample is an episode as the robot plays it,
which the server warms the model up on before it says it is ready. @seq.infer answers the latest observation with a
chunk of actions, its first dimension time. docs/quickstart.md walks through it with examples/quickstart_robot.py.
"""
import numpy as np
import seq


@seq.policy(hz=15, action_chunk_valid_len=8, action_chunk_replan_at=5)
class QuickstartPolicy_user:
    @seq.sample
    def sample_user(self):
        observation_user = {"image": np.zeros((224, 224, 3), dtype=np.uint8), "state": np.zeros(7, dtype=np.float32)}
        return [("observe", observation_user), ("act",)] * 3          # an observation, then an act, three times

    @seq.infer
    def infer_user(self, observation_user):
        frame_user = observation_user["image"]                       # your model here: the next 10 steps, 7 joints each
        return {"actions": np.full((10, 7), float(frame_user.mean()), dtype=np.float32)}
