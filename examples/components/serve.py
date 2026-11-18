# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
serve, alone: a policy server in one call, with batching on, and five robots that act on it at the same time.

    python examples/components/serve.py

On a GPU, sequence_ai.serve(...) is the whole server: here it runs on a thread, so that one file holds both sides.
"""
import socket
import threading
import time

import numpy as np

import sequence_ai

batches = []                                       # the size of each batch the model was called with


def policy_on_a_list(observations: list) -> list:
    """Your model here, on a batch: one forward pass for all the observations waiting, one output for each, in order."""
    time.sleep(0.05)
    batches.append(len(observations))
    return [{"actions": np.full((8, 7), observation["state"][0], dtype=np.float32)} for observation in observations]


example = {"image": np.zeros((224, 224, 3), dtype=np.uint8), "state": np.zeros(7, dtype=np.float32)}
with socket.socket() as probe:                     # a free port on this machine, for this example
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
threading.Thread(target=sequence_ai.serve, args=(policy_on_a_list,), daemon=True,
                 kwargs={"host": "127.0.0.1", "port": port, "batch": [1, 2, 4, 8], "example": example,
                         "log_level": "warning"}).start()

answers, ready = {}, threading.Barrier(5)


def robot(i: int) -> None:
    with sequence_ai.connect(f"http://127.0.0.1:{port}") as link:
        link.wait_until_ready()                    # until the warm-up on the example is done
        ready.wait()                               # the five act at the same moment
        observation = {"image": np.full((224, 224, 3), 100 + i, dtype=np.uint8), "state": np.full(7, float(i))}
        answers[i] = link.act(observation)["actions"]


robots = [threading.Thread(target=robot, args=(i,)) for i in range(5)]
for thread in robots:
    thread.start()
for thread in robots:
    thread.join()
print(f"5 robots, each with its own actions: the first of each {[float(answers[i][0, 0]) for i in range(5)]}")
print(f"the warm-up ran batches of {batches[:8]}; the robots' acts reached the model as batches of {batches[8:]}")
