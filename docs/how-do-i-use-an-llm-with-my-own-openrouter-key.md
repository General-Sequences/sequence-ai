# How do I use an LLM with my own OpenRouter key?

Pass `server="openrouter"` to `sequence_ai.chat()`, with your own OpenRouter key in `OPENROUTER_API_KEY`. The call
goes from your machine to that server and returns the same result as any chat: `.content`, `.tool_calls`,
`.reasoning`, `.usage`.

```bash
pip install sequence-ai
export OPENROUTER_API_KEY=...      # your own OpenRouter key
```

```python
import sequence_ai

result_user = sequence_ai.chat(
    "qwen/qwen3-next-80b-a3b-instruct",                # the model's name on that server
    [{"role": "user", "content": "List the steps to put the red block in the bowl."}],
    server="openrouter",
)
print(result_user.content)
```

`api_key=` passes the key in the call instead of the environment.

## An agent that calls tools

`Conversation` keeps the history, and sends every turn to its `server`:

```python
import json

import sequence_ai

tools_user = [{"type": "function", "function": {
    "name": "move_above", "description": "Move the gripper above an object.",
    "parameters": {"type": "object", "properties": {"object": {"type": "string"}}, "required": ["object"]}}}]


def run_tool_user(call_user):
    arguments_user = json.loads(call_user["function"]["arguments"])
    return {"above": arguments_user["object"]}         # your robot's code here


conversation_user = sequence_ai.Conversation("qwen/qwen3-next-80b-a3b-instruct", server="openrouter",
                                             system="You plan robot tasks.", tools=tools_user)
reply_user = conversation_user.send("Put the red block in the bowl.")
while reply_user.tool_calls:
    for call_user in reply_user.tool_calls:
        conversation_user.tool_result(call_user["id"], run_tool_user(call_user))
    reply_user = conversation_user.send()
print(reply_user.content)
```

## Streaming

`stream=True` yields the text as it is generated; `.result` is the whole reply once the loop ends:

```python
import sequence_ai

stream_user = sequence_ai.chat("qwen/qwen3-next-80b-a3b-instruct", [{"role": "user", "content": "Describe the scene."}],
                               server="openrouter", stream=True)
for piece_user in stream_user:
    print(piece_user, end="", flush=True)
print(stream_user.result.usage)
```

## A server of your own

A server of your own with the OpenAI chat completions API works the same way: pass its address as `server`.

```python
import sequence_ai

result_user = sequence_ai.chat("my-model", [{"role": "user", "content": "Hello"}],
                               server="http://localhost:8000/v1")      # the model as your server names it
```

`api_key` is sent to an address only when you pass it.
