import os
import json
import ex6

import openai
from openai.types.chat import ChatCompletionMessageParam

from typing import List




@ex6.override
def invoke_llm(ctx):
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY", "")
    )

    messages: List[ChatCompletionMessageParam] = [{"role": m.role, "content": m.get_msg(ctx)} for m in ctx.messages]

    stream = client.chat.completions.create(
        model=ctx.model,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True}
    )

    result = {"input_tokens": 0, "output_tokens": 0, "tool_calls": [], "finish_reason": "stop"}
    tool_calls_acc = {}

    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None

        if delta and delta.content:
            yield delta.content

        if delta and delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in tool_calls_acc:
                    tool_calls_acc[idx] = {"id": tc.id, "name": "", "args": ""}
                if tc.function:
                    if tc.function.name:
                        tool_calls_acc[idx]["name"] = tc.function.name
                    if tc.function.arguments:
                        tool_calls_acc[idx]["args"] += tc.function.arguments

        if chunk.choices and chunk.choices[0].finish_reason:
            result["finish_reason"] = chunk.choices[0].finish_reason

        if chunk.usage:
            result["input_tokens"] = chunk.usage.prompt_tokens
            result["output_tokens"] = chunk.usage.completion_tokens

    for tc in tool_calls_acc.values():
        try:
            tc["args"] = json.loads(tc["args"]) if tc["args"] else {}
        except:
            pass
        result["tool_calls"].append(tc)

    yield result
