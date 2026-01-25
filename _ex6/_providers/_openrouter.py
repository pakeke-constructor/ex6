
import os
import json
import ex6

import openai
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

from typing import List




def msg_to_dict(m: ex6.Message, ctx: ex6.Context):
    d = {"role": m.role, "content": m.get_msg(ctx)}
    if m.tool_calls:
        d["tool_calls"] = [
            {"id": tc["id"], "type": "function",
             "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])}}
            for tc in m.tool_calls
        ]
    if m.tool_call_id:
        d["tool_call_id"] = m.tool_call_id
    return d


@ex6.override
def invoke_llm(ctx: ex6.Context):
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY", "")
    )

    messages: List[ChatCompletionMessageParam] = [msg_to_dict(m, ctx) for m in ctx.messages]
    tools = ctx.get_tool_schemas()

    stream = client.chat.completions.create(
        model=ctx.model,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
        tools= tools if (not not tools) else None # pyright: ignore
    )

    input_tokens, output_tokens = 0, 0
    finish_reason = "stop"
    tool_calls_acc = {}

    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None

        if delta and delta.content:
            yield ex6.ResponseChunk("text", delta.content)

        # CoT (OpenRouter reasoning field) - check for reasoning in delta
        if delta and hasattr(delta, 'reasoning') and delta.reasoning:
            yield ex6.ResponseChunk("cot", delta.reasoning, len(delta.reasoning))

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
            finish_reason = chunk.choices[0].finish_reason

        if chunk.usage:
            input_tokens = chunk.usage.prompt_tokens
            output_tokens = chunk.usage.completion_tokens

    tool_calls = []
    for tc in tool_calls_acc.values():
        try:
            tc["args"] = json.loads(tc["args"]) if tc["args"] else {}
        except:
            pass
        tool_calls.append(tc)
        # Yield tool call as chunk (full JSON for visibility)
        yield ex6.ResponseChunk("tool", json.dumps(tc))

    yield ex6.LLMResult(input_tokens, output_tokens, tool_calls, finish_reason)

