import json
import ex6
from litellm import completion, completion_cost
from datetime import date



# TODO:
# TODO:
# TODO:
# TODO:
# Store the daily-usage in a file somewhere.
# This isnt actually daily-usage; it is SESSION USAGE.
###
# Claude-code, opencode, cursor, etc, ALL of them use temporary files.
# we should use temp-files too.
# maybe just a function `ex6.get_save_directory()`?



# Daily budget tracking
_daily_cost = 0.0
_daily_limit = 10.0  # default $10/day
_last_reset = date.today()


def set_daily_limit(limit: float):
    global _daily_limit
    _daily_limit = limit


def get_daily_cost() -> float:
    _maybe_reset()
    return _daily_cost


def _maybe_reset():
    global _daily_cost, _last_reset
    today = date.today()
    if today != _last_reset:
        _daily_cost = 0.0
        _last_reset = today


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
    global _daily_cost
    _maybe_reset()

    if _daily_cost >= _daily_limit:
        yield ex6.LLMResult(error=f"daily budget exceeded (${_daily_cost:.2f}/${_daily_limit:.2f})")
        return

    messages = [msg_to_dict(m, ctx) for m in ctx.messages]
    tools = ctx.get_tool_schemas()

    response = completion(
        model=ctx.model,
        messages=messages,
        stream=True,
        tools=tools if tools else None
    )

    input_tokens, output_tokens = 0, 0
    finish_reason = "stop"
    tool_calls_acc = {}

    for chunk in response:
        delta = chunk.choices[0].delta if chunk.choices else None

        if delta and delta.content:
            yield ex6.ResponseChunk("text", delta.content)

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

        if hasattr(chunk, 'usage') and chunk.usage:
            input_tokens = chunk.usage.prompt_tokens or 0
            output_tokens = chunk.usage.completion_tokens or 0

    tool_calls = []
    for tc in tool_calls_acc.values():
        try:
            tc["args"] = json.loads(tc["args"]) if tc["args"] else {}
        except:
            pass
        tool_calls.append(tc)
        yield ex6.ResponseChunk("tool", json.dumps(tc))

    # Calculate cost
    cost = None
    if input_tokens and output_tokens:
        try:
            cost = completion_cost(model=ctx.model, prompt_tokens=input_tokens, completion_tokens=output_tokens)
            _daily_cost += cost
        except:
            pass

    yield ex6.LLMResult(input_tokens, output_tokens, tool_calls, finish_reason, cost=cost)
