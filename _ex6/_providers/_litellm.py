import json
import re
import threading
import inspect
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


# ==================== CODE MODE ====================

def extract_tools_block(content: str) -> str | None:
    """Extract ```tools block from content."""
    m = re.search(r'```tools\s*\n(.*?)```', content, re.DOTALL)
    return m.group(1).strip() if m else None


def exec_sandboxed(code: str, env: dict):
    """Execute code. Placeholder for RestrictedPython later."""
    exec(code, env)


def _wrap_tool_threaded(fn, ctx, results: list, threads: list):
    """Wrap tool to run in thread. Appends result dict to results list."""
    def wrapper(*args, **kwargs):
        call_str = f'{fn.__name__}({", ".join(repr(a) for a in args)})'
        result = {"call": call_str, "value": None}
        results.append(result)
        def run():
            result["value"] = fn(ctx, *args, **kwargs)
        t = threading.Thread(target=run)
        t.start()
        threads.append(t)
    return wrapper


def _build_tool_docs(ctx: ex6.Context) -> str:
    """Generate tool documentation for system prompt."""
    tools = ctx.get_tools()
    if not tools:
        return "No tools available."
    lines = ["You have access to tools. To call them, emit a ```tools block:", "```tools"]
    lines.append('read_file("path")  # example')
    lines.append("for f in files:    # loops work too")
    lines.append('    read_file(f)')
    lines.append("```")
    lines.append("")
    lines.append("Available tools:")
    for name, fn in tools.items():
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())[1:]  # skip ctx
        args = ", ".join(p.name for p in params)
        doc = (fn.__doc__ or "").split('\n')[0].strip()
        lines.append(f"  {name}({args}) - {doc}")
    return "\n".join(lines)


tool_system_prompt = ex6.Message(role="system", content=_build_tool_docs)


@ex6.override
def call_tools(ctx: ex6.Context, llm_result: ex6.LLMResult) -> bool:
    # Get last assistant message
    content = ""
    for msg in reversed(ctx.messages):
        if msg.role == "assistant":
            content = msg.content if isinstance(msg.content, str) else ""
            break

    code = extract_tools_block(content)
    if not code:
        # Fall back to native tool calls
        return _call_tools_native(ctx, llm_result)

    # Code mode
    tools = ctx.get_tools()
    results, threads = [], []

    env = {"__builtins__": __builtins__}
    for name, fn in tools.items():
        env[name] = _wrap_tool_threaded(fn, ctx, results, threads)

    exec_sandboxed(code, env)

    for t in threads:
        t.join()

    if results:
        parts = [f"<tool_result {r['call']}>\n{r['value']}\n</tool_result>" for r in results]
        ctx.messages.append(ex6.Message(role="user", content="\n\n".join(parts)))

    return len(results) > 0


def _call_tools_native(ctx: ex6.Context, llm_result: ex6.LLMResult) -> bool:
    """Native tool calling (OpenAI-style tool_calls)."""
    if not llm_result.tool_calls:
        return False

    tools = ctx.get_tools()
    threads, results = [], []

    for tc in llm_result.tool_calls:
        fn = tools.get(tc["name"])
        if not fn:
            continue
        result = {"id": tc["id"], "value": None}
        results.append(result)
        def run_tool(fn=fn, tc=tc, result=result):
            result["value"] = fn(ctx, **tc["args"])
        t = threading.Thread(target=run_tool)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    for r in results:
        ctx.messages.append(ex6.Message(role="tool", content=str(r["value"] or ""), tool_call_id=r["id"]))

    return True