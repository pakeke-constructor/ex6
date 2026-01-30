import json
import re
import threading
import inspect
import time
import ex6
from litellm import completion, completion_cost
from datetime import date
from RestrictedPython import compile_restricted


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

    # If code mode prompt is in context, don't pass native tools
    use_code_mode = tool_system_prompt in ctx.messages
    tools = None if use_code_mode else (ctx.get_tool_schemas() or None)

    try:
        response = completion(
            model=ctx.model,
            messages=messages,
            stream=True,
            tools=tools,
            timeout=30,
            request_timeout=30,
        )
    except Exception as e:
        ex6.log.error(f"completion failed: {e}")
        yield ex6.LLMResult(error=str(e))
        return

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


# Sandbox setup
SAFE_BUILTINS = {
    "None": None, "True": True, "False": False,
    "int": int, "float": float, "bool": bool, "complex": complex,
    "abs": abs, "round": round, "pow": pow,
    "list": list, "tuple": tuple, "set": set, "dict": dict, "frozenset": frozenset,
    "range": range, "len": len, "enumerate": enumerate, "zip": zip,
    "min": min, "max": max, "sum": sum, "all": all, "any": any,
    "str": str, "repr": repr, "format": format,
    "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
}

def _no_import(*args, **kwargs):
    raise ImportError("imports disabled")


def exec_sandboxed(code: str, env: dict):
    """Execute code in RestrictedPython sandbox."""
    sandbox_globals = {"__builtins__": SAFE_BUILTINS.copy()}
    sandbox_globals["__import__"] = _no_import
    sandbox_globals.update(env)  # add tools

    byte_code = compile_restricted(code, '<tools>', 'exec')
    if byte_code.errors:
        raise SyntaxError(f"restricted compile: {byte_code.errors}")
    exec(byte_code.code, sandbox_globals)


def _wrap_tool_threaded(fn, ctx, results: list, threads: list):
    """Wrap tool to run in thread. Appends result dict to results list."""
    def wrapper(*args, **kwargs):
        call_str = f'{fn.__name__}({", ".join(repr(a) for a in args)})'
        result = {"call": call_str, "value": None}
        results.append(result)
        def run():
            try:
                result["value"] = fn(ctx, *args, **kwargs)
            except Exception as e:
                ex6.log.error(f"tool {call_str} failed: {e}")
                result["value"] = f"ERROR: {e}"
        t = threading.Thread(target=run)
        t.start()
        threads.append(t)
    return wrapper


def _build_tool_docs(ctx: ex6.Context) -> str:
    """Generate tool documentation for system prompt."""
    tools = ctx.get_tools()
    if not tools:
        return "" # no tools available
    lines = ["# Tools/Functions\nYou have access to tools via code-blocks.\nTo call them, emit a ```tools ``` block, like so:", "```tools"]
    lines.append('read_file("path")  # reads path')
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
    ctx.data["litellm:tool_results"] = results  # expose for renderer

    env = {}
    for name, fn in tools.items():
        env[name] = _wrap_tool_threaded(fn, ctx, results, threads)

    try:
        exec_sandboxed(code, env)
    except Exception as e:
        ex6.log.error(f"code mode exec failed: {e}")

    for t in threads:
        t.join()
    ctx.data.pop("litellm:tool_results", None)

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
            try:
                result["value"] = fn(ctx, **tc["args"])
            except Exception as e:
                ex6.log.error(f"tool {tc['name']} failed: {e}")
                result["value"] = f"ERROR: {e}"
        t = threading.Thread(target=run_tool)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    for r in results:
        ctx.messages.append(ex6.Message(role="tool", content=str(r["value"] or ""), tool_call_id=r["id"]))

    return True



SPINNER = ['/', '-', '\\', '|']

def make_tools_renderer(ctx: ex6.Context) -> ex6.RenderFn:
    def render(buf: ex6.ScreenBuffer, x: int, y: int, w: int) -> int:
        results = ctx.data.get("litellm:tool_results", [])
        if not results:
            return 0
        frame = int(time.time() * 8) % 4
        for i, r in enumerate(results):
            done = r["value"] is not None
            icon = 'x' if done else SPINNER[frame]
            line = f"[{icon}] {r['call']}"[:w]
            for j, ch in enumerate(line):
                buf.put(x + j, y + i, ch, txt_color='red')
        return len(results)
    return render


@ex6.output_renderer
def render_tools_block(output: list[ex6.OutputLine], ctx: ex6.Context) -> None:
    i = 0
    while i < len(output):
        line = output[i]
        if isinstance(line, str) and line.startswith('```tools'):
            j = i + 1
            while j < len(output):
                if isinstance(output[j], str) and output[j].strip() == '```':
                    break
                j += 1
            del output[i:j+1]
            output.insert(i, make_tools_renderer(ctx))
        i += 1


