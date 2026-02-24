import json
import re
import threading
import inspect
import time
import ex6
import openai, os
from datetime import date
from RestrictedPython import compile_restricted_exec
from dataclasses import dataclass


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

@dataclass
class ModelInfo:
    input: float # cost / Mtok
    output: float # cost / Mtok
    output_cached: float # cost / Mtok


# $/M tokens
COSTS = {
    "openai/gpt-4o":                    ModelInfo(2.5, 10, 1.25),
    "openai/gpt-4.1":                   ModelInfo(2, 8, 0.5),
    "openai/gpt-4.1-mini":              ModelInfo(0.4, 1.6, 0.1),
    "openai/gpt-4.1-nano":              ModelInfo(0.1, 0.4, 0.025),
    "openai/o3":                        ModelInfo(2, 8, 0.5),
    "openai/o4-mini":                   ModelInfo(1.1, 4.4, 0.275),
    "anthropic/claude-sonnet-4":        ModelInfo(3, 15, 0.3),
    "anthropic/claude-haiku-4":         ModelInfo(0.8, 4, 0.08),
    "anthropic/claude-opus-4":          ModelInfo(15, 75, 1.5),
    "google/gemini-2.5-pro-preview":    ModelInfo(1.25, 10, 0.315),
    "google/gemini-2.5-flash-preview":  ModelInfo(0.15, 0.6, 0.0375),
}


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

    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
    )

    try:
        response = client.chat.completions.create(
            model=ctx.model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            tools=tools,
            timeout=30,
        )
    except Exception as e:
        ex6.debug_print(f"completion failed: {e}")
        yield ex6.LLMResult(error=str(e))
        return

    input_tokens, output_tokens = 0, 0
    finish_reason = "stop"
    tool_calls_acc = {}

    for chunk in response:
        delta = chunk.choices[0].delta if chunk.choices else None

        if delta and delta.content:
            yield ex6.ResponseChunk("text", delta.content)

        # CoT (OpenRouter reasoning field)
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
    if ctx.model not in COSTS:
        raise ValueError(f"no pricing for model '{ctx.model}' — add it to COSTS in provider.py")
    info = COSTS[ctx.model]
    cost = (input_tokens * info.input + output_tokens * info.output) / 1_000_000
    _daily_cost += cost

    result = ex6.LLMResult(input_tokens, output_tokens, tool_calls, finish_reason, cost=cost)
    _log_invoke(ctx, messages, result)
    yield result


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

    result = compile_restricted_exec(code, '<tools>')
    if result.errors:
        raise SyntaxError(f"restricted compile: {result.errors}")
    exec(result.code, sandbox_globals)


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
                ex6.debug_print(f"tool {call_str} failed: {e}")
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
    lines = [
    "# Tools/Functions",
    "You have access to tools via tool-blocks.",
    "tool-blocks are sandboxed python scripts, with a bunch of functions for you to use.",
    "To call them, emit a ```tools ``` block-",
    "",
    "## EXAMPLE:",
    "User: Can you read the files I talked about?",
    "Assistant: Let me read the files:",
    "```tools",
    ]
    lines.append('read_file("path")')
    lines.append("for f in files:")
    lines.append('    read_file(f)')
    lines.append("```")
    lines.append(" ================ ")
    lines.append("# Available Tools:")
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
        ex6.debug_print(f"code mode exec failed: {e}")

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
                ex6.debug_print(f"tool {tc['name']} failed: {e}")
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

def make_tools_renderer(code: str, ctx: ex6.Context) -> ex6.RenderFn:
    # Parse code lines as tool calls to display
    lines = [ln.strip() for ln in code.strip().split('\n') if ln.strip()]
    def render(buf: ex6.ScreenBuffer, x: int, y: int, w: int) -> int:
        frame = int(time.time() * 8) % 4
        running = ctx.llm_suspended  # tools are running
        icon = SPINNER[frame] if running else '...'
        for i, call in enumerate(lines):
            buf.puts(x, y + i, f"[{icon}]", txt_color='yellow', style='bold')
            buf.puts(x + 6, y + i, call[:w-4], txt_color='blue')
        return len(lines)
    return render


@ex6.output_renderer
def render_tools_block(output: list[ex6.OutputLine], ctx: ex6.Context) -> None:
    i = 0
    while i < len(output):
        line = output[i]
        if isinstance(line, tuple) and line[1].startswith('```tools'):
            role = line[0]
            j = i + 1
            code_lines = []
            while j < len(output):
                ln = output[j]
                # stop at role boundary (unclosed block)
                if isinstance(ln, tuple) and ln[0] != role:
                    j -= 1  # don't consume the boundary line
                    break
                if isinstance(ln, tuple) and ln[1].strip() == '```':
                    break
                if isinstance(ln, tuple):
                    code_lines.append(ln[1])
                j += 1
            code = '\n'.join(code_lines)
            del output[i:j+1]
            output.insert(i, make_tools_renderer(code, ctx))
        i += 1


def make_tool_result_renderer(call: str, content: str) -> ex6.RenderFn:
    is_error = content.startswith("ERROR:")
    preview = content[:50].replace('\n', ' ')
    def render(buf: ex6.ScreenBuffer, x: int, y: int, w: int) -> int:
        if is_error:
            buf.puts(x, y, "[x]", txt_color='red', style='bold')
        else:
            buf.puts(x, y, "[v]", txt_color='green', style='bold')
        buf.puts(x + 4, y, call, txt_color='blue')
        buf.puts(x + 5 + len(call), y, preview[:w - 6 - len(call)], txt_color='bright_black')
        return 1
    return render


@ex6.output_renderer
def render_tool_results(output: list[ex6.OutputLine], ctx: ex6.Context) -> None:
    i = 0
    while i < len(output):
        line = output[i]
        if isinstance(line, tuple) and line[1].startswith('<tool_result '):
            # Parse: <tool_result call_here>
            call = line[1][13:].rstrip('>')
            # Collect content until </tool_result>
            j = i + 1
            content_lines = []
            while j < len(output):
                if isinstance(output[j], tuple) and output[j][1].strip() == '</tool_result>':
                    break
                if isinstance(output[j], tuple):
                    content_lines.append(output[j][1])
                j += 1
            content = '\n'.join(content_lines)
            del output[i:j+1]
            output.insert(i, make_tool_result_renderer(call, content))
        i += 1


def _log_invoke(ctx, messages, result):
    from datetime import datetime
    import random
    folder = ex6.get_folder() / "logs"
    folder.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    fname = f"invoke-{ts}-{random.randint(1000,9999)}.txt"

    lines = [
        "============",
        f"model: {ctx.model}",
        f"input_tokens: {result.input_tokens}",
        f"output_tokens: {result.output_tokens}",
        f"cost: ${result.cost:.4f}" if result.cost else "cost: N/A",
        f"finish_reason: {result.finish_reason}",
        "============",
        "",
        "=== CONTEXT ===",
    ]
    for m in messages:
        lines.append(f"[{m['role']}]")
        lines.append(m['content'])
        lines.append("")

    (folder / fname).write_text("\n".join(lines), encoding="utf-8")


