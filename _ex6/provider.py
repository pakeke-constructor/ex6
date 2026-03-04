import json
import re
import threading
import inspect
import time
import ex6
import openai
import os
from datetime import date
from RestrictedPython import compile_restricted_exec
from dataclasses import dataclass
from typing import Optional


# Daily budget tracking
_daily_cost: Optional[float] = None  # None = not yet loaded from disk
_daily_limit: float = 10.0  # default $10/day
_last_reset = date.today()
_cost_lock = threading.Lock()

@dataclass
class ModelInfo:
    context_window: int # ctx-window size
    input: float # cost / Mtok
    output: float # cost / Mtok
    cache_read: float # cost / Mtok
    cache_write: float = 0 # cost / Mtok (explicit caching, e.g. Anthropic)


# $/M tokens (input, output, cache_read, cache_write)
MODELS = {
    # --- Anthropic ---
    "anthropic/claude-opus-4.6":         ModelInfo(200_000, 5, 25, 0.5, 6.25),
    "anthropic/claude-sonnet-4.6":       ModelInfo(200_000, 3, 15, 0.3, 3.75),
    "anthropic/claude-sonnet-4":         ModelInfo(200_000, 3, 15, 0.3, 3.75),
    "anthropic/claude-haiku-4":          ModelInfo(200_000, 0.8, 4, 0.08, 1),
    # --- OpenAI ---
    "openai/gpt-5":                      ModelInfo(400_000, 1.25, 10, 0.125),
    "openai/gpt-5-mini":                 ModelInfo(400_000, 0.25, 2, 0.025),
    "openai/gpt-5-codex":               ModelInfo(400_000, 1.25, 10, 0.125),
    "openai/gpt-5.2-codex":             ModelInfo(400_000, 1.75, 14, 0.175),
    "openai/gpt-5.1-codex-mini":         ModelInfo(400_000, 0.25, 2, 0.025),
    "openai/codex-mini":                 ModelInfo(200_000, 1.5, 6, 0.375),
    "openai/o4-mini":                    ModelInfo(200_000, 1.1, 4.4, 0.275),
    # --- Google ---
    "google/gemini-3-pro-preview":       ModelInfo(1_048_576, 2, 12, 0.2),
    "google/gemini-3-flash-preview":     ModelInfo(1_048_576, 0.5, 3, 0.05),
    "google/gemini-2.5-pro":             ModelInfo(1_048_576, 1.25, 10, 0.125),
    "google/gemini-2.5-flash":           ModelInfo(1_048_576, 0.3, 2.5, 0.03),
    "google/gemini-2.5-flash-lite":      ModelInfo(1_048_576, 0.1, 0.4, 0.01),
    # --- xAI ---
    "x-ai/grok-4":                       ModelInfo(256_000, 3, 15, 0.75),
    "x-ai/grok-4-fast":                  ModelInfo(2_000_000, 0.2, 0.5, 0.05),
    # --- DeepSeek ---
    "deepseek/deepseek-chat-v3.1":       ModelInfo(128_000, 0.15, 0.75, 0),
    "deepseek/deepseek-r1":              ModelInfo(128_000, 0.7, 2.5, 0),
    # --- Other ---
    "qwen/qwen3-coder":                  ModelInfo(262_144, 0.22, 1, 0.022),
    "moonshotai/kimi-k2.5":              ModelInfo(262_144, 0.45, 2.2, 0.225),
}


def set_daily_limit(limit: float):
    global _daily_limit
    _daily_limit = limit


def is_over_budget() -> bool:
    global _daily_cost, _last_reset
    with _cost_lock:
        today = date.today()
        if _daily_cost is None:
            try:
                data = json.loads((ex6.get_folder() / "usage.json").read_text())
                if data.get("date") == str(today):
                    _daily_cost = data.get("cost", 0.0)
            except: pass
            if _daily_cost is None:
                _daily_cost = 0.0
        if today != _last_reset:
            _daily_cost = 0.0
            _last_reset = today
        return _daily_cost >= _daily_limit


def increment_cost(cost: float):
    global _daily_cost
    with _cost_lock:
        _daily_cost += cost
        try:
            (ex6.get_folder() / "usage.json").write_text(
                json.dumps({"date": str(date.today()), "cost": _daily_cost}))
        except: pass


_cached_contexts = {}  # id(ctx) -> ttl

def cache_manually(ctx: ex6.Context, ttl="1h"):
    """Mark a context for long-lived system/tool caching. Idempotent."""
    assert ctx.model.startswith("anthropic/"), "Manual caching only works on anthropic-models"
    if id(ctx) in _cached_contexts:
        return
    _cached_contexts[id(ctx)] = ttl


def _apply_cache_control(content: str | list[dict], cc: dict) -> list[dict]:
    """Add cache_control to a message content field. Returns array-format content."""
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    if isinstance(content, list) and content:
        content[-1]["cache_control"] = cc
    return content


def msg_to_dict(m: ex6.Message, ctx: ex6.Context):
    d: dict = {"role": m.role, "content": m.get_msg(ctx)}
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
    if is_over_budget():
        yield ex6.LLMResult(error=f"daily budget exceeded (${_daily_cost:.2f}/${_daily_limit:.2f})")
        return

    messages = [msg_to_dict(m, ctx) for m in ctx.messages]

    # If code mode prompt is in context, don't pass native tools
    use_code_mode = tool_system_prompt in ctx.messages
    tools = None if use_code_mode else (ctx.get_tool_schemas() or None)

    # Anthropic prompt caching
    if ctx.model.startswith("anthropic/"):
        # 1h cache on last system message + tools (if ctx was registered via cache())
        if id(ctx) in _cached_contexts:
            ttl = _cached_contexts[id(ctx)]
            cc = {"type": "ephemeral", "ttl": ttl}
            # Only cache last system msg (prefix-based, covers everything before it)
            for msg in reversed(messages):
                if msg["role"] == "system":
                    msg["content"] = _apply_cache_control(msg["content"], cc)
                    break
            if tools:
                tools[-1]["function"]["cache_control"] = cc
        # Ephemeral breakpoint on second-to-last message (conversation prefix)
        # Skip if it already has a cache_control (e.g. 1h from above)
        if len(messages) >= 2:
            target = messages[-2]
            content = target["content"]
            already_cached = (isinstance(content, list) and content
                              and "cache_control" in content[-1])
            if not already_cached:
                target["content"] = _apply_cache_control(
                    content, {"type": "ephemeral"})

    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
    )

    ex6.debug_print(f"[invoke] model={ctx.model} msgs={len(messages)} code_mode={use_code_mode}")
    try:
        response = client.chat.completions.create( # type: ignore[arg-type]
            model=ctx.model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            tools=tools,
            timeout=30,
        )
    except Exception as e:
        ex6.debug_print(f"[invoke] API EXCEPTION: {e}")
        yield ex6.LLMResult(error=str(e))
        return

    ex6.debug_print("[invoke] stream started")
    input_tokens, output_tokens, cached_tokens = 0, 0, 0
    finish_reason = "stop"
    tool_calls_acc = {}
    chunk_count = 0

    for chunk in response:
        chunk_count += 1
        delta = chunk.choices[0].delta if chunk.choices else None

        if delta and delta.content:
            yield ex6.ResponseChunk("text", delta.content)

        # CoT (OpenRouter reasoning field)
        reasoning = getattr(delta, 'reasoning', None) if delta else None
        if reasoning:
            yield ex6.ResponseChunk("cot", reasoning, len(reasoning))

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
            details = getattr(chunk.usage, 'prompt_tokens_details', None)
            if details:
                cached_tokens = getattr(details, 'cached_tokens', 0) or 0

    ex6.debug_print(f"[invoke] stream done, {chunk_count} chunks, finish={finish_reason}")

    tool_calls = []
    for tc in tool_calls_acc.values():
        try:
            tc["args"] = json.loads(tc["args"]) if tc["args"] else {}
        except:
            pass
        tool_calls.append(tc)
        yield ex6.ResponseChunk("tool", json.dumps(tc))

    # Calculate cost
    if ctx.model not in MODELS:
        raise ValueError(f"no pricing for model '{ctx.model}' — add it to MODELS in provider.py")
    info = MODELS[ctx.model]
    uncached_input = input_tokens - cached_tokens
    cost = (uncached_input * info.input + cached_tokens * info.cache_read + output_tokens * info.output) / 1_000_000
    increment_cost(cost)

    result = ex6.LLMResult(input_tokens, output_tokens, tool_calls, finish_reason, cost=cost)
    ex6.debug_print(f"[invoke] result: in={input_tokens} out={output_tokens} cost=${cost:.4f} tools={len(tool_calls)}")
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


_TOOL_DOCS_HEADER = """\
# Tools/Functions
You have access to tools via tool-blocks.
tool-blocks are sandboxed python scripts, with a bunch of functions for you to use.
To call them, emit a ```tools ``` block-

## EXAMPLE:
<chat-example>
USER: Can you read the files I talked about?
ASSISTANT: Let me read the files:
```tools
read_file("file.txt")
for f in files:
    read_file(f)
```
USER: <tool_result file.txt>
API_KEY=0xffffffffffffffffff
</tool_result>
<tool_result readme.txt>
todo; write this
</tool_result>
ASSISTANT: file.txt contains an API key, and readme.txt has todo.
</chat-example>


# Available Tools:"""

def _build_tool_docs(ctx: ex6.Context) -> str:
    tools = ctx.get_tools()
    if not tools:
        return ""
    lines = [_TOOL_DOCS_HEADER]
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
    ex6.debug_print(f"[call_tools] CALLED, tool_calls={len(llm_result.tool_calls)}")
    # Get last assistant message
    content = ""
    for msg in reversed(ctx.messages):
        if msg.role == "assistant":
            content = msg.content if isinstance(msg.content, str) else ""
            break

    ex6.debug_print(f"[call_tools] assistant content len={len(content)}, preview={content[:80]!r}")
    code = extract_tools_block(content)
    ex6.debug_print(f"[call_tools] extracted code={code!r}")
    if not code:
        # Fall back to native tool calls
        ex6.debug_print(f"[call_tools] no code block, falling back to native")
        return _call_tools_native(ctx, llm_result)

    # Code mode
    tools = ctx.get_tools()
    ex6.debug_print(f"[call_tools] code mode, {len(tools)} tools available: {list(tools.keys())}")
    results, threads = [], []
    ctx.data["provider:tool_results"] = results  # expose for renderer

    env = {}
    for name, fn in tools.items():
        env[name] = _wrap_tool_threaded(fn, ctx, results, threads)

    try:
        exec_sandboxed(code, env)
        ex6.debug_print(f"[call_tools] exec done, {len(threads)} threads spawned")
    except Exception as e:
        ex6.debug_print(f"[call_tools] exec FAILED: {e}")

    for t in threads:
        t.join()
    ex6.debug_print(f"[call_tools] all threads joined, {len(results)} results")
    ctx.data.pop("provider:tool_results", None)

    if results:
        parts = [f"<tool_result {r['call']}>\n{r['value']}\n</tool_result>" for r in results]
        ctx.messages.append(ex6.Message(role="user", content="\n\n".join(parts)))

    ex6.debug_print(f"[call_tools] returning {len(results) > 0}")
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
def render_tools_block(role: str, output: list[ex6.OutputLine], ctx: ex6.Context) -> None:
    i = 0
    while i < len(output):
        line = output[i]
        if isinstance(line, str) and line.startswith('```tools'):
            j = i + 1
            code_lines = []
            while j < len(output):
                ln = output[j]
                if isinstance(ln, str) and ln.strip() == '```':
                    break
                if isinstance(ln, str):
                    code_lines.append(ln)
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
def render_tool_results(role: str, output: list[ex6.OutputLine], ctx: ex6.Context) -> None:
    i = 0
    while i < len(output):
        line = output[i]
        if isinstance(line, str) and line.startswith('<tool_result '):
            call = line[13:].rstrip('>')
            j = i + 1
            content_lines = []
            while j < len(output):
                if isinstance(output[j], str) and output[j].strip() == '</tool_result>':
                    break
                if isinstance(output[j], str):
                    content_lines.append(output[j])
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


