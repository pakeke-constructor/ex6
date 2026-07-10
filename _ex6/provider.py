import json
import hashlib
import ex6
import openai
import os
from _ex6.models import M, ModelInfo




_cached_contexts = {}  # ctx.name -> ttl
_CACHE_FILE = None  # lazy

def _cache_file():
    global _CACHE_FILE
    if not _CACHE_FILE:
        _CACHE_FILE = ex6.get_folder() / "cache_state.json"
    return _CACHE_FILE


def cache_manually(ctx: ex6.Context, ttl="1h"):
    """Mark a context for long-lived caching. Filesystem-backed, idempotent."""
    import time
    # Fingerprint: hash system messages + tools
    parts = []
    for m in ctx.messages:
        if m.role == "system":
            c = m.get_msg(ctx)
            parts.append(c if isinstance(c, str) else json.dumps(c))
    parts.append(json.dumps(ctx.get_tool_schemas()))
    fp = hashlib.sha256("".join(parts).encode()).hexdigest()[:16]

    # Check filesystem — skip if same content and still within TTL
    f = _cache_file()
    state = json.loads(f.read_text()) if f.exists() else {}
    entry = state.get(ctx.name)
    ttl_sec = 3600 if ttl == "1h" else 300
    if entry and entry["fp"] == fp and (time.time() - entry["ts"]) < ttl_sec:
        _cached_contexts[ctx.name] = ttl
        return

    _cached_contexts[ctx.name] = ttl
    state[ctx.name] = {"fp": fp, "ts": time.time()}
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(state))
    ex6.debug_print(f"[cache] registered {ctx.name} (ttl={ttl}, fp={fp})")


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
    messages = [msg_to_dict(m, ctx) for m in ctx.messages]

    if ex6.is_over_budget():
        result = ex6.LLMResult(error=f"daily budget exceeded (${ex6.get_daily_cost():.2f}/${ex6.get_daily_limit():.2f})")
        _log_invoke(ctx, messages, result)
        yield result
        return

    tools = ctx.get_tool_schemas() or None

    # Anthropic prompt caching
    if ctx.model.startswith("anthropic/"):
        # 1h cache on last system message + tools (if ctx was registered via cache())
        if ctx.name in _cached_contexts:
            ttl = _cached_contexts[ctx.name]
            cc = {"type": "ephemeral", "ttl": ttl}
            # Only cache last system msg (prefix-based, covers everything before it)
            for msg in reversed(messages):
                if msg["role"] == "system":
                    msg["content"] = _apply_cache_control(msg["content"], cc)
                    break
            if tools:
                tools[-1]["function"]["cache_control"] = cc
        # Ephemeral breakpoint on last message (cache entire conversation prefix)
        # Using [-1] not [-2]: assistant+tool_calls msgs have empty content and
        # Anthropic silently ignores cache_control on them, breaking caching.
        if len(messages) >= 2:
            target = messages[-1]
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

    extra = {}
    if ctx.model.startswith("anthropic/"):
        body = {"provider": {"order": ["Anthropic"], "allow_fallbacks": False}}
        if ctx.reasoning != "none":
            body["reasoning"] = {"effort": ctx.reasoning}
        extra["extra_body"] = body

    ex6.debug_print(f"[invoke] model={ctx.model} msgs={len(messages)}")
    try:
        response = client.chat.completions.create( # type: ignore[arg-type]
            model=ctx.model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            tools=tools,
            timeout=30,
            **extra,
        )
    except Exception as e:
        ex6.debug_print(f"[invoke] API EXCEPTION: {e}")
        result = ex6.LLMResult(error=str(e))
        _log_invoke(ctx, messages, result)
        yield result
        return

    ex6.debug_print("[invoke] stream started")
    input_tokens, output_tokens, cached_tokens, cache_write_tokens = 0, 0, 0, 0
    provider_cost = None  # OpenRouter may return cost directly
    finish_reason = "stop"
    tool_calls_acc = {}
    chunk_count = 0

    try:
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
                    cache_write_tokens = getattr(details, 'cache_write_tokens', 0) or 0
                provider_cost = getattr(chunk.usage, 'cost', None)
    except Exception as e:
        ex6.debug_print(f"[invoke] stream exception: {e}")
        result = ex6.LLMResult(error=str(e))
        _log_invoke(ctx, messages, result, cached_tokens, cache_write_tokens)
        yield result
        return
    ex6.debug_print(f"[invoke] stream done, {chunk_count} chunks, finish={finish_reason}")

    tool_calls = []
    for tc in tool_calls_acc.values():
        try:
            tc["args"] = json.loads(tc["args"]) if tc["args"] else {}
        except:
            pass
        tool_calls.append(tc)
        yield ex6.ResponseChunk("tool", json.dumps(tc))

    # Use provider-reported cost if available, otherwise estimate
    if provider_cost is not None:
        cost = provider_cost
    else:
        info = M.get(ctx.model)
        if info is None:
            raise ValueError(f"no pricing for model '{ctx.model}' — add it to M in provider.py")
        uncached_input = input_tokens - cached_tokens - cache_write_tokens
        cost = (uncached_input * info.input + cached_tokens * info.cache_read
                + cache_write_tokens * info.cache_write + output_tokens * info.output) / 1_000_000
    ex6.add_cost(cost)

    result = ex6.LLMResult(input_tokens, output_tokens, tool_calls, finish_reason, cost=cost)
    ex6.debug_print(f"[invoke] result: in={input_tokens} out={output_tokens} cost=${cost:.4f} cached_tokens={cached_tokens} cache_write_tokens={cache_write_tokens} tools={len(tool_calls)}")
    _log_invoke(ctx, messages, result, cached_tokens, cache_write_tokens)
    yield result



@ex6.command
def usage():
    """Show today's spending."""
    from _ex6.commands import _text_panel
    lines = [
        f"Today: ${ex6.get_daily_cost():.4f} / ${ex6.get_daily_limit():.2f}",
    ]
    _text_panel(lines)


def _log_invoke(ctx, messages, result, cached_tokens=0, cache_write_tokens=0):
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
        f"cached_tokens: {cached_tokens}",
        f"cache_write_tokens: {cache_write_tokens}",
        f"cost: ${result.cost:.4f}" if result.cost else "cost: N/A",
        f"finish_reason: {result.finish_reason}",
        f"error: {result.error}" if result.error else "error: None",
        "============",
        "",
        "=== CONTEXT ===",
    ]
    for m in messages:
        lines.append(f"[{m['role']}]")
        c = m['content']
        lines.append(c if isinstance(c, str) else json.dumps(c))
        lines.append("")

    (folder / fname).write_text("\n".join(lines), encoding="utf-8")


