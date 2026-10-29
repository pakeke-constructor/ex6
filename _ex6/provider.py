# Dependencies: openai
# Install: python -m pip install openai
# Requires OPENROUTER_API_KEY for OpenRouter requests.

import json
import hashlib
import sqlite3
import ex6
import openai
import os
import base64
import copy
from _ex6.models import M, ModelInfo


_audio_models = None


def supports_audio(ctx: ex6.Context) -> bool:
    """True if ctx uses this provider and OpenRouter lists audio input for its model."""
    global _audio_models
    if ctx.get_invoker() is not invoke_openrouter:
        return False
    if _audio_models is None:
        import urllib.request
        with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=30) as r:
            models = json.load(r)["data"]
        _audio_models = {m["id"] for m in models if "audio" in m["architecture"]["input_modalities"]}
    return ctx.model.split(":")[0] in _audio_models


def _cache_file():
    return ex6.get_folder() / "cache_state.json"


def cache_manually(ctx: ex6.Context, ttl="1h"):
    """Mark a context for long-lived caching. Filesystem-backed, idempotent."""
    import time
    # Fingerprint: hash system messages + tools
    parts = []
    for m in ctx.get_messages():
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
        ctx.app.plugin_data.setdefault("provider:cached_contexts", {})[ctx.name] = ttl
        return

    ctx.app.plugin_data.setdefault("provider:cached_contexts", {})[ctx.name] = ttl
    state[ctx.name] = {"fp": fp, "ts": time.time()}
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(state))
    ctx.app.debug_print(f"[cache] registered {ctx.name} (ttl={ttl}, fp={fp})")


def _apply_cache_control(content: str | list[dict], cc: dict) -> list[dict]:
    """Add cache_control to a message content field. Returns array-format content."""
    blocks: list[dict]
    if isinstance(content, str):
        blocks = [{"type": "text", "text": content}]
    else:
        blocks = copy.deepcopy(content)
    if blocks:
        blocks[-1]["cache_control"] = cc
    return blocks


def _data_url(attachment):
    if not os.path.isfile(attachment.path):
        raise ValueError(f"Attachment file is missing: {attachment.path}")
    with open(attachment.path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{attachment.mime_type};base64,{encoded}"


def _tool_content(result: ex6.ToolResult):
    if not result.attachments:
        return result.text
    content: list[dict] = [{"type": "text", "text": result.text}]
    for attachment in result.attachments:
        url = _data_url(attachment)
        if isinstance(attachment, ex6.ImageAttachment):
            content.append({"type": "image_url", "image_url": {"url": url, "detail": attachment.detail}})
        elif isinstance(attachment, ex6.AudioAttachment):
            # OpenRouter drops input_audio parts in tool messages, but forwards audio data URLs in image_url.
            content.append({"type": "image_url", "image_url": {"url": url}})
        else:
            raise ValueError(f"Unsupported attachment type: {type(attachment).__name__}")
    return content


def msg_to_dict(m: ex6.Message, ctx: ex6.Context):
    value = m.get_msg(ctx)
    content = _tool_content(value) if isinstance(value, ex6.ToolResult) else value
    if m.attachments:
        content = [{"type": "text", "text": content}]
        for token, attachment in m.attachments.items():
            content += [{"type": "text", "text": token},
                        {"type": "image_url", "image_url": {"url": _data_url(attachment), "detail": attachment.detail}}]
    d: dict = {"role": m.role, "content": content}
    if m.tool_calls:
        d["tool_calls"] = [
            {"id": tc["id"], "type": "function",
             "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])}}
            for tc in m.tool_calls
        ]
    if m.tool_call_id:
        d["tool_call_id"] = m.tool_call_id
    return d



@ex6.handler(order=100)
def invoke_llm(ctx: ex6.Context):
    return invoke_openrouter


def invoke_openrouter(ctx: ex6.Context):
    messages = [msg_to_dict(m, ctx) for m in ctx.get_messages()]

    tools = ctx.get_tool_schemas() or None

    if ctx.app.budget.is_over():
        result = ex6.LLMResult(error=f"daily budget exceeded (${ctx.app.budget.get_cost():.2f}/${ctx.app.budget.get_limit():.2f})")
        _log_invoke(ctx, "openrouter", messages, tools, result)
        yield result
        return

    if ctx.get_native_tools():
        result = ex6.LLMResult(error=f"openrouter provider supports no native tools: {ctx.get_native_tools()}")
        _log_invoke(ctx, "openrouter", messages, tools, result)
        yield result
        return

    # Anthropic prompt caching
    if ctx.model.startswith("anthropic/"):
        # 1h cache on last system message + tools (if ctx was registered via cache())
        cached_contexts = ctx.app.plugin_data.setdefault("provider:cached_contexts", {})
        if ctx.name in cached_contexts:
            ttl = cached_contexts[ctx.name]
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

    ctx.app.debug_print(f"[invoke] model={ctx.model} msgs={len(messages)}")
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
        ctx.app.debug_print(f"[invoke] API EXCEPTION: {e}")
        result = ex6.LLMResult(error=str(e))
        _log_invoke(ctx, "openrouter", messages, tools, result)
        yield result
        return

    ctx.app.debug_print("[invoke] stream started")
    input_tokens, output_tokens, cached_tokens, cache_write_tokens = 0, 0, 0, 0
    provider_cost = None  # OpenRouter may return cost directly
    raw_usage = {}
    finish_reason = "stop"
    tool_calls_acc = {}
    chunk_count = 0
    text = ""

    try:
        for chunk in response:
            chunk_count += 1
            delta = chunk.choices[0].delta if chunk.choices else None

            if delta and delta.content:
                text += delta.content
                yield ex6.ResponseChunk("text", delta.content)

            # CoT (OpenRouter reasoning field)
            reasoning = getattr(delta, 'reasoning', None) if delta else None
            if reasoning:
                yield ex6.ResponseChunk("cot", reasoning)

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
                raw_usage = chunk.usage.model_dump()
                input_tokens = chunk.usage.prompt_tokens or 0
                output_tokens = chunk.usage.completion_tokens or 0
                details = getattr(chunk.usage, 'prompt_tokens_details', None)
                if details:
                    cached_tokens = getattr(details, 'cached_tokens', 0) or 0
                    cache_write_tokens = getattr(details, 'cache_write_tokens', 0) or 0
                provider_cost = getattr(chunk.usage, 'cost', None)
    except Exception as e:
        ctx.app.debug_print(f"[invoke] stream exception: {e}")
        result = ex6.LLMResult(error=str(e))
        _log_invoke(ctx, "openrouter", messages, tools, result, text)
        yield result
        return
    ctx.app.debug_print(f"[invoke] stream done, {chunk_count} chunks, finish={finish_reason}")

    tool_calls = []
    error = None
    for tc in tool_calls_acc.values():
        try:
            tc["args"] = json.loads(tc["args"]) if tc["args"] else {}
        except json.JSONDecodeError as e:
            error = f"malformed tool args for {tc['name']}: {e}: {tc['args'][:500]!r}"
        if not error and not isinstance(tc["args"], dict):
            error = f"malformed tool args for {tc['name']}: not an object: {tc['args']!r}"
        tool_calls.append(tc)

    # Use provider-reported cost if available, otherwise estimate
    if provider_cost is not None:
        cost = provider_cost
    else:
        info = M.get(ctx.model.split(":")[0])  # strip ":online" web-search suffix
        if info is None:
            raise ValueError(f"no pricing for model '{ctx.model}' — add it to M in provider.py")
        uncached_input = input_tokens - cached_tokens - cache_write_tokens
        cost = (uncached_input * info.input + cached_tokens * info.cache_read
                + cache_write_tokens * info.cache_write + output_tokens * info.output) / 1_000_000
    ctx.app.budget.add_cost(cost)

    if error:
        tool_calls = []
    result = ex6.LLMResult(input_tokens, output_tokens, tool_calls, finish_reason, cost=cost, error=error,
                           cached_tokens=cached_tokens, cache_write_tokens=cache_write_tokens, raw_usage=raw_usage)
    ctx.app.debug_print(f"[invoke] result: in={input_tokens} out={output_tokens} cost=${cost:.4f} cached_tokens={cached_tokens} cache_write_tokens={cache_write_tokens} tools={len(tool_calls)}")
    _log_invoke(ctx, "openrouter", messages, tools, result, text)
    yield result



@ex6.command
def usage(tui):
    """Show today's spending."""
    from _ex6.commands import _text_panel
    lines = [
        f"Today: ${tui.app.budget.get_cost():.4f} / ${tui.app.budget.get_limit():.2f}",
    ]
    _text_panel(tui, lines)


def _log_safe(value):
    if isinstance(value, dict) and value.get("type") in ("base64", "image") and "data" in value:  # anthropic / MCP images
        return {**value, "data": "<base64 data omitted>"}
    if isinstance(value, dict):
        return {k: _log_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_log_safe(v) for v in value]
    if isinstance(value, str) and value.startswith("data:") and ";base64," in value:
        return "<base64 data omitted>"
    return value


_DB_LIMIT = 1_000_000_000  # bytes of live data in invokes.db; oldest invokes are culled past this


def _db():
    db = sqlite3.connect(ex6.get_folder() / "invokes.db", timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS blobs(hash TEXT PRIMARY KEY, data TEXT);
        CREATE TABLE IF NOT EXISTS invokes(id INTEGER PRIMARY KEY, ts, cwd, ctx, model, provider,
            input_tokens, output_tokens, cached_tokens, cache_write_tokens, cost, finish_reason, error,
            messages TEXT, tools TEXT, output TEXT);
    """)
    return db


def _blob(db, value):
    data = json.dumps(_log_safe(value), sort_keys=True)
    h = hashlib.sha256(data.encode()).hexdigest()
    db.execute("INSERT OR IGNORE INTO blobs VALUES (?, ?)", (h, data))
    return h


def _db_size(db):
    return db.execute("SELECT (page_count - freelist_count) * page_size "
                      "FROM pragma_page_count(), pragma_freelist_count(), pragma_page_size()").fetchone()[0]


def _cull(db):
    """Over the limit: drop oldest invokes and their orphaned blobs until under 90% of it."""
    if _db_size(db) <= _DB_LIMIT:
        return
    while _db_size(db) > _DB_LIMIT * 0.9:
        db.execute("DELETE FROM invokes WHERE id IN "
                   "(SELECT id FROM invokes ORDER BY id LIMIT (SELECT COUNT(*) / 10 + 1 FROM invokes))")
        db.execute("DELETE FROM blobs WHERE hash NOT IN "
                   "(SELECT value FROM invokes, json_each(invokes.messages) UNION SELECT tools FROM invokes)")


def _log_invoke(ctx, provider, messages, tools, result, text=""):
    """Record a finished invoke. messages/tools: the provider-format payload actually sent."""
    from datetime import datetime
    output = {"text": text, "tool_calls": result.tool_calls, "native_calls": result.native_calls}
    db = _db()
    try:
        with db:  # one transaction per invoke
            hashes = [_blob(db, m) for m in messages]
            db.execute(
                "INSERT INTO invokes (ts, cwd, ctx, model, provider, input_tokens, output_tokens, cached_tokens,"
                " cache_write_tokens, cost, finish_reason, error, messages, tools, output)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (datetime.now().isoformat(timespec="seconds"), ctx.cwd or os.getcwd(), ctx.name, ctx.model,
                 provider, result.input_tokens, result.output_tokens, result.cached_tokens,
                 result.cache_write_tokens, result.cost, result.finish_reason, result.error,
                 json.dumps(hashes), _blob(db, tools or []), json.dumps(_log_safe(output))))
            _cull(db)
    finally:
        db.close()


