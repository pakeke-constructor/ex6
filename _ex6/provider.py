import json
import threading
import ex6
import openai
import os
from datetime import date
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



# Model constants (for robust static checks, avoids typos)
class ModelNamespace:
    OPUS_46        = "anthropic/claude-opus-4.6"
    SONNET_46      = "anthropic/claude-sonnet-4.6"
    HAIKU_45       = "anthropic/claude-haiku-4.5"
    GPT5           = "openai/gpt-5"
    GPT5_MINI      = "openai/gpt-5-mini"
    GPT5_CODEX     = "openai/gpt-5-codex"
    GPT52_CODEX    = "openai/gpt-5.2-codex"
    GPT51_CODEX_MINI = "openai/gpt-5.1-codex-mini"
    CODEX_MINI     = "openai/codex-mini"
    O4_MINI        = "openai/o4-mini"
    GEMINI3_PRO    = "google/gemini-3-pro-preview"
    GEMINI3_FLASH  = "google/gemini-3-flash-preview"
    GEMINI31_FLASH_LITE = "google/gemini-3.1-flash-lite-preview"
    GEMINI25_PRO   = "google/gemini-2.5-pro"
    GEMINI25_FLASH = "google/gemini-2.5-flash"
    GEMINI25_FLASH_LITE = "google/gemini-2.5-flash-lite"
    GROK4          = "x-ai/grok-4"
    GROK41_FAST    = "x-ai/grok-4.1-fast"
    DEEPSEEK_CHAT  = "deepseek/deepseek-chat-v3.1"
    DEEPSEEK_R1    = "deepseek/deepseek-r1"
    QWEN3_CODER    = "qwen/qwen3-coder"
    KIMI_K25       = "moonshotai/kimi-k2.5"



# $/M tokens (input, output, cache_read, cache_write)
MODELS = {
    # --- Anthropic ---
    "anthropic/claude-opus-4.6":         ModelInfo(200_000, 5, 25, 0.5, 6.25),
    "anthropic/claude-sonnet-4.6":       ModelInfo(200_000, 3, 15, 0.3, 3.75),
    "anthropic/claude-haiku-4.5":        ModelInfo(200_000, 1, 5, 0.1, 1.25),
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
    "google/gemini-3.1-flash-lite-preview": ModelInfo(1_048_576, 0.25, 1.5, 0.025),
    "google/gemini-2.5-pro":             ModelInfo(1_048_576, 1.25, 10, 0.125),
    "google/gemini-2.5-flash":           ModelInfo(1_048_576, 0.3, 2.5, 0.03),
    "google/gemini-2.5-flash-lite":      ModelInfo(1_048_576, 0.1, 0.4, 0.01),
    # --- xAI ---
    "x-ai/grok-4":                       ModelInfo(256_000, 3, 15, 0.75),
    "x-ai/grok-4.1-fast":                ModelInfo(2_000_000, 0.2, 0.5, 0.05),
    # --- DeepSeek ---
    "deepseek/deepseek-chat-v3.1":       ModelInfo(32_768, 0.15, 0.75, 0),
    "deepseek/deepseek-r1":              ModelInfo(64_000, 0.7, 2.5, 0),
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

    tools = ctx.get_tool_schemas() or None

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

    extra = {}
    if ctx.model.startswith("anthropic/"):
        extra["extra_body"] = {"provider": {"order": ["Anthropic"], "allow_fallbacks": False}}

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
        yield ex6.LLMResult(error=str(e))
        return

    ex6.debug_print("[invoke] stream started")
    input_tokens, output_tokens, cached_tokens, cache_write_tokens = 0, 0, 0, 0
    provider_cost = None  # OpenRouter may return cost directly
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
                cache_write_tokens = getattr(details, 'cache_write_tokens', 0) or 0
            provider_cost = getattr(chunk.usage, 'cost', None)

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
        if ctx.model not in MODELS:
            raise ValueError(f"no pricing for model '{ctx.model}' — add it to MODELS in provider.py")
        info = MODELS[ctx.model]
        uncached_input = input_tokens - cached_tokens - cache_write_tokens
        cost = (uncached_input * info.input + cached_tokens * info.cache_read
                + cache_write_tokens * info.cache_write + output_tokens * info.output) / 1_000_000
    increment_cost(cost)

    result = ex6.LLMResult(input_tokens, output_tokens, tool_calls, finish_reason, cost=cost)
    ex6.debug_print(f"[invoke] result: in={input_tokens} out={output_tokens} cost=${cost:.4f} tools={len(tool_calls)}")
    _log_invoke(ctx, messages, result)
    yield result


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
        c = m['content']
        lines.append(c if isinstance(c, str) else json.dumps(c))
        lines.append("")

    (folder / fname).write_text("\n".join(lines), encoding="utf-8")


