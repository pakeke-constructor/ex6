# OpenRouter Prompt Caching Debug Report

Date: 2026-03-16

## Background

ex6 uses OpenRouter's OpenAI-compatible API to call Anthropic Claude models. Provider code is in `_ex6/provider.py`. The `invoke_llm` function builds messages, applies `cache_control` annotations, and streams responses via the `openai` Python SDK pointed at `https://openrouter.ai/api/v1`.

Caching was showing `cached_tokens=0` and `cache_write_tokens=0` on every call, even with 15k+ input tokens. This report documents the root cause and fix.

---

## OpenRouter Caching: How It Works

OpenRouter supports two approaches for Anthropic prompt caching:

### 1. Automatic (top-level `cache_control`)

Place `cache_control` in the request body (via `extra_body`), not on individual messages. OpenRouter handles breakpoint placement automatically.

```python
extra_body = {
    "cache_control": {"type": "ephemeral"},  # or {"type": "ephemeral", "ttl": "1h"}
    "provider": {"order": ["Anthropic"], "allow_fallbacks": False}
}
```

- Only works when routed directly to Anthropic (not Bedrock/Vertex).
- OpenRouter decides where to place breakpoints.

### 2. Explicit breakpoints (inline `cache_control`)

Place `cache_control` inside individual message content blocks and/or tool schemas. This is what `provider.py` uses.

```python
# On a message's content:
msg["content"] = [{"type": "text", "text": "...", "cache_control": {"type": "ephemeral"}}]

# On the last tool schema:
tools[-1]["function"]["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
```

- Max 4 explicit breakpoints per request.
- Default TTL is 5 minutes; `"ttl": "1h"` extends to 1 hour.
- Works across all Anthropic-compatible providers (Anthropic, Bedrock, Vertex).

Source: https://openrouter.ai/docs/guides/best-practices/prompt-caching

---

## Anthropic Prompt Caching Behaviour

- **Prefix-based**: caching works on the conversation prefix. A `cache_control` breakpoint marks the END of a cacheable prefix. Everything from the start of the conversation to that breakpoint is eligible for caching.
- **Minimum token threshold**: the cacheable prefix must be at least ~1024 tokens (for Claude Opus/Sonnet). Content below this threshold is silently ignored — no error, no cache write, just `cached_tokens=0`.
- **Breakpoints on empty assistant messages with `tool_calls` are ignored**: if an assistant message has `tool_calls` and empty/no text content, placing `cache_control` on it does nothing. Anthropic silently skips it. This is the root cause of the bug.
- **Multiple breakpoints**: each breakpoint defines a separate cacheable prefix endpoint. You can have up to 4. The system uses prefix matching — if a later call shares the same prefix up to a previous breakpoint, it gets a cache hit.
- **Cache hits across calls**: if call 1 caches prefix A-B-C with a breakpoint on C, and call 2 sends A-B-C-D-E with a breakpoint on E, call 2 will get a cache hit on A-B-C (from call 1's cache) and write a new cache entry for A-B-C-D-E.

---

## ex6 Architecture (Relevant Parts)

### Context (`ex6.py:444`)

- `Context.messages`: list of `Message` objects forming the conversation.
- `Context.invoke(text)`: appends a user message, runs `invoke_llm` on a background thread, loops on tool calls via `call_tools`.
- `Context.get_tool_schemas()`: returns OpenAI-format tool schemas from all messages' `.tools` dicts.

### Message (`ex6.py:295`)

- `role`: "system", "user", "assistant", "tool"
- `content`: str or callable
- `tool_calls`: list of `{"id", "name", "args"}` dicts (for assistant messages that call tools)
- `tool_call_id`: str (for tool result messages)
- `get_msg(ctx)`: returns content, resolving callables and snapshotting.

### Provider flow (`_ex6/provider.py`)

1. `msg_to_dict(m, ctx)` converts `Message` to OpenAI-format dict.
2. `invoke_llm(ctx)` builds the messages list, applies cache_control, calls OpenRouter, streams chunks, yields `ResponseChunk` and `LLMResult`.
3. `cache_manually(ctx, ttl)` registers a context for long-lived (1h) caching of system prompt + tools. Fingerprints content and persists to disk.

### Cache control application (`provider.py:203-225`)

Two breakpoints are applied:

1. **Long-lived (1h)**: on the last system message content + last tool schema. Only if `cache_manually()` was called for this context.
2. **Ephemeral (5min)**: on the second-to-last message. Intent: cache the conversation prefix so the next turn (which appends assistant response + new user message) gets a cache hit.

---

## The Bug

### Symptom

`cached_tokens=0` and `cache_write_tokens=0` on every invocation, even with 15k+ input tokens. Full-price billing on every call.

### Root Cause

The ephemeral breakpoint is placed on `messages[-2]` (second-to-last message). In the tool-call flow, the message sequence at invoke time is:

```
[0] system           (small, ~100 tokens)
[1] user             ("Read ex6.py and summarize it")
[2] assistant         content="" tool_calls=[read_file]    <-- breakpoint goes HERE
[3] tool             content=<46k chars of file>           tool_call_id=call_001
```

The breakpoint lands on `msg[2]` — an assistant message with empty content and `tool_calls`. **Anthropic silently ignores `cache_control` on such messages.** Result: zero cache write, zero cache read.

The bulk of tokens (`msg[3]`, the tool result) is AFTER the (ignored) breakpoint, so it's never part of a cached prefix.

### Why this is devastating

The tool-call loop is the most common pattern: user asks question -> assistant calls tool -> tool returns big result -> assistant responds -> user follows up. The second-to-last message is almost always an assistant+tool_calls message (or a small user message before it). The huge tool results are always LAST.

This means caching essentially never works in production, and every turn re-processes the entire conversation at full price.

### Test data

```
CURRENT (second-to-last breakpoint):
  Invoke 1: in=15950  write=0      cached=0      cost=$0.095
  Invoke 2: in=16573  write=16560  cached=0      cost=$0.118

FIXED (last-message breakpoint):
  Invoke 1: in=15950  write=15949  cached=0      cost=$0.120
  Invoke 2: in=16763  write=811    cached=15949   cost=$0.028
```

Invoke 2 cost: $0.118 -> $0.028 (4.2x reduction). Compounds on every subsequent turn.

---

## The Fix

Change the ephemeral breakpoint from `messages[-2]` to `messages[-1]`.

### Why this works

Prefix caching matches from the START of the conversation. The breakpoint position only controls how much of the prefix to cache — it doesn't need to be "before the new content." If call 1 caches messages 0-3 with a breakpoint on msg[3], and call 2 sends messages 0-5 with a breakpoint on msg[5], call 2 still gets a cache hit on the shared prefix 0-3.

By putting the breakpoint on the last message, we ensure the entire conversation (including large tool results) is cached. The next call will match that entire prefix and only pay for the new content.

### Code change in `_ex6/provider.py`

```python
# BEFORE (line 218):
if len(messages) >= 2:
    target = messages[-2]

# AFTER:
if len(messages) >= 2:
    target = messages[-1]
```

The rest of the logic (skip if already has cache_control) stays the same.

---

## Reading Cache Stats from OpenRouter

OpenRouter returns cache info in the streaming usage chunk:

```python
chunk.usage.prompt_tokens_details.cached_tokens       # in __dict__
chunk.usage.prompt_tokens_details.cache_write_tokens   # in model_extra (pydantic)
```

Note: `cache_write_tokens` is a non-standard field that the OpenAI SDK puts in `model_extra`, not `__dict__`. Access via `getattr(details, 'cache_write_tokens', 0)` works because pydantic resolves `model_extra` fields through `__getattr__`.

OpenRouter also returns cost directly:
```python
chunk.usage.model_extra['cost']           # total cost (float, dollars)
chunk.usage.model_extra['cost_details']   # breakdown: upstream_inference_prompt_cost, etc.
```

---

## Test Script

`test_caching.py` in the repo root. Simulates a realistic tool-call conversation (system prompt + tools + user message + assistant tool_call + big tool result), then does follow-up turns. Compares current vs fixed breakpoint placement. Run with:

```
python test_caching.py
```
