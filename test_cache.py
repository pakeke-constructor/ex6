"""Test provider.py's actual caching logic against OpenRouter/Anthropic.
Simulates what invoke_llm does: applies cache_control to messages, then calls the API.
"""
import os, json, requests, time

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
BASE = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
PROVIDER = {"order": ["Anthropic"], "allow_fallbacks": False}

# --- Replicate provider.py's _apply_cache_control ---
def apply_cc(content, cc):
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    if isinstance(content, list) and content:
        content[-1]["cache_control"] = cc
    return content

def call(label, model, messages):
    body = {
        "model": model,
        "stream": False,
        "max_tokens": 5,
        "temperature": 0,
        "messages": messages,
        "provider": PROVIDER,
    }
    resp = requests.post(BASE, headers=HEADERS, json=body)
    data = resp.json()
    if "error" in data:
        print(f"  {label}: ERROR: {data['error']}")
        return
    usage = data.get("usage", {})
    details = usage.get("prompt_tokens_details", {})
    print(f"  {label}: prompt={usage.get('prompt_tokens')} "
          f"cached={details.get('cached_tokens',0)} "
          f"cache_write={details.get('cache_write_tokens',0)} "
          f"cost=${usage.get('cost', '?')}")

# ================================================================
# Test 1: provider.py's "Layer 2" — ephemeral on messages[-2]
# Simulates a multi-turn conversation where system+history is the prefix
# ================================================================
print("=== TEST 1: Layer 2 (ephemeral on messages[-2]) ===")
print("    Simulates: system(big) + user + assistant + user (new)")

TS = str(time.time())  # unique per run, busts any prior cache
SYS1 = f"[{TS}] You are helpful. " + ("Be concise. " * 1500)
SYS2 = f"[{TS}] You are a math tutor. " + ("Show your work. " * 1500)
SYS3 = f"[{TS}] You are a coding assistant. " + ("Write clean code. " * 1500)

msgs_t1 = [
    {"role": "system", "content": SYS1},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
    {"role": "user", "content": "Reply OK."},
]
# Apply provider.py Layer 2: ephemeral on messages[-2] (the assistant msg)
msgs_t1[-2]["content"] = apply_cc(msgs_t1[-2]["content"], {"type": "ephemeral"})

call("cold", "anthropic/claude-sonnet-4.6", msgs_t1)
call("warm", "anthropic/claude-sonnet-4.6", msgs_t1)

# ================================================================
# Test 2: Layer 1 — 1h cache on system msg (cache_manually scenario)
# ================================================================
print("\n=== TEST 2: Layer 1 (1h cache on system msg) ===")
print("    Simulates: cache_manually() registered context")

msgs_t2 = [
    {"role": "system", "content": apply_cc(SYS2, {"type": "ephemeral", "ttl": "1h"})},
    {"role": "user", "content": "Reply OK."},
]
# Also apply Layer 2 on messages[-2] (the system msg — but it already has cc, so skip)
# This matches provider.py's "already_cached" check

call("cold", "anthropic/claude-sonnet-4.6", msgs_t2)
call("warm", "anthropic/claude-sonnet-4.6", msgs_t2)

# ================================================================
# Test 3: Both layers — 1h on system + ephemeral on messages[-2]
# ================================================================
print("\n=== TEST 3: Both layers (1h system + ephemeral prefix) ===")
print("    Simulates: cache_manually ctx with conversation history")

msgs_t3 = [
    {"role": "system", "content": apply_cc(SYS3, {"type": "ephemeral", "ttl": "1h"})},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
    {"role": "user", "content": "Reply OK."},
]
# Layer 2: ephemeral on messages[-2]
msgs_t3[-2]["content"] = apply_cc(msgs_t3[-2]["content"], {"type": "ephemeral"})

call("cold", "anthropic/claude-sonnet-4.6", msgs_t3)
call("warm", "anthropic/claude-sonnet-4.6", msgs_t3)

# ================================================================
# Test 4: Small system prompt — does caching silently fail?
# ================================================================
print("\n=== TEST 4: Small system prompt (under threshold) ===")
print("    Expect: no caching (system prompt too small)")

SMALL_SYS = "You are helpful. Be concise."
msgs_t4 = [
    {"role": "system", "content": apply_cc(SMALL_SYS, {"type": "ephemeral"})},
    {"role": "user", "content": "Reply OK."},
]

call("cold", "anthropic/claude-sonnet-4.6", msgs_t4)
call("warm", "anthropic/claude-sonnet-4.6", msgs_t4)
