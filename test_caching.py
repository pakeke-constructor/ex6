"""Test OpenRouter prompt caching for Anthropic models.

Simulates a REAL conversation: system prompt, tools, user ask, assistant tool_call,
tool result (big file read), assistant reply, then a follow-up. Checks caching on
the follow-up where the big prefix should be cached.

Run from ex6 root: python test_caching.py
"""

import sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__))

import ex6
import openai
from _ex6.provider import cache_manually, msg_to_dict, _apply_cache_control, _cached_contexts

MODEL = "anthropic/claude-opus-4.6"
RUN_ID = f"run-{int(time.time())}-{os.getpid()}"

# Realistic system prompt (not padded — similar size to real coder agent)
SYSTEM_PROMPT = f"""You are a coding assistant. Be concise. Run ID: {RUN_ID}"""

# Read ex6.py as the "big tool result" — this is what happens in production
BIG_FILE = open(os.path.join(os.path.dirname(__file__), "ex6.py"), "r").read()


def read_file(ctx, path: str):
    """Read a file and return its contents."""
    return open(path).read()

def edit_file(ctx, path: str, old: str, new: str):
    """Edit a file by replacing old with new."""
    return "OK"


def build_payload_current(ctx):
    """Current provider.py logic (mirrors invoke_llm exactly)."""
    messages = [msg_to_dict(m, ctx) for m in ctx.messages]
    tools = ctx.get_tool_schemas() or None

    if ctx.model.startswith("anthropic/"):
        if ctx.name in _cached_contexts:
            ttl = _cached_contexts[ctx.name]
            cc = {"type": "ephemeral", "ttl": ttl}
            for msg in reversed(messages):
                if msg["role"] == "system":
                    msg["content"] = _apply_cache_control(msg["content"], cc)
                    break
            if tools:
                tools[-1]["function"]["cache_control"] = cc
        if len(messages) >= 2:
            target = messages[-2]
            content = target["content"]
            already_cached = (isinstance(content, list) and content
                              and "cache_control" in content[-1])
            if not already_cached:
                target["content"] = _apply_cache_control(
                    content, {"type": "ephemeral"})

    return messages, tools


def build_payload_fixed(ctx):
    """Fixed: put ephemeral breakpoint on the LAST message (not second-to-last).
    Prefix caching matches from the start regardless of breakpoint position.
    Putting it on last msg ensures the full conversation gets cached, including
    tool results that are often the bulk of tokens."""
    messages = [msg_to_dict(m, ctx) for m in ctx.messages]
    tools = ctx.get_tool_schemas() or None

    if ctx.model.startswith("anthropic/"):
        if ctx.name in _cached_contexts:
            ttl = _cached_contexts[ctx.name]
            cc = {"type": "ephemeral", "ttl": ttl}
            for msg in reversed(messages):
                if msg["role"] == "system":
                    msg["content"] = _apply_cache_control(msg["content"], cc)
                    break
            if tools:
                tools[-1]["function"]["cache_control"] = cc
        # Fixed: breakpoint on last message
        if len(messages) >= 2:
            target = messages[-1]
            content = target["content"]
            already_cached = (isinstance(content, list) and content
                              and "cache_control" in content[-1])
            if not already_cached:
                target["content"] = _apply_cache_control(
                    content, {"type": "ephemeral"})

    return messages, tools


def call_and_report(ctx, label, use_fixed=False):
    """Call OpenRouter with inline breakpoints, report cache stats."""
    build_fn = build_payload_fixed if use_fixed else build_payload_current
    messages, tools = build_fn(ctx)

    # Show message structure
    print(f"\n--- {label} ---")
    for i, m in enumerate(messages):
        c = m["content"]
        has_cc = False
        if isinstance(c, list):
            has_cc = any("cache_control" in b for b in c)
        tc = ""
        if m.get("tool_calls"):
            tc = f" tool_calls={[t['function']['name'] for t in m['tool_calls']]}"
        tid = f" tool_call_id={m['tool_call_id']}" if m.get("tool_call_id") else ""
        print(f"  msg[{i}] role={m['role']:9s} cc={has_cc} len={len(str(c)):>6}{tc}{tid}")
    if tools:
        has_cc = "cache_control" in tools[-1].get("function", {})
        print(f"  tools: {len(tools)}, last cc={has_cc}")

    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
    )

    response = client.chat.completions.create(
        model=ctx.model,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
        tools=tools,
        timeout=60,
        extra_body={"provider": {"order": ["Anthropic"], "allow_fallbacks": False}},
    )

    text = ""
    cached, cache_write, input_tok, output_tok = 0, 0, 0, 0

    for chunk in response:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            text += delta.content

        if hasattr(chunk, 'usage') and chunk.usage:
            u = chunk.usage
            input_tok = u.prompt_tokens or 0
            output_tok = u.completion_tokens or 0
            details = getattr(u, 'prompt_tokens_details', None)
            if details:
                cached = getattr(details, 'cached_tokens', 0) or 0
                cache_write = getattr(details, 'cache_write_tokens', 0) or 0
            cost = getattr(u, 'cost', None)
            if hasattr(u, 'model_extra'):
                cost = u.model_extra.get('cost', cost)

    print(f"  >> in={input_tok} out={output_tok} cached={cached} write={cache_write} cost=${cost or 0:.4f}")
    print(f"  >> response: {text[:100]}")
    return text, input_tok, output_tok, cached, cache_write


def make_ctx(name):
    """Create a context with realistic tool-call history pre-loaded."""
    ctx = ex6.Context(name=name, model=MODEL, reasoning="none")
    ctx.messages.append(ex6.Message(role="system", content=SYSTEM_PROMPT,
                                     tools={"read_file": read_file, "edit_file": edit_file}))
    cache_manually(ctx, ttl="1h")

    # User asks to read a file
    ctx.messages.append(ex6.Message(role="user", content="Read ex6.py and summarize it."))
    # Assistant calls tool
    ctx.messages.append(ex6.Message(
        role="assistant", content="",
        tool_calls=[{"id": "call_001", "name": "read_file", "args": {"path": "ex6.py"}}]
    ))
    # Tool result (the big file)
    ctx.messages.append(ex6.Message(role="tool", content=BIG_FILE, tool_call_id="call_001"))
    return ctx


def run_test():
    print(f"=== Cache Test (realistic tool-call flow): {MODEL} ===")
    print(f"=== Run ID: {RUN_ID} ===")
    print(f"=== Big file size: {len(BIG_FILE)} chars ===")

    # ---- TEST A: current (buggy) logic ----
    print(f"\n{'='*60}")
    print(f"  TEST A: CURRENT provider.py logic")
    print(f"{'='*60}")

    ctx_a = make_ctx("cache_a")

    text1, in1, out1, cached1, write1 = call_and_report(
        ctx_a, "Invoke 1: after tool result", use_fixed=False)
    ctx_a.messages.append(ex6.Message(role="assistant", content=text1))

    ctx_a.messages.append(ex6.Message(role="user", content="What's the Context class do?"))
    text2, in2, out2, cached2, write2 = call_and_report(
        ctx_a, "Invoke 2: follow-up", use_fixed=False)

    print(f"\n  CURRENT: invoke1 write={write1} | invoke2 cached={cached2}")

    # ---- TEST B: fixed logic ----
    print(f"\n{'='*60}")
    print(f"  TEST B: FIXED logic (skip assistant+tool_calls)")
    print(f"{'='*60}")

    ctx_b = make_ctx("cache_b")

    text3, in3, out3, cached3, write3 = call_and_report(
        ctx_b, "Invoke 1: after tool result", use_fixed=True)
    ctx_b.messages.append(ex6.Message(role="assistant", content=text3))

    ctx_b.messages.append(ex6.Message(role="user", content="What's the Context class do?"))
    text4, in4, out4, cached4, write4 = call_and_report(
        ctx_b, "Invoke 2: follow-up", use_fixed=True)

    print(f"\n  FIXED: invoke1 write={write3} | invoke2 cached={cached4}")

    # ---- VERDICT ----
    print(f"\n{'='*60}")
    print(f"  VERDICT")
    print(f"{'='*60}")
    print(f"  CURRENT: invoke1(in={in1} write={write1}) invoke2(in={in2} cached={cached2})")
    print(f"  FIXED:   invoke1(in={in3} write={write3}) invoke2(in={in4} cached={cached4})")
    if write3 > write1 or cached4 > cached2:
        print("\n  FIX CONFIRMED: skipping assistant+tool_calls improves caching")
    elif cached2 > 0 and cached4 > 0:
        print("\n  Both work (maybe OpenRouter cached from test A)")
    else:
        print("\n  Neither works — deeper issue")


if __name__ == "__main__":
    run_test()
