"""
PROBLEM: there are a lot of ways for ex6 providers to fail invisibly.
E.g, malformed tool input/output, bad formatting, AND MOST DANGEROUSLY: *incorrect caching.*
So, this file gives a way to test silent failures.

Live smoke test for SILENT provider failures: prompt caching, tool args/results, attachments, protocol drift.
Not a formal suite. Hits the real backends with cheap models.

    python -m _ex6.providers_test [anthropic] [openai] [openrouter]   # from repo root; default: anthropic openai ($0, subscription quota)
    run_tests(["anthropic"])                                          # from python; returns list of FAIL lines

Loaded as a plugin too, but import has no side effects (own App is built inside run_tests).

openrouter is opt-in: costs real money. Exit code 1 on any FAIL. WARN = degraded but not broken.
Every run appends one JSON line per provider (all checks + numbers) to
<ex6.get_folder("providers_test")>/test-logs.jsonl. Compare against past runs to spot drift
(e.g. overhead tokens creeping up, cache ratios dropping). See .plans/provider-tests.md.
"""
import json
import random
import struct
import sys
import tempfile
import time
import zlib
from contextlib import closing
from datetime import datetime
from unittest.mock import patch

import ex6
from _ex6 import provider_openrouter, provider_anthropic, provider_openai
from _ex6.models import M

PROVIDERS = {
    # max_overhead: input tokens of a near-empty prompt; catches backends silently injecting context.
    "anthropic": dict(model=M.HAIKU_45.id, invoker=provider_anthropic.invoke_anthropic, usage=("anthropic:usage", "utilization"),
                      min_ratio=0.8, max_overhead=2000),
    "openai": dict(model=M.GPT_LUNA_LATEST.id, invoker=provider_openai.invoke_openai, usage=("openai:usage", "percent"),
                   min_ratio=0.5, max_overhead=500),
    "openrouter": dict(model=M.HAIKU_45.id, invoker=provider_openrouter.invoke_openrouter, usage=None,
                       min_ratio=0.8, max_overhead=500),
}
CANARIES = ["invalid output", "unexpected control request", "stalled", "exception", "diverged", "malformed", "failed:"]
TRICKY_TEXT = 'héllo "wörld" \\ back\\slash, ünïcode ✓ 日本語'
TRICKY_ITEMS = ["a b", "{json: 1}", "line1\nline2"]
COLORS = {"red": (255, 0, 0), "green": (0, 200, 0), "blue": (0, 0, 255),
          "magenta": (255, 0, 255), "yellow": (255, 255, 0), "aqua": (0, 255, 255)}
COLOR_LIST = ", ".join(COLORS)

app = None  # isolated App, built by run_tests
checks = []  # (provider, name, status, detail)


def check(prov, name, ok, detail="", warn=False):
    status = "PASS" if ok else ("WARN" if warn else "FAIL")
    checks.append((prov, name, status, detail))
    print(f"  {status} {name}  {detail}")


# --- tools ---

def _record(ctx, name, args):
    ctx.data_volatile.setdefault("calls", []).append((len(ctx.data_volatile.get("results", [])), name, args))


def echo_args(ctx: ex6.Context, text: str, n: int, items: list[str]):
    """Records its arguments. Pass values exactly as the user gives them."""
    _record(ctx, "echo_args", dict(text=text, n=n, items=items))
    return "recorded"


def big_result(ctx: ex6.Context):
    """Returns a large document containing a secret code."""
    _record(ctx, "big_result", {})
    filler = "".join(f"row {i:05d}: nothing to see here, keep reading.\n" for i in range(1800))  # ~80KB
    return filler + f"The secret code is {ctx.data['test:nonce']}."


def noop(ctx: ex6.Context):
    """Does nothing; returns empty output."""
    _record(ctx, "noop", {})
    return ""


def _png(size, rgb=None):
    """Solid color, or random noise (incompressible, ~3 bytes/pixel) when rgb is None."""
    row = lambda: b"\x00" + (bytes(rgb) * size if rgb else random.randbytes(size * 3))
    raw = b"".join(row() for _ in range(size))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    path = tempfile.mktemp(suffix=".png")
    with open(path, "wb") as f:
        f.write(png)
    return ex6.ImageAttachment(path, "image/png", size, size)


def color_swatch(ctx: ex6.Context):
    """Returns an image of a solid color swatch."""
    _record(ctx, "color_swatch", {})
    rgb = COLORS[ctx.data["test:color"]]
    return ex6.ToolResult("Here is the swatch.", (_png(64, rgb),))


# --- harness ---

def recorded(fn):
    def invoke(ctx):
        with closing(fn(ctx)) as stream:
            for item in stream:
                if isinstance(item, ex6.LLMResult):
                    ctx.data_volatile.setdefault("results", []).append(item)
                yield item
    return invoke


def make_ctx(cfg, name, system, tools=()):
    ctx = ex6.Context(app, name, model=cfg["model"], invoke_llm=recorded(cfg["invoker"]), yolo=True,
                      messages=[ex6.Message("system", system, tools=list(tools))])
    ctx.data["test:nonce"] = f"{random.randint(0, 16**8):08x}"
    ctx.data["test:color"] = random.choice(list(COLORS))
    return ctx


def run(ctx, prompt, timeout=300, attachments=None):
    """Returns (reply text, LLMResults of this turn)."""
    start = len(ctx.data_volatile.get("results", []))
    ctx.invoke(prompt, attachments=attachments)
    t = time.time()
    while ctx.is_running():
        if time.time() - t > timeout:
            ctx.stop_early = True
            raise TimeoutError(f"{ctx.name}: no reply after {timeout}s")
        time.sleep(0.2)
    results = ctx.data_volatile.get("results", [])[start:]
    last = [m for m in ctx.get_messages() if m.role == "assistant"]
    return (last[-1].content if last else ""), results


def errors(results):
    return [r.error for r in results if r.error]


# --- tests ---

def test_basic(p, cfg, nums):
    ctx = make_ctx(cfg, "t_basic", "Reply with only the word you are given, nothing else.")
    nonce = ctx.data["test:nonce"]
    reply, rs = run(ctx, nonce)
    check(p, "basic: no error", not errors(rs), str(errors(rs)))
    check(p, "basic: system prompt honored", reply.strip() == nonce, repr(reply[:200]))
    nums["overhead_tokens"] = overhead = rs[-1].input_tokens if rs else 0
    check(p, "basic: overhead tokens", 0 < overhead <= cfg["max_overhead"], f"in={overhead} max={cfg['max_overhead']}")
    if cfg["usage"]:
        key, field = cfg["usage"]
        check(p, "basic: usage captured (footer)", field in app.plugin_data.get(key, {}), str(app.plugin_data.get(key)))


def test_tool_args(p, cfg, nums):
    ctx = make_ctx(cfg, "t_args", "You are a test agent. Follow instructions exactly.", [echo_args])
    expected = dict(text=TRICKY_TEXT, n=-42, items=TRICKY_ITEMS)
    reply, rs = run(ctx, "Call echo_args once with exactly these JSON args, then reply 'done':\n" + json.dumps(expected, ensure_ascii=False))
    calls = [a for _, name, a in ctx.data_volatile.get("calls", []) if name == "echo_args"]
    check(p, "tool args: no error", not errors(rs), str(errors(rs)))
    check(p, "tool args: called", len(calls) == 1, f"{len(calls)} calls")
    check(p, "tool args: exact round-trip", calls[:1] == [expected], repr(calls[:1]))


def test_tool_results(p, cfg, nums):
    ctx = make_ctx(cfg, "t_results", "You are a test agent. Follow instructions exactly.", [big_result, noop])
    reply, rs = run(ctx, "In a single turn, call big_result and noop in parallel. Then reply with only the secret code.")
    calls = ctx.data_volatile.get("calls", [])
    check(p, "tool results: no error", not errors(rs), str(errors(rs)))
    check(p, "tool results: parallel calls", sorted(n for _, n, _ in calls) == ["big_result", "noop"] and len({i for i, _, _ in calls}) == 1,
          repr([(i, n) for i, n, _ in calls]))
    check(p, "tool results: big result not truncated", ctx.data["test:nonce"] in reply, repr(reply[:200]))


def test_image(p, cfg, nums):
    ctx = make_ctx(cfg, "t_image", "You are a test agent. Follow instructions exactly.", [color_swatch])
    reply, rs = run(ctx, f"Call color_swatch, then reply with only the color name, one of: {COLOR_LIST}.")
    check(p, "image in tool result: no error", not errors(rs), str(errors(rs)))
    check(p, "image in tool result: seen", ctx.data["test:color"] in reply.lower(), f"want {ctx.data['test:color']}, got {reply[:100]!r}")


def test_big_images(p, cfg, nums):
    # ~1MB each: big stdin writes once deadlocked the claude pipe (stall, no error).
    ctx = make_ctx(cfg, "t_big_images", "You are a test agent. Follow instructions exactly.")
    images = {"[image1]": _png(600), "[image2]": _png(600)}
    reply, rs = run(ctx, "How many images did I send? Reply with only the number.", timeout=90, attachments=images)
    check(p, "big user images: no error", not errors(rs), str(errors(rs)))
    check(p, "big user images: both seen", "2" in reply, repr(reply[:100]))


def test_caching(p, cfg, nums):
    pad = "".join(f"Reference entry {i:04d}: code {i * 7919 % 10007:05d}, unused.\n" for i in range(700))  # ~7k tokens
    ctx = make_ctx(cfg, "t_cache", f"Run {random.random()}. You are a test agent; be brief.\n\n{pad}", [noop])
    if p == "openrouter":
        provider_openrouter.cache_manually(ctx)
    procs, rs = [], []
    for prompt in ["Say hi.", "Say bye.", "Call noop, then say done."]:
        _, turn = run(ctx, prompt)
        rs += turn
        procs.append(id(provider_anthropic._processes.get(ctx)))
    check(p, "cache: no error", not errors(rs) and len(rs) >= 4, f"{len(rs)} results, {errors(rs)}")
    ratios = [round(r.cached_tokens / max(r.input_tokens, 1), 3) for r in rs]
    nums["cache_ratios"] = ratios
    nums["cache_usage"] = [(r.input_tokens, r.cached_tokens, r.cache_write_tokens) for r in rs]
    later = ratios[1:]
    if p == "openai":
        check(p, "cache: any hits after turn 1", any(later), str(ratios))
        check(p, "cache: hit ratio", min(later or [0]) >= cfg["min_ratio"], str(ratios), warn=any(later))
        return
    check(p, "cache: hit ratio after turn 1", min(later or [0]) >= cfg["min_ratio"], str(ratios))
    rewrites = [r.cache_write_tokens for r in rs[1:] if r.cache_write_tokens > 0.5 * r.input_tokens]
    check(p, "cache: no full-prefix rewrites", not rewrites, str(nums["cache_usage"]))
    if p == "anthropic":
        created = (rs[0].raw_usage.get("cache_creation") or {}) if rs else {}
        check(p, "cache: 1h TTL used", created.get("ephemeral_1h_input_tokens", 0) > 0, str(created))
        check(p, "cache: claude process never restarted", len(set(procs)) == 1 and procs[0] != id(None), str(procs))


def test_resume(p, cfg, nums):
    # A fork has no backend state: history (tool calls, results, images) must be rebuilt structurally.
    ctx = make_ctx(cfg, "t_resume", "You are a test agent. Follow instructions exactly.", [color_swatch, big_result, noop])
    _, rs = run(ctx, "Call color_swatch, big_result and noop in parallel, then reply 'ok'. Don't mention the results.")
    fork = ctx.fork("t_resume_fork")
    reply, rs2 = run(fork, f"Reply with only: <swatch color, one of: {COLOR_LIST}> <secret code>")
    check(p, "resume: no error", not errors(rs + rs2), str(errors(rs + rs2)))
    check(p, "resume: tool result image seen", fork.data["test:color"] in reply.lower(), repr(reply[:100]))
    check(p, "resume: tool result text seen", fork.data["test:nonce"] in reply, repr(reply[:100]))
    reply, rs3 = run(fork, "Reply with only the secret code again.")  # anthropic: session file is gone by now
    check(p, "resume: history kept on next turn", not errors(rs3) and fork.data["test:nonce"] in reply, repr(reply[:100]))


TESTS = [test_basic, test_tool_args, test_tool_results, test_image, test_big_images, test_caching, test_resume]


def run_tests(names=("anthropic", "openai")):
    global app
    app = ex6.App()
    for mod in (provider_openrouter, provider_anthropic, provider_openai):
        app.register_module(mod)
    checks.clear()
    for p in names:
        cfg = PROVIDERS[p]
        print(f"\n== {p} ({cfg['model']})")
        start, nums = len(checks), {}
        with patch.object(app, "debug_print", wraps=app.debug_print) as debug:
            for test in TESTS:
                try:
                    test(p, cfg, nums)
                except Exception as e:
                    check(p, f"{test.__name__}: crashed", False, repr(e))
        lines = [" ".join(str(a) for a in call.args) for call in debug.call_args_list]
        hits = [line for line in lines if any(c in line.lower() for c in CANARIES)]
        check(p, "debug canaries", not hits, "\n      ".join([""] + hits[:20]))
        for ctx in list(provider_anthropic._processes):
            provider_anthropic._processes.pop(ctx).close()
        folder = ex6.get_folder("providers_test")
        folder.mkdir(exist_ok=True)
        with open(folder / "test-logs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"), "provider": p, "model": cfg["model"],
                                "nums": nums, "checks": [c[1:] for c in checks[start:]]}) + "\n")
    fails = [f"{p}: {n}  {d}" for p, n, s, d in checks if s == "FAIL"]
    print(f"\n{len(fails)} FAIL, {sum(c[2] == 'WARN' for c in checks)} WARN, {len(checks)} checks")
    return fails


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(1 if run_tests(sys.argv[1:] or ("anthropic", "openai")) else 0)
