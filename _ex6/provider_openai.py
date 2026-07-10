
import json
import ex6
import openai
import os
from _ex6.provider import _log_invoke


# ---------------------------------------------------------------------------
# ChatGPT-subscription backend (Codex Responses API). Uses your ChatGPT login
# from ~/.codex/auth.json instead of paying API prices. $0 metered cost.
# ---------------------------------------------------------------------------

_CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"  # OpenAI's public Codex OAuth app


def _auth_path():
    from pathlib import Path
    home = os.environ.get("CODEX_HOME") or (Path.home() / ".codex")
    return Path(home) / "auth.json"


def _codex_auth():
    """Read ChatGPT OAuth token + account id from the Codex CLI login."""
    tokens = json.loads(_auth_path().read_text())["tokens"]
    return tokens["access_token"], tokens["account_id"]


def _refresh_codex_token():
    """Exchange the refresh token for a fresh access token and write the rotated
    tokens back to auth.json. Returns the new access token.

    The refresh token is single-use, so we re-read auth.json here (Codex may have
    rotated it already) and persist the new one atomically."""
    import httpx
    from datetime import datetime, timezone
    path = _auth_path()
    data = json.loads(path.read_text())
    tokens = data["tokens"]
    resp = httpx.post(
        "https://auth.openai.com/oauth/token",
        json={
            "client_id": _CODEX_CLIENT_ID,
            "grant_type": "refresh_token",
            "refresh_token": tokens["refresh_token"],
            "scope": "openid profile email",
        },
        timeout=30,
    )
    resp.raise_for_status()
    new = resp.json()
    tokens["access_token"] = new["access_token"]
    if new.get("id_token"):
        tokens["id_token"] = new["id_token"]
    if new.get("refresh_token"):  # single-use — must persist the rotation
        tokens["refresh_token"] = new["refresh_token"]
    data["last_refresh"] = datetime.now(timezone.utc).isoformat()
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)  # atomic swap; auth.json is live-read by the Codex CLI
    ex6.debug_print("[codex] refreshed access token")
    return tokens["access_token"]


# Subscription usage, scraped from the Codex rate-limit response headers.
# primary = 5h window. Populated on every invoke; read by the footer override.
_usage = {}  # {"percent", "reset_after", "ts"}


def _capture_usage(headers):
    import time
    pct = headers.get("x-codex-primary-used-percent")
    if pct is None:
        return
    _usage.update(
        percent=float(pct),
        reset_after=float(headers.get("x-codex-primary-reset-after-seconds", 0)),
        weekly=float(headers.get("x-codex-secondary-used-percent", 0)),
        ts=time.time(),
    )


def _codex_client(access_token, account_id):
    return openai.OpenAI(
        base_url="https://chatgpt.com/backend-api/codex",
        api_key=access_token,
        default_headers={
            "chatgpt-account-id": account_id,
            "OpenAI-Beta": "responses=experimental",
            "originator": "codex_cli_rs",
            "User-Agent": "codex_cli_rs/0.81.0 (Windows 11; x86_64)",
        },
    )


def _to_responses_input(ctx: ex6.Context):
    """Convert ex6 chat messages -> (instructions, Responses `input` items)."""
    instructions = []
    items = []
    for m in ctx.messages:
        c = m.get_msg(ctx)
        if m.role == "system":
            instructions.append(c if isinstance(c, str) else json.dumps(c))
        elif m.role == "tool":
            items.append({"type": "function_call_output",
                          "call_id": m.tool_call_id, "output": c})
        elif m.role == "assistant":
            if c:
                items.append({"type": "message", "role": "assistant",
                              "content": [{"type": "output_text", "text": c}]})
            for tc in (m.tool_calls or []):
                items.append({"type": "function_call", "call_id": tc["id"],
                              "name": tc["name"], "arguments": json.dumps(tc["args"])})
        else:  # user
            items.append({"type": "message", "role": "user",
                          "content": [{"type": "input_text", "text": c}]})
    return "\n\n".join(instructions), items


def invoke_llm(ctx: ex6.Context):
    access_token, account_id = _codex_auth()
    instructions, input_items = _to_responses_input(ctx)

    # chat-completions tool schema -> flat Responses schema
    tools = [{"type": "function", "name": t["function"]["name"],
              "description": t["function"]["description"],
              "parameters": t["function"]["parameters"]}
             for t in (ctx.get_tool_schemas() or [])]

    body = {"store": False, "include": ["reasoning.encrypted_content"]}
    kw = {}
    if ctx.reasoning != "none":
        kw["reasoning"] = {"effort": ctx.reasoning, "summary": "auto"}

    def start(token):
        raw = _codex_client(token, account_id).responses.with_raw_response.create(
            model=ctx.model.split("/", 1)[1],
            instructions=instructions or "You are a helpful coding assistant.",
            input=input_items,
            tools=tools or openai.NOT_GIVEN,
            stream=True,
            extra_body=body,
            timeout=120,
            **kw,
        )
        _capture_usage(raw.headers)
        return raw.parse()

    ex6.debug_print(f"[codex] model={ctx.model} items={len(input_items)}")
    try:
        try:
            stream = start(access_token)
        except openai.AuthenticationError:
            # Access token expired — refresh via the OAuth refresh token and retry once.
            ex6.debug_print("[codex] 401 — refreshing token")
            stream = start(_refresh_codex_token())
    except Exception as e:
        ex6.debug_print(f"[codex] API EXCEPTION: {e}")
        result = ex6.LLMResult(error=str(e))
        _log_invoke(ctx, input_items, result)
        yield result
        return

    input_tokens, output_tokens, cached_tokens = 0, 0, 0
    finish_reason = "stop"
    tool_calls = []
    try:
        for event in stream:
            t = event.type
            if t == "response.output_text.delta":
                yield ex6.ResponseChunk("text", event.delta)
            elif t == "response.reasoning_summary_text.delta":
                yield ex6.ResponseChunk("cot", event.delta, len(event.delta))
            elif t == "response.output_item.done" and event.item.type == "function_call":
                it = event.item
                try:
                    args = json.loads(it.arguments) if it.arguments else {}
                except:
                    args = {}
                tool_calls.append({"id": it.call_id, "name": it.name, "args": args})
            elif t == "response.completed":
                u = event.response.usage
                if u:
                    input_tokens = u.input_tokens or 0
                    output_tokens = u.output_tokens or 0
                    d = getattr(u, "input_tokens_details", None)
                    cached_tokens = getattr(d, "cached_tokens", 0) or 0 if d else 0
            elif t in ("response.failed", "error"):
                err = getattr(getattr(event, "response", None), "error", None) or getattr(event, "message", t)
                raise RuntimeError(err)
    except Exception as e:
        ex6.debug_print(f"[codex] stream exception: {e}")
        result = ex6.LLMResult(error=str(e))
        _log_invoke(ctx, input_items, result, cached_tokens)
        yield result
        return

    for tc in tool_calls:
        yield ex6.ResponseChunk("tool", json.dumps(tc))
    if tool_calls:
        finish_reason = "tool_calls"

    ex6.add_cost(0)  # subscription — no API charge
    result = ex6.LLMResult(input_tokens, output_tokens, tool_calls, finish_reason, cost=0)
    ex6.debug_print(f"[codex] result: in={input_tokens} out={output_tokens} cached={cached_tokens} tools={len(tool_calls)}")
    _log_invoke(ctx, input_items, result, cached_tokens)
    yield result


def _fmt_reset(secs):
    secs = max(0, int(secs))
    h, m = secs // 3600, (secs % 3600) // 60
    return f"{h}h{m:02d}m" if h else f"{m}m"


@ex6.override
def render_work_mode_footer(tui, buf, r, ctx):
    """Default yolo indicator, plus a subscription-usage bar for codex contexts."""
    import time
    x, y, w, h = r
    th = ex6.get_theme()
    on = ctx.yolo
    buf.puts(x, y, "  yolo ON" if on else "  yolo OFF",
             txt_color=th.success if on else th.muted)

    if ctx.invoke_llm is not invoke_llm or "percent" not in _usage:
        return

    pct = _usage["percent"]
    remaining = _usage["reset_after"] - (time.time() - _usage["ts"])
    filled = min(10, max(0, round(pct / 10)))
    mid = f" {pct:.0f}% resets in {_fmt_reset(remaining)}"
    weekly = f"  ({_usage['weekly']:.0f}% weekly used)"
    bx = x + w - (10 + len(mid) + len(weekly)) - 2
    buf.puts(bx, y, "█" * filled, txt_color=th.accent)
    buf.puts(bx + filled, y, "░" * (10 - filled), txt_color=th.muted)
    buf.puts(bx + 10, y, mid, txt_color=th.accent_alt)
    buf.puts(bx + 10 + len(mid), y, weekly, txt_color=th.muted)

