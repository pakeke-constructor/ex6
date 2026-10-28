# Dependencies: openai, httpx
# Install: python -m pip install openai httpx
# Requires ChatGPT login credentials in ~/.codex/auth.json (via Codex CLI).


import json
import ex6
import openai
import os
import time
import base64
from _ex6.provider import _log_invoke


# ---------------------------------------------------------------------------
# ChatGPT-subscription backend (Codex Responses API). Uses your ChatGPT login
# from ~/.codex/auth.json instead of paying API prices. $0 metered cost.
# ---------------------------------------------------------------------------

_CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"  # OpenAI's public Codex OAuth app
_CODEX_VERSION = "0.81.0"  # codex_cli_rs version we impersonate; bump occasionally


def _user_agent():
    import platform
    return (f"codex_cli_rs/{_CODEX_VERSION} "
            f"({platform.system()} {platform.release()}; {platform.machine()})")


def _auth_path():
    from pathlib import Path
    home = os.environ.get("CODEX_HOME") or (Path.home() / ".codex")
    return Path(home) / "auth.json"


def _codex_auth():
    """Read ChatGPT OAuth token + account id from the Codex CLI login."""
    tokens = json.loads(_auth_path().read_text())["tokens"]
    return tokens["access_token"], tokens["account_id"]


def _refresh_codex_token(debug):
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
    debug("[codex] refreshed access token")
    return tokens["access_token"]


def _capture_usage(headers, app):
    import time
    pct = headers.get("x-codex-primary-used-percent")
    if pct is None:
        return
    app.plugin_data.setdefault("openai:usage", {}).update(
        percent=float(pct),
        reset_after=float(headers.get("x-codex-primary-reset-after-seconds", 0)),
        window=float(headers.get("x-codex-primary-window-minutes", 0)),
        secondary=float(headers.get("x-codex-secondary-used-percent", 0)),
        secondary_window=float(headers.get("x-codex-secondary-window-minutes", 0)),
        ts=time.time(),
    )


def _codex_client(access_token, account_id):
    return openai.OpenAI(
        base_url="https://chatgpt.com/backend-api/codex",
        api_key=access_token,
        max_retries=0,
        default_headers={
            "chatgpt-account-id": account_id,
            "OpenAI-Beta": "responses=experimental",
            "originator": "codex_cli_rs",
            "User-Agent": _user_agent(),
        },
    )


def _input_image(attachment):
    if not isinstance(attachment, ex6.ImageAttachment):
        raise ValueError(f"Unsupported attachment type: {type(attachment).__name__}")
    if not os.path.isfile(attachment.path):
        raise ValueError(f"Attachment file is missing: {attachment.path}")
    with open(attachment.path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return {"type": "input_image", "image_url": f"data:{attachment.mime_type};base64,{encoded}",
            "detail": attachment.detail}


def _responses_tool_output(result: ex6.ToolResult):
    if not result.attachments:
        return result.text
    return [{"type": "input_text", "text": result.text}] + [_input_image(a) for a in result.attachments]


def _to_responses_input(ctx: ex6.Context):
    """Convert ex6 chat messages -> (instructions, Responses `input` items)."""
    instructions = []
    items = []
    for m in ctx.get_messages():
        c = m.get_msg(ctx)
        if m.role == "system":
            instructions.append(c if isinstance(c, str) else json.dumps(c))
        elif m.role == "tool":
            output = _responses_tool_output(c) if isinstance(c, ex6.ToolResult) else c
            items.append({"type": "function_call_output",
                          "call_id": m.tool_call_id, "output": output})
        elif m.role == "assistant":
            if c:
                items.append({"type": "message", "role": "assistant",
                              "content": [{"type": "output_text", "text": c}]})
            for tc in (m.tool_calls or []):
                items.append({"type": "function_call", "call_id": tc["id"],
                              "name": tc["name"], "arguments": json.dumps(tc["args"])})
        else:  # user
            content = [{"type": "input_text", "text": c}]
            for token, attachment in m.attachments.items():
                content += [{"type": "input_text", "text": token}, _input_image(attachment)]
            items.append({"type": "message", "role": "user", "content": content})
    return "\n\n".join(instructions), items


# Native tools: add to a context's tools explicitly. name -> Responses tool, sent verbatim.
WEB_SEARCH = ex6.NativeTool("web_search")
NATIVE_TOOLS = {"web_search": {"type": "web_search", "external_web_access": True}}
WEB_TOOLS = [WEB_SEARCH]


@ex6.handler
def invoke_llm(ctx: ex6.Context):
    if ctx.model.startswith("openai/") and ":" not in ctx.model:
        return invoke_openai


def invoke_openai(ctx: ex6.Context):
    unknown = [n for n in ctx.get_native_tools() if n not in NATIVE_TOOLS]
    if unknown:
        result = ex6.LLMResult(error=f"openai provider doesn't support native tools: {unknown}")
        _log_invoke(ctx, [], result)
        yield result
        return
    access_token, account_id = _codex_auth()
    instructions, input_items = _to_responses_input(ctx)

    # chat-completions tool schema -> flat Responses schema
    tools = [{"type": "function", "name": t["function"]["name"],
              "description": t["function"]["description"],
              "parameters": t["function"]["parameters"]}
             for t in (ctx.get_tool_schemas() or [])]
    tools += [NATIVE_TOOLS[n] for n in ctx.get_native_tools()]

    body = {"store": False, "include": ["reasoning.encrypted_content"], "prompt_cache_key": ctx.name}
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
        _capture_usage(raw.headers, ctx.app)
        return raw.parse()

    ctx.app.debug_print(f"[codex] model={ctx.model} items={len(input_items)}")
    try:
        try:
            stream = start(access_token)
        except openai.AuthenticationError:
            # Access token expired — refresh via the OAuth refresh token and retry once.
            ctx.app.debug_print("[codex] 401 — refreshing token")
            stream = start(_refresh_codex_token(ctx.app.debug_print))
    except Exception as e:
        ctx.app.debug_print(f"[codex] API EXCEPTION: {e}")
        result = ex6.LLMResult(error=str(e))
        _log_invoke(ctx, input_items, result)
        yield result
        return

    input_tokens, output_tokens, cached_tokens = 0, 0, 0
    raw_usage = {}
    finish_reason = "stop"
    tool_calls = []
    native_calls = []
    try:
        for event in stream:
            t = event.type
            if t == "response.output_text.delta":
                yield ex6.ResponseChunk("text", event.delta)
            elif t == "response.reasoning_summary_text.delta":
                yield ex6.ResponseChunk("cot", event.delta)
            elif t == "response.output_item.done" and event.item.type == "function_call":
                it = event.item
                try:
                    args = json.loads(it.arguments) if it.arguments else {}
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"malformed tool args for {it.name}: {e}: {it.arguments[:500]!r}")
                if not isinstance(args, dict):
                    raise RuntimeError(f"malformed tool args for {it.name}: not an object: {it.arguments[:500]!r}")
                tool_calls.append({"id": it.call_id, "name": it.name, "args": args})
            elif t == "response.output_item.done" and event.item.type == "web_search_call":
                item = event.item.model_dump(exclude_none=True)
                status = item.get("status")
                action = item.get("action", {})
                args = {k: action[k] for k in ("query", "url", "pattern") if k in action} or action
                native_calls.append({"id": item["id"], "name": "web_search", "args": args,
                                     "result": status if status == "completed" else f"ERROR: {status}"})
                ctx.app.debug_print(f"[codex] native web_search_call: {json.dumps(item)}")
                yield ex6.ResponseChunk("cot", f"\n[web_search] {json.dumps(item.get('action'))}\n")
            elif t == "response.completed":
                u = event.response.usage
                if u:
                    raw_usage = u.model_dump()
                    input_tokens = u.input_tokens or 0
                    output_tokens = u.output_tokens or 0
                    d = getattr(u, "input_tokens_details", None)
                    cached_tokens = getattr(d, "cached_tokens", 0) or 0 if d else 0
            elif t in ("response.failed", "error"):
                err = getattr(getattr(event, "response", None), "error", None) or getattr(event, "message", t)
                raise RuntimeError(err)
    except Exception as e:
        ctx.app.debug_print(f"[codex] stream exception: {e}")
        result = ex6.LLMResult(error=str(e))
        _log_invoke(ctx, input_items, result, cached_tokens)
        yield result
        return

    if tool_calls:
        finish_reason = "tool_calls"

    ctx.app.budget.add_cost(0)  # subscription — no API charge
    result = ex6.LLMResult(input_tokens, output_tokens, tool_calls, finish_reason, cost=0, native_calls=native_calls,
                           cached_tokens=cached_tokens, raw_usage=raw_usage)
    ctx.app.debug_print(f"[codex] result: in={input_tokens} out={output_tokens} cached={cached_tokens} tools={len(tool_calls)} native={len(native_calls)}")
    _log_invoke(ctx, input_items, result, cached_tokens)
    yield result


def _fmt_reset(secs):
    secs = max(0, int(secs))
    days, remainder = divmod(secs, 86400)
    h, m = remainder // 3600, (remainder % 3600) // 60
    if days:
        return f"{days}d {h}h"
    return f"{h}h{m:02d}m" if h else f"{m}m"


def _render_openai_usage(tui, buf, x, y, w, th):
    import time
    usage = tui.app.plugin_data.setdefault("openai:usage", {})
    if "percent" not in usage:
        msg = "(unknown usage)"
        buf.puts(x + w - len(msg) - 2, y, msg, txt_color=th.muted)
        return
    pct = usage["percent"]
    remaining = usage["reset_after"] - (time.time() - usage["ts"])
    filled = min(10, max(0, round(pct / 10)))
    days = usage["window"] / 1440
    window = f"{days:g}d" if days >= 1 else f"{usage['window'] / 60:g}h"
    mid = f" {pct:.0f}% used / {window}, resets in {_fmt_reset(remaining)}"
    secondary = ""
    if usage["secondary_window"]:
        secondary = f"  ({usage['secondary']:.0f}% secondary used)"
    bx = x + w - (10 + len(mid) + len(secondary)) - 2
    buf.puts(bx, y, "█" * filled, txt_color=th.accent)
    buf.puts(bx + filled, y, "░" * (10 - filled), txt_color=th.muted)
    buf.puts(bx + 10, y, mid, txt_color=th.accent_alt)
    buf.puts(bx + 10 + len(mid), y, secondary, txt_color=th.muted)


@ex6.handler
def render_work_mode_footer(tui, buf, r, ctx):
    if ctx.get_invoker() is not invoke_openai:
        return False
    x, y, w, h = r
    th = tui.app.theme
    on = ctx.yolo
    buf.puts(x, y, "  yolo ON" if on else "  yolo OFF",
             txt_color=th.success if on else th.muted)
    _render_openai_usage(tui, buf, x, y, w, th)
    return True

