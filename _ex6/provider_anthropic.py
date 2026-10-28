# Dependencies: openai (via provider.py's shared logging helper).
# Install: python -m pip install openai
# External: authenticated Claude Code CLI (`claude`) on PATH; no anthropic SDK needed.

"""
Claude subscription backend: drives a long-lived `claude -p` process.
ex6 tools are exposed as an SDK MCP server (control messages over claude's own
stdin/stdout); ex6 itself runs them and hands results back by tool_use id.

Why it's shaped this way:
We're driving an external process we don't control, over pipes carrying chat
events and MCP control messages interleaved, that can stall, reorder, crash, or
quietly disagree with us. Earlier versions hung forever after tool calls. So the priority is:
never hang, never silently diverge from what ex6 thinks happened.

The claude process is treated as a disposable cache of ex6's history, not as
state worth preserving. When anything is off, we throw it away and rebuild from
history. That costs a lost prompt cache; trying to repair a confused process
costs far more, and is where bugs hide. Keeping the moving parts few (one
consumer, state derived from history) is what keeps that guarantee cheap.
See .plans/anthropic-worker.md.

Context overhead (can't be removed without --bare, which needs an API key):
- system: "You are a Claude agent, built on Anthropic's Claude Agent SDK."
- first user msg: <system-reminder>s with environment (cwd, platform, OS), model name, date.
- small "<total_tokens> left" reminders appended to some turns.
See _notes/claude-p-findings.md.
"""
import atexit
import base64
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time

import ex6
from _ex6.provider import _log_invoke


_INACTIVITY_TIMEOUT = 300  # seconds without any claude/MCP event mid-turn
_CALL_TIMEOUT = 30  # seconds for claude to send tools/call for a result ex6 has ready


def _capture_usage(event, app):
    info = event.get("rate_limit_info") or {}
    windows = info.get("unifiedWindows") or {}
    usage = app.plugin_data.setdefault("anthropic:usage", {})
    for key in ("five_hour", "seven_day"):
        w = windows.get(key)
        if w:
            usage[key] = {"utilization": w.get("utilization", 0), "resets_at": w.get("resetsAt", 0)}
    # Show whichever window is closest to its limit. Top-level utilization is a fallback
    # only; the CLI usually just reports it inside unifiedWindows.
    rate_type = info.get("rateLimitType", "")
    util = info.get("utilization")
    resets_at = info.get("resetsAt", 0)
    for key in ("five_hour", "seven_day"):
        w = windows.get(key)
        if w and (util is None or w.get("utilization", 0) > util):
            rate_type, util, resets_at = key, w.get("utilization", 0), w.get("resetsAt", 0)
    if util is not None:
        usage["utilization"] = util
        usage["resets_at"] = resets_at
        usage["rate_limit_type"] = rate_type
    usage["ts"] = time.time()
    app.debug_print(f"[claude] rate_limit: {rate_type} util={util}")


def _fmt_reset(secs):
    secs = max(0, int(secs))
    days, remainder = divmod(secs, 86400)
    h, m = remainder // 3600, (remainder % 3600) // 60
    if days:
        return f"{days}d {h}h"
    return f"{h}h{m:02d}m" if h else f"{m}m"


@ex6.handler
def render_work_mode_footer(tui, buf, r, ctx):
    if ctx.get_invoker() is not invoke_anthropic:
        return False
    x, y, w, h = r
    th = tui.app.theme
    on = ctx.yolo
    buf.puts(x, y, "  yolo ON" if on else "  yolo OFF",
             txt_color=th.success if on else th.muted)

    usage = tui.app.plugin_data.get("anthropic:usage")
    if not usage or "utilization" not in usage:
        msg = "(unknown usage)"
        buf.puts(x + w - len(msg) - 2, y, msg, txt_color=th.muted)
        return True
    pct = usage["utilization"] * 100
    remaining = usage["resets_at"] - time.time()
    rate_type = usage.get("rate_limit_type", "")
    window = "5h" if "five_hour" in rate_type else "7d" if "seven_day" in rate_type else rate_type
    filled = min(10, max(0, round(pct / 10)))
    mid = f" {pct:.0f}% used / {window}, resets in {_fmt_reset(remaining)}"
    bx = x + w - (10 + len(mid)) - 2
    buf.puts(bx, y, "█" * filled, txt_color=(220, 140, 40))
    buf.puts(bx + filled, y, "░" * (10 - filled), txt_color=th.muted)
    buf.puts(bx + 10, y, mid, txt_color=(200, 150, 70))
    return True


# ctx -> _ClaudeProcess. Module-level (not data_volatile) so a stale process is
# closed when replaced; otherwise it'd leak, as reader threads keep it alive.
_processes = {}
atexit.register(lambda: [p.close() for p in list(_processes.values())])


@ex6.context_removed
def _close_removed_context(ctx):
    process = _processes.pop(ctx, None)
    if process:
        process.close()


def _model_name(model):
    # "anthropic/claude-sonnet-4.6" -> "claude-sonnet-4-6"
    return model.split("/", 1)[-1].replace(".", "-")


def _mcp_tools(ctx):
    tools = []
    for schema in ctx.get_tool_schemas():
        fn = schema["function"]
        tools.append({
            "name": fn["name"],
            "description": fn["description"],
            "inputSchema": fn["parameters"],
            "_meta": {"anthropic/maxResultSizeChars": 500000},  # else claude swaps big results for a file preview
        })
    return tools


# Native tools: add to a context's tools explicitly. Names are claude's built-in tool names.
WEB_SEARCH = ex6.NativeTool("WebSearch")
WEB_FETCH = ex6.NativeTool("WebFetch")
WEB_TOOLS = [WEB_SEARCH, WEB_FETCH]


def _native_tools(ctx):
    names = ctx.get_native_tools()
    unknown = [n for n in names if n not in (WEB_SEARCH.__name__, WEB_FETCH.__name__)]
    if unknown:
        raise ValueError(f"claude provider doesn't support native tools: {unknown}")
    return names


def _system_prompt(ctx):
    parts = []
    for message in ctx.get_messages():
        if message.role == "system":
            value = message.get_msg(ctx)
            parts.append(value if isinstance(value, str) else json.dumps(value))
    return "\n\n".join(parts) or "You are a helpful coding assistant."


def _tool_result_payload(result):
    content = [{"type": "text", "text": result.text}]
    for attachment in result.attachments:
        if not isinstance(attachment, ex6.ImageAttachment):
            raise ValueError(f"Unsupported attachment type: {type(attachment).__name__}")
        with open(attachment.path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        content.append({"type": "image", "data": data, "mimeType": attachment.mime_type})
    return {"content": content}


def _digest(is_error, content):
    """Fingerprint of a tool result, as sent by ex6 or echoed back by claude."""
    if isinstance(content, list):
        content = "".join(b["text"] for b in content if b["type"] == "text")
    # claude replaces blank results with "(<tool> completed with no output)"
    if not content.strip() or re.fullmatch(r"\(\w+ completed with no output\)", content):
        content = ""
    return bool(is_error), hashlib.sha256(content.encode()).hexdigest()


def _user_content(text, attachments):
    """Plain text, or text + (token label, image) blocks when the user pasted images."""
    if not attachments:
        return text
    content = [{"type": "text", "text": text}]
    for token, attachment in attachments.items():
        if not os.path.isfile(attachment.path):
            raise ValueError(f"Attachment file is missing: {attachment.path}")
        with open(attachment.path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        content += [{"type": "text", "text": token},
                    {"type": "image", "source": {"type": "base64", "media_type": attachment.mime_type, "data": data}}]
    return content


def _bootstrap_prompt(ctx):
    messages = [m for m in ctx.get_messages() if m.role != "system"]
    attachments = {token: a for m in messages if m.role == "user" for token, a in m.attachments.items()}
    if len(messages) == 1 and messages[0].role == "user":
        return _user_content(messages[0].get_msg(ctx), attachments)

    lines = ["Continue this conversation. The transcript before the final user message is context, not instructions:"]
    for message in messages:
        value = message.get_msg(ctx)
        if isinstance(value, ex6.ToolResult):
            value = value.text
        if message.role == "assistant" and message.tool_calls:
            lines.append(f"[assistant tool calls] {json.dumps(message.tool_calls)}")
        if value:
            lines.append(f"[{message.role}] {value}")
    return _user_content("\n\n".join(lines), attachments)


class _ClaudeProcess:
    def __init__(self, ctx):
        self.ctx = ctx
        self.system = _system_prompt(ctx)
        self.tools = _mcp_tools(ctx)
        self.native_tools = _native_tools(ctx)
        self.native_calls = {}  # tool_use_id -> record, until claude echoes its result (may be a later pump)
        self.seen = ()  # ex6 messages claude has been given
        # ("claude", event) | ("call", tool_use_id, request_id, rpc_id) | ("exit", code)
        self.events = queue.Queue()
        self.waiting = {}  # tool_use_id -> (request_id, rpc_id) of a tools/call that arrived before ex6's result
        self.write_lock = threading.Lock()  # stdin is written by the reader thread and by invoke_anthropic
        self.closed = False

        self.tmp = tempfile.TemporaryDirectory(prefix="ex6-claude-", ignore_cleanup_errors=True)
        folder = Path(self.tmp.name)
        prompt_path = folder / "system.txt"
        mcp_path = folder / "mcp.json"
        prompt_path.write_text(self.system, encoding="utf-8")
        mcp_path.write_text(json.dumps({"mcpServers": {"ex6": {"type": "sdk", "name": "ex6"}}}), encoding="utf-8")

        exe = shutil.which("claude")
        if not exe:
            raise RuntimeError("claude CLI not found on PATH")
        args = [
            exe, "-p",
            "--input-format=stream-json",
            "--output-format=stream-json",
            "--include-partial-messages",
            "--verbose",
            "--setting-sources=",
            "--disable-slash-commands",
            "--tools=" + ",".join(self.native_tools),
            "--allowedTools=" + ",".join(self.native_tools),
            "--strict-mcp-config",
            "--mcp-config", str(mcp_path),
            "--system-prompt-file", str(prompt_path),
            "--dangerously-skip-permissions",
            "--no-session-persistence",
            "--model", _model_name(ctx.model),
        ]
        if ctx.reasoning != "none":
            args.extend(["--effort", ctx.reasoning])

        env = os.environ.copy()
        env.setdefault("CLAUDE_CODE_PROMPT_CACHE_TTL", "1h")
        env.setdefault("MCP_TOOL_TIMEOUT", str(2 * 3600 * 1000))  # ex6 tools may wait on the user
        env.setdefault("MAX_MCP_OUTPUT_TOKENS", "1000000")  # ex6 tools own output size
        env["CLAUDE_CODE_ATTRIBUTION_HEADER"] = "0"
        env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] = "1"
        env["CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS"] = "1"
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        env["CLAUDE_AGENT_SDK_MCP_NO_PREFIX"] = "1"  # tools named as ex6 names them
        self.proc = subprocess.Popen(
            args,
            cwd=ctx.cwd or os.getcwd(),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()
        self._send_event({"type": "control_request", "request_id": "init",
                          "request": {"subtype": "initialize", "hooks": None}})

    def _read_stdout(self):
        code = "stdout closed"
        try:
            for line in self.proc.stdout:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    event = None
                if not isinstance(event, dict):
                    self.ctx.app.debug_print(f"[claude] invalid output: {line.rstrip()}")
                    continue
                if event.get("type") == "control_request":
                    self._on_control(event)
                    continue
                if event.get("type") == "control_response":
                    continue
                if event.get("type") == "rate_limit_event":
                    _capture_usage(event, self.ctx.app)
                self.events.put(("claude", event))
            code = self.proc.wait()
        except Exception as e:
            code = f"stdout reader crashed: {e!r}"
        finally:
            self.events.put(("exit", code))

    def _read_stderr(self):
        for line in self.proc.stderr:
            self.ctx.app.debug_print(f"[claude] {line.rstrip()}")

    def _on_control(self, event):
        """Answers claude's control requests. Stateless MCP requests are answered here, since
        they can arrive when no pump runs; tools/call goes to the pump."""
        request_id = event["request_id"]
        request = event["request"]
        if request.get("subtype") != "mcp_message":
            self.ctx.app.debug_print(f"[claude] unexpected control request: {request.get('subtype')}")
            self._send_event({"type": "control_response", "response": {
                "subtype": "error", "request_id": request_id, "error": "unsupported"}})
            return
        message = request["message"]
        rpc_id = message.get("id")
        method = message.get("method")
        if method == "tools/call":
            tool_use_id = message["params"]["_meta"]["claudecode/toolUseId"]
            self.events.put(("call", tool_use_id, request_id, rpc_id))
            return
        if method == "initialize":
            result = {"protocolVersion": message["params"].get("protocolVersion", "2025-06-18"),
                      "capabilities": {"tools": {}},
                      "serverInfo": {"name": "ex6", "version": "1"}}
        elif method == "tools/list":
            result = {"tools": self.tools}
        else:
            result = {}
        # Every mcp_message needs a response, notifications included (no id), or claude stalls.
        reply = {"jsonrpc": "2.0", "result": result}
        if rpc_id is not None:
            reply["id"] = rpc_id
        self._send_mcp(request_id, reply)

    def _send_mcp(self, request_id, reply):
        self._send_event({"type": "control_response", "response": {
            "subtype": "success", "request_id": request_id, "response": {"mcp_response": reply}}})

    def _send_event(self, event):
        with self.write_lock:
            self.proc.stdin.write(json.dumps(event) + "\n")
            self.proc.stdin.flush()

    def new_messages(self, messages):
        """Messages ex6 added since claude's last turn, or None if the histories
        diverged (clear, truncate, edits, interrupted tool loop, changed prompt/tools)."""
        if self.closed or self.proc.poll() is not None:
            return None
        if self.system != _system_prompt(self.ctx) or self.tools != _mcp_tools(self.ctx):
            return None
        if self.native_tools != _native_tools(self.ctx):
            return None
        seen = self.seen
        if len(messages) <= len(seen) or any(a is not b for a, b in zip(seen, messages)):
            return None
        new = list(messages[len(seen):])
        # claude's own reply, appended by ex6. Missing means it was dropped (e.g. stopped).
        if new[0].role != "assistant":
            return None
        pending = {c["id"] for c in new[0].tool_calls or []}
        new = new[1:]
        if pending:
            ok = len(new) == len(pending) and all(m.role == "tool" for m in new)
            return new if ok and {m.tool_call_id for m in new} == pending else None
        if len(new) == 1 and new[0].role == "user":
            return new
        return None

    def send_user(self, content):
        self._send_event({"type": "user", "message": {"role": "user", "content": content}})

    def _send(self, request_id, rpc_id, result):
        self._send_mcp(request_id, {"jsonrpc": "2.0", "id": rpc_id, "result": _tool_result_payload(result)})
        return _digest(False, result.text)

    def _result(self, usage, output_tokens, calls, native_calls, finish_reason):
        cache_read = usage.get("cache_read_input_tokens", 0)
        cache_write = usage.get("cache_creation_input_tokens", 0)
        input_tokens = usage.get("input_tokens", 0) + cache_read + cache_write
        result = ex6.LLMResult(input_tokens, output_tokens, calls, finish_reason, cost=0, native_calls=native_calls,
                               cached_tokens=cache_read, cache_write_tokens=cache_write, raw_usage=usage)
        _log_invoke(self.ctx, [], result, cache_read, cache_write)
        return result

    def pump(self, results):
        """Yields one API call's worth of output: stops at tool calls or end of turn.
        results: {tool_use_id: ToolResult} for the calls claude is waiting on."""
        sent = {}  # tool_use_id -> digest of the result delivered to claude
        echoed = set()  # tool_use_ids claude confirmed with a matching tool_result
        calls = []
        native_calls = []
        usage = {}
        output_tokens = 0
        stash, self.waiting = self.waiting, {}
        for tool_use_id, ids in stash.items():  # calls that arrived before ex6 had results
            self.events.put(("call", tool_use_id, *ids))
        last_event = time.monotonic()
        while True:
            if self.ctx.stop_early:
                return
            idle = time.monotonic() - last_event
            if idle > _INACTIVITY_TIMEOUT or (idle > _CALL_TIMEOUT and len(sent) < len(results)):
                self.ctx.app.debug_print(f"[claude] stalled: pending={sorted(results)} "
                                         f"sent={sorted(sent)} waiting={sorted(self.waiting)}")
                raise TimeoutError(f"Claude stalled for {idle:.0f}s")
            try:
                item = self.events.get(timeout=0.5)
            except queue.Empty:
                continue
            last_event = time.monotonic()
            if item[0] == "exit":
                raise RuntimeError(f"Claude exited: {item[1]}")
            if item[0] == "call":
                _, tool_use_id, request_id, rpc_id = item
                if tool_use_id in results and tool_use_id not in sent:
                    sent[tool_use_id] = self._send(request_id, rpc_id, results[tool_use_id])
                else:
                    self.waiting[tool_use_id] = (request_id, rpc_id)
                continue

            event = item[1]
            if event.get("type") == "stream_event":
                inner = event["event"]
                if inner["type"] == "message_start":
                    usage = inner["message"].get("usage") or {}
                elif inner["type"] == "message_delta":
                    output_tokens = (inner.get("usage") or {}).get("output_tokens", 0)
                elif inner["type"] == "content_block_delta":
                    delta = inner["delta"]
                    if delta["type"] == "text_delta":
                        yield ex6.ResponseChunk("text", delta["text"])
                    elif delta["type"] == "thinking_delta":
                        yield ex6.ResponseChunk("cot", delta["thinking"])
                elif inner["type"] == "message_stop" and calls:
                    # claude emits one assistant event per block; message_stop follows all of them.
                    self._check_echoed(results, echoed)
                    yield self._result(usage, output_tokens, calls, native_calls, "tool_calls")
                    return

            elif event.get("type") == "assistant":
                for block in event["message"]["content"]:
                    if block["type"] == "tool_use":
                        if block["name"] in self.native_tools:
                            record = {"id": block["id"], "name": block["name"], "args": block["input"], "result": None}
                            self.native_calls[block["id"]] = record
                            native_calls.append(record)
                            self.ctx.app.debug_print(f"[claude] native tool_use: {json.dumps(block)}")
                            yield ex6.ResponseChunk("cot", f"\n[{block['name']}] {json.dumps(block['input'])}\n")
                        else:
                            if not isinstance(block["input"], dict):
                                raise RuntimeError(f"malformed tool args for {block['name']}: {block['input']!r}"[:700])
                            calls.append({"id": block["id"], "name": block["name"], "args": block["input"]})

            elif event.get("type") == "user":
                # claude echoes every tool result it accepted; it must be exactly what ex6 sent.
                for block in event["message"]["content"]:
                    if block["type"] != "tool_result":
                        continue
                    tool_use_id = block["tool_use_id"]
                    if tool_use_id in self.native_calls:
                        content = block.get("content")
                        text = content if isinstance(content, str) else json.dumps(content)
                        self.native_calls.pop(tool_use_id)["result"] = ("ERROR: " if block.get("is_error") else "") + text
                        self.ctx.app.debug_print(f"[claude] native tool_result {tool_use_id}: {text[:500]}")
                        continue
                    if tool_use_id not in sent:
                        raise RuntimeError(f"Claude resolved tool call {tool_use_id} itself: "
                                           f"{str(block.get('content'))[:200]}")
                    if _digest(block.get("is_error"), block["content"]) != sent[tool_use_id]:
                        raise RuntimeError(f"Claude altered the result of tool call {tool_use_id}")
                    echoed.add(tool_use_id)

            elif event.get("type") == "result":
                if event.get("is_error"):
                    raise RuntimeError(event.get("result") or "Claude invocation failed")
                self._check_echoed(results, echoed)
                yield self._result(usage, output_tokens, [], native_calls, event.get("stop_reason") or "stop")
                return

    def _check_echoed(self, results, echoed):
        if echoed != set(results):
            raise RuntimeError(f"Claude never confirmed tool results: {sorted(set(results) - echoed)}")

    def close(self):
        """Idempotent and exception-proof; safe from any thread."""
        if self.closed:
            return
        self.closed = True
        for step in (self._kill, self.tmp.cleanup):
            try:
                step()
            except Exception as e:
                self.ctx.app.debug_print(f"[claude] close: {step.__name__} failed: {e!r}")

    def _kill(self):
        # Kill first: Windows can't delete temp files a live child still holds.
        if self.proc.poll() is None:
            if os.name == "nt":  # terminate() would orphan node behind a .cmd shim
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)], capture_output=True)
            else:
                self.proc.kill()
        self.proc.wait(timeout=5)


@ex6.handler
def invoke_llm(ctx: ex6.Context):
    if ctx.model.startswith("anthropic/") and ":" not in ctx.model:
        return invoke_anthropic


def invoke_anthropic(ctx: ex6.Context):
    messages = ctx.get_messages()
    process = _processes.get(ctx)
    completed = False
    try:
        new = process.new_messages(messages) if process else None
        results = {}
        if new is None:
            if process:
                ctx.app.debug_print("[claude] history diverged; restarting process")
                process.close()
            process = _processes[ctx] = _ClaudeProcess(ctx)
            process.send_user(_bootstrap_prompt(ctx))
        elif new[-1].role == "tool":
            for message in new:
                value = message.get_msg(ctx)
                if not isinstance(value, ex6.ToolResult):
                    value = ex6.ToolResult(str(value))
                results[message.tool_call_id] = value
        else:
            process.send_user(_user_content(new[-1].get_msg(ctx), new[-1].attachments))
        process.seen = messages

        ctx.app.debug_print(f"[claude] model={ctx.model}")
        yield from process.pump(results)
        completed = not ctx.stop_early
    except Exception as e:
        ctx.app.debug_print(f"[claude] exception: {e}")
        result = ex6.LLMResult(error=str(e))
        _log_invoke(ctx, [], result)
        yield result
    finally:
        # Interrupted or failed mid-turn: claude's state is unknown, so start fresh next time.
        if not completed and process:
            process.close()
            _processes.pop(ctx, None)
