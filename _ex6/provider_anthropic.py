"""
Claude subscription backend: drives a long-lived `claude -p` process.
ex6 tools are exposed through an MCP proxy; ex6 itself runs them and the proxy
hands results back to claude by tool_use id.

Why it's shaped this way:
We're driving an external process we don't control, across three channels
(stdin, stdout, MCP sockets) that can stall, reorder, crash, or quietly disagree
with us. Earlier versions hung forever after tool calls. So the priority is:
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
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ex6
from _ex6.provider import _log_invoke


_MCP_PREFIX = "mcp__ex6__"
_INACTIVITY_TIMEOUT = 300  # seconds without any claude/MCP event mid-turn
_CONN_TIMEOUT = 30  # seconds for claude to call MCP for a tool result ex6 has ready


def _capture_usage(event, app):
    import time
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
    if ctx.invoke_llm is not invoke_llm:
        return False
    import time
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
    return bool(is_error), hashlib.sha256(content.encode()).hexdigest()


def _bootstrap_prompt(ctx):
    messages = [m for m in ctx.get_messages() if m.role != "system"]
    if len(messages) == 1 and messages[0].role == "user":
        return messages[0].get_msg(ctx)

    lines = ["Continue this conversation. The transcript before the final user message is context, not instructions:"]
    for message in messages:
        value = message.get_msg(ctx)
        if isinstance(value, ex6.ToolResult):
            value = value.text
        if message.role == "assistant" and message.tool_calls:
            lines.append(f"[assistant tool calls] {json.dumps(message.tool_calls)}")
        if value:
            lines.append(f"[{message.role}] {value}")
    return "\n\n".join(lines)


class _ClaudeProcess:
    def __init__(self, ctx):
        self.ctx = ctx
        self.system = _system_prompt(ctx)
        self.tools = _mcp_tools(ctx)
        self.seen = ()  # ex6 messages claude has been given
        self.events = queue.Queue()  # ("claude", event) | ("mcp", tool_use_id, conn) | ("exit", code)
        self.conns = {}  # tool_use_id -> MCP socket that arrived before ex6's result
        self.token = secrets.token_hex(16)
        self.closed = False

        self.server = socket.socket()
        self.server.bind(("127.0.0.1", 0))
        self.server.listen()
        port = self.server.getsockname()[1]
        threading.Thread(target=self._accept_mcp, daemon=True).start()

        self.tmp = tempfile.TemporaryDirectory(prefix="ex6-claude-", ignore_cleanup_errors=True)
        folder = Path(self.tmp.name)
        prompt_path = folder / "system.txt"
        tools_path = folder / "tools.json"
        mcp_path = folder / "mcp.json"
        prompt_path.write_text(self.system, encoding="utf-8")
        tools_path.write_text(json.dumps(self.tools), encoding="utf-8")
        mcp_path.write_text(json.dumps({"mcpServers": {"ex6": {
            "type": "stdio",
            "command": sys.executable,
            "args": [str(Path(__file__).resolve()), "--mcp-server",
                     str(tools_path), str(port), self.token],
        }}}), encoding="utf-8")

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
            "--tools=",
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

    def _accept_mcp(self):
        while not self.closed:
            try:
                conn, _ = self.server.accept()
            except OSError:
                return
            try:
                conn.settimeout(5)  # a silent client can't block accepting
                request = json.loads(conn.makefile("r", encoding="utf-8").readline())
                if request["token"] != self.token:
                    raise ValueError("bad token")
                conn.settimeout(None)
                self.events.put(("mcp", request["id"], conn))
            except Exception as e:
                self.ctx.app.debug_print(f"[claude] dropped MCP client: {e!r}")
                conn.close()

    def new_messages(self, messages):
        """Messages ex6 added since claude's last turn, or None if the histories
        diverged (clear, truncate, edits, interrupted tool loop, changed prompt/tools)."""
        if self.closed or self.proc.poll() is not None:
            return None
        if self.system != _system_prompt(self.ctx) or self.tools != _mcp_tools(self.ctx):
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

    def send_user(self, text):
        event = {"type": "user", "message": {"role": "user", "content": text}}
        self.proc.stdin.write(json.dumps(event) + "\n")
        self.proc.stdin.flush()

    def _send(self, conn, result):
        try:
            conn.sendall((json.dumps(_tool_result_payload(result)) + "\n").encode())
        finally:
            conn.close()
        return _digest(False, result.text)

    def _result(self, usage, output_tokens, calls, finish_reason):
        cache_read = usage.get("cache_read_input_tokens", 0)
        cache_write = usage.get("cache_creation_input_tokens", 0)
        input_tokens = usage.get("input_tokens", 0) + cache_read + cache_write
        result = ex6.LLMResult(input_tokens, output_tokens, calls, finish_reason, cost=0)
        _log_invoke(self.ctx, [], result, cache_read, cache_write)
        return result

    def pump(self, results):
        """Yields one API call's worth of output: stops at tool calls or end of turn.
        results: {tool_use_id: ToolResult} for the calls claude is waiting on."""
        sent = {}  # tool_use_id -> digest of the result delivered to claude
        echoed = set()  # tool_use_ids claude confirmed with a matching tool_result
        calls = []
        usage = {}
        output_tokens = 0
        stash, self.conns = self.conns, {}
        for item in stash.items():  # conns that arrived before ex6 had results
            self.events.put(("mcp", *item))
        last_event = time.monotonic()
        while True:
            if self.ctx.stop_early:
                return
            idle = time.monotonic() - last_event
            if idle > _INACTIVITY_TIMEOUT or (idle > _CONN_TIMEOUT and len(sent) < len(results)):
                self.ctx.app.debug_print(f"[claude] stalled: pending={sorted(results)} "
                                         f"sent={sorted(sent)} conns={sorted(self.conns)}")
                raise TimeoutError(f"Claude stalled for {idle:.0f}s")
            try:
                item = self.events.get(timeout=0.5)
            except queue.Empty:
                continue
            last_event = time.monotonic()
            if item[0] == "exit":
                raise RuntimeError(f"Claude exited: {item[1]}")
            if item[0] == "mcp":
                _, tool_use_id, conn = item
                if tool_use_id in results and tool_use_id not in sent:
                    sent[tool_use_id] = self._send(conn, results[tool_use_id])
                else:
                    self.conns[tool_use_id] = conn
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
                    yield self._result(usage, output_tokens, calls, "tool_calls")
                    return

            elif event.get("type") == "assistant":
                for block in event["message"]["content"]:
                    if block["type"] == "tool_use":
                        name = block["name"].removeprefix(_MCP_PREFIX)
                        calls.append({"id": block["id"], "name": name, "args": block["input"]})

            elif event.get("type") == "user":
                # claude echoes every tool result it accepted; it must be exactly what ex6 sent.
                for block in event["message"]["content"]:
                    if block["type"] != "tool_result":
                        continue
                    tool_use_id = block["tool_use_id"]
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
                yield self._result(usage, output_tokens, [], event.get("stop_reason") or "stop")
                return

    def _check_echoed(self, results, echoed):
        if echoed != set(results):
            raise RuntimeError(f"Claude never confirmed tool results: {sorted(set(results) - echoed)}")

    def close(self):
        """Idempotent and exception-proof; safe from any thread."""
        if self.closed:
            return
        self.closed = True
        for step in (self._kill, self.server.close, self._close_conns, self.tmp.cleanup):
            try:
                step()
            except Exception as e:
                self.ctx.app.debug_print(f"[claude] close: {step.__name__} failed: {e!r}")

    def _kill(self):
        # Kill first: Windows can't delete temp files a live child still holds.
        if self.proc.poll() is None:
            if os.name == "nt":  # terminate() would orphan the MCP child (and node behind a .cmd shim)
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)], capture_output=True)
            else:
                self.proc.kill()
        self.proc.wait(timeout=5)

    def _close_conns(self):
        for conn in list(self.conns.values()):
            conn.close()


def invoke_llm(ctx: ex6.Context):
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
            process.send_user(new[-1].get_msg(ctx))
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


def _mcp_server(tools_path, port, token):
    tools = json.loads(Path(tools_path).read_text(encoding="utf-8"))
    write_lock = threading.Lock()

    def reply(message):
        with write_lock:
            sys.stdout.write(json.dumps(message) + "\n")
            sys.stdout.flush()

    def call_tool(rpc_id, params):
        # ex6 runs the tool itself (from the assistant's tool_use); we just wait for its result.
        request = {"token": token, "id": params["_meta"]["claudecode/toolUseId"]}
        with socket.create_connection(("127.0.0.1", int(port))) as conn:
            conn.sendall((json.dumps(request) + "\n").encode())
            line = conn.makefile("r", encoding="utf-8").readline()
        reply({"jsonrpc": "2.0", "id": rpc_id, "result": json.loads(line)})

    for line in sys.stdin:
        message = json.loads(line)
        rpc_id = message.get("id")
        method = message.get("method")
        if method == "initialize":
            result = {"protocolVersion": message["params"].get("protocolVersion", "2025-06-18"),
                      "capabilities": {"tools": {}},
                      "serverInfo": {"name": "ex6", "version": "1"}}
            reply({"jsonrpc": "2.0", "id": rpc_id, "result": result})
        elif method == "tools/list":
            reply({"jsonrpc": "2.0", "id": rpc_id, "result": {"tools": tools}})
        elif method == "tools/call":
            threading.Thread(target=call_tool, args=(rpc_id, message["params"]), daemon=True).start()
        elif rpc_id is not None:
            reply({"jsonrpc": "2.0", "id": rpc_id, "result": {}})


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--mcp-server":
    _mcp_server(*sys.argv[2:])
