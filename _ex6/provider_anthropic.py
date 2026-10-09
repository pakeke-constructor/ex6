"""
Claude subscription backend: drives a long-lived `claude -p` process.
ex6 tools are exposed through an MCP proxy; ex6 itself runs them and the proxy
hands results back to claude by tool_use id.

Context overhead (can't be removed without --bare, which needs an API key):
- system: "You are a Claude agent, built on Anthropic's Claude Agent SDK."
- first user msg: <system-reminder>s with environment (cwd, platform, OS), model name, date.
- small "<total_tokens> left" reminders appended to some turns.
See _notes/claude-p-findings.md.
"""
import atexit
import base64
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ex6
from _ex6.provider import _log_invoke


_MCP_PREFIX = "mcp__ex6__"


def _capture_usage(event, app):
    import time
    info = event.get("rate_limit_info") or {}
    windows = info.get("unifiedWindows") or {}
    usage = app.plugin_data.setdefault("anthropic:usage", {})
    for key in ("five_hour", "seven_day"):
        w = windows.get(key)
        if w:
            usage[key] = {"utilization": w.get("utilization", 0), "resets_at": w.get("resetsAt", 0)}
    if "utilization" in info:
        usage["utilization"] = info["utilization"]
        usage["resets_at"] = info.get("resetsAt", 0)
        usage["rate_limit_type"] = info.get("rateLimitType", "")
    usage["ts"] = time.time()
    app.debug_print(f"[claude] rate_limit: {info.get('rateLimitType')} util={info.get('utilization')}")


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
        self.open_calls = []  # tool_use ids claude is waiting on
        self.events = queue.Queue()  # claude stdout events
        self.mcp_queue = queue.Queue()  # (tool_use_id, conn) from the MCP proxy
        self.mcp_conns = {}
        self.token = secrets.token_hex(16)
        self.closed = False

        self.server = socket.socket()
        self.server.bind(("127.0.0.1", 0))
        self.server.listen()
        port = self.server.getsockname()[1]
        threading.Thread(target=self._accept_mcp, daemon=True).start()

        self.tmp = tempfile.TemporaryDirectory(prefix="ex6-claude-")
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
        env.setdefault("MCP_TOOL_TIMEOUT", str(24 * 3600 * 1000))  # ex6 tools may wait on the user
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
        for line in self.proc.stdout:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                self.ctx.app.debug_print(f"[claude] invalid output: {line.rstrip()}")
                continue
            if event.get("type") == "rate_limit_event":
                _capture_usage(event, self.ctx.app)
            self.events.put(("claude", event))
        self.events.put(("exit", self.proc.wait()))

    def _read_stderr(self):
        for line in self.proc.stderr:
            self.ctx.app.debug_print(f"[claude] {line.rstrip()}")

    def _accept_mcp(self):
        while not self.closed:
            try:
                conn, _ = self.server.accept()
                request = json.loads(conn.makefile("r", encoding="utf-8").readline())
                if request.get("token") != self.token:
                    conn.close()
                    continue
                self.mcp_queue.put((request["id"], conn))
            except OSError:
                return

    def new_messages(self, messages):
        """Messages ex6 added since claude's last turn, or None if the histories
        diverged (clear, truncate, edits, interrupted tool loop, changed prompt/tools)."""
        if self.proc.poll() is not None:
            return None
        if self.system != _system_prompt(self.ctx) or self.tools != _mcp_tools(self.ctx):
            return None
        seen = self.seen
        if len(messages) <= len(seen) or any(a is not b for a, b in zip(seen, messages)):
            return None
        new = list(messages[len(seen):])
        if new[0].role == "assistant":  # claude's own reply, appended by ex6
            new = new[1:]
        if self.open_calls:
            ids = [m.tool_call_id for m in new]
            return new if ids == self.open_calls else None
        if len(new) == 1 and new[0].role == "user":
            return new
        return None

    def send_user(self, text):
        event = {"type": "user", "message": {"role": "user", "content": text}}
        self.proc.stdin.write(json.dumps(event) + "\n")
        self.proc.stdin.flush()

    def _mcp_conn(self, tool_use_id):
        while tool_use_id not in self.mcp_conns:
            try:
                call_id, conn = self.mcp_queue.get(timeout=1)
                self.mcp_conns[call_id] = conn
            except queue.Empty:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"Claude exited with code {self.proc.returncode}")
        return self.mcp_conns.pop(tool_use_id)

    def complete_tools(self, tool_messages):
        for message in tool_messages:
            value = message.get_msg(self.ctx)
            if not isinstance(value, ex6.ToolResult):
                value = ex6.ToolResult(str(value))
            conn = self._mcp_conn(message.tool_call_id)
            conn.sendall((json.dumps(_tool_result_payload(value)) + "\n").encode())
            conn.close()
        self.open_calls = []

    def _result(self, usage, output_tokens, calls, finish_reason):
        cache_read = usage.get("cache_read_input_tokens", 0)
        cache_write = usage.get("cache_creation_input_tokens", 0)
        input_tokens = usage.get("input_tokens", 0) + cache_read + cache_write
        result = ex6.LLMResult(input_tokens, output_tokens, calls, finish_reason, cost=0)
        _log_invoke(self.ctx, [], result, cache_read, cache_write)
        return result

    def read_turn(self):
        """Yields one API call's worth of output: stops at tool calls or end of turn."""
        calls = []
        usage = {}
        output_tokens = 0
        while True:
            kind, event = self.events.get()
            if kind == "exit":
                raise RuntimeError(f"Claude exited with code {event}")

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
                    self.open_calls = [c["id"] for c in calls]
                    yield self._result(usage, output_tokens, calls, "tool_calls")
                    return

            elif event.get("type") == "assistant":
                for block in event["message"]["content"]:
                    if block["type"] == "tool_use":
                        name = block["name"].removeprefix(_MCP_PREFIX)
                        calls.append({"id": block["id"], "name": name, "args": block["input"]})

            elif event.get("type") == "result":
                if event.get("is_error"):
                    yield ex6.LLMResult(error=event.get("result") or "Claude invocation failed")
                    return
                yield self._result(usage, output_tokens, [], event.get("stop_reason") or "stop")
                return

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.server.close()
        if self.proc.poll() is None:
            self.proc.terminate()
        self.tmp.cleanup()


def invoke_llm(ctx: ex6.Context):
    messages = ctx.get_messages()
    process = _processes.get(ctx)
    completed = False
    try:
        new = process.new_messages(messages) if process else None
        if process is None or new is None:
            if process:
                ctx.app.debug_print("[claude] history diverged; restarting process")
                process.close()
            process = _processes[ctx] = _ClaudeProcess(ctx)
            process.send_user(_bootstrap_prompt(ctx))
        elif new[-1].role == "tool":
            process.complete_tools(new)
        else:
            process.send_user(new[-1].get_msg(ctx))
        process.seen = messages

        ctx.app.debug_print(f"[claude] model={ctx.model}")
        yield from process.read_turn()
        completed = True
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
