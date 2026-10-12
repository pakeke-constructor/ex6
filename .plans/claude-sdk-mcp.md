# Anthropic provider: SDK MCP server, unprefixed tool names

## Motivation

Error seen intermittently:

`Claude resolved tool call toolu_... itself: <tool_use_error>Error: No such tool available: powershell</tool_use_error>`

Happens for `powershell`, `read_file`, etc. The tools are in context, but claude exposes them as `mcp__ex6__powershell`. Everything ex6-side (system prompts, agents, the bootstrap transcript in `_bootstrap_prompt`) calls them `powershell`. The model sometimes uses the bare name; claude rejects it; the echo check raises.

Fix: tools should be named exactly what ex6 calls them. Claude Code supports this only for SDK-type MCP servers with `CLAUDE_AGENT_SDK_MCP_NO_PREFIX=1` (verified on claude 2.1.295; the env var is ignored for stdio servers). SDK MCP traffic goes over claude's own stdin/stdout as control messages, so the stdio proxy subprocess, socket server, and token all go away. Net: fewer moving parts, fewer lines. Keeps the anthropic-worker.md guarantees: never hang, never silently diverge, one recovery path (raise → close → bootstrap from history).

## Relevant files

- `_ex6/provider_anthropic.py`: `_ClaudeProcess` (`__init__`, `_read_stdout`, `_accept_mcp`, `pump`, `_send`, `close`, `_close_conns`), `_digest`, `_mcp_server`, `__main__` block, `_MCP_PREFIX`.
- `.plans/anthropic-worker.md`: prior design; still applies except the MCP transport.

## Protocol (verified with a smoke test)

- mcp config: `{"mcpServers": {"ex6": {"type": "sdk", "name": "ex6"}}}`. Keep `--strict-mcp-config`, `--tools=`.
- env: `CLAUDE_AGENT_SDK_MCP_NO_PREFIX=1`.
- On start, send on stdin: `{"type": "control_request", "request_id": "init", "request": {"subtype": "initialize", "hooks": null}}`. Claude replies with a `control_response` on stdout (ignore it).
- Claude sends on stdout: `{"type": "control_request", "request_id": R, "request": {"subtype": "mcp_message", "server_name": "ex6", "message": <jsonrpc>}}`. Methods seen: `initialize`, `notifications/initialized`, `tools/list`, `tools/call`.
- Reply on stdin: `{"type": "control_response", "response": {"subtype": "success", "request_id": R, "response": {"mcp_response": {"jsonrpc": "2.0", "id": <rpc id>, "result": ...}}}}`.
- `tools/call` params carry `_meta["claudecode/toolUseId"]` as before, and `name` is unprefixed.
- `system/init` event reports `tools: ['read_file']`, and the model's `tool_use.name` is `read_file`.
- EVERY `mcp_message` control_request needs a control_response, notifications included. Reply to notifications with `"mcp_response": {"jsonrpc": "2.0", "result": {}}` (no id), as the python claude-agent-sdk does. If you skip it, claude stalls 30s, sends `initialize` again, and never asks for `tools/list`, so the model sees no tools.

## Resolved: long-outstanding calls (verified 2026-10-12, claude 2.1.295, haiku)

- No SDK control-request cap. A `tools/call` held 330s was accepted, both with `MCP_TOOL_TIMEOUT` unset and with it set to 2h.
- `MCP_TOOL_TIMEOUT` DOES apply to SDK servers. Set to 10000, the call failed at 10s: claude echoed `is_error` tool_result `MCP server "ex6" tool "wait_tool" timed out after 10s`, then sent `notifications/cancelled {"requestId": <rpc id>, "reason": "SdkError: Request timed out"}`, and the model carried on. So keep `env.setdefault("MCP_TOOL_TIMEOUT", 2h)`.
- While a call is outstanding, claude emits `{"type": "tool_progress", "tool_use_id": "<id>-heartbeat-N", "elapsed_time_seconds": ...}` on stdout every 30s. These reach `self.events` as claude events and reset pump idle time. Harmless, but don't treat them as unexpected.

## Design

- Stdin writes now come from two threads → `self.write_lock` around `_send_event` (the renamed stdin writer, shared by `send_user` and control responses).
- `_read_stdout` answers stateless MCP requests itself (`initialize`, `tools/list`, notifications get the id-less empty result above, any other request with an id gets `{}`), same logic as the current `_mcp_server`. They can arrive at startup or between turns, when no pump runs, so the pump can't own them.
- `tools/call` is the only stateful one: enqueue `("call", tool_use_id, request_id)` into `self.events`. It replaces `("mcp", id, conn)`.
- `pump` handles `"call"` the same as the old `"mcp"`: if the result is known, reply via a control_response and record its digest in `sent`; otherwise stash in `self.waiting = {tool_use_id: request_id}` (replaces `self.conns`). Same stash re-queue at pump start. Same deadlines: `_CONN_TIMEOUT` now means "call request didn't arrive".
- `_send(request_id, rpc_id, result)`: build the jsonrpc result with `_tool_result_payload`, wrap it, write it, return the digest. The `"call"` event therefore carries rpc_id too: `("call", tool_use_id, request_id, rpc_id)`.
- Delete: `socket`/`secrets` imports, `self.server`, `self.token`, `_accept_mcp`, `_close_conns`, `server.close` in `close()`, `tools.json`, `_mcp_server`, the `__main__` block, `_MCP_PREFIX` and its `removeprefix`.
- Keep `_kill` with `taskkill /T` (node behind the `.cmd` shim). The comment about the MCP child goes.
- `_digest` placeholder regex: drop the `mcp__ex6__` part (verify the actual placeholder text, probably `(powershell completed with no output)`).
- Keep `_meta.anthropic/maxResultSizeChars`, `MAX_MCP_OUTPUT_TOKENS`, and the echo check unchanged.
- `MCP_TOOL_TIMEOUT`: keep (verified it applies to SDK servers).

## Verification

A temporary `smoke_test.py` against real `claude -p` (haiku is fine), deleted afterwards:

1. ~~Long-outstanding call~~: done, see "Resolved" above.
2. Re-run the echo-format cases (empty, whitespace, `\r\n`, unicode, 60k chars, image) through the SDK route; confirm the placeholder text for the `_digest` regex.
3. Parallel calls, results in any order; call arrives before / after ex6 has results.
4. Tool named like a built-in (e.g. `Read`) with `--tools=`: no collision. Low priority; ex6 tools are snake_case.
5. Full ex6 session: sequential and parallel tools, multiple turns, `/clr` → restart and bootstrap, stop mid-turn, context removal. Confirm no orphan `claude`/`node` processes.

Then `py_compile` and `git diff --check`, and review the full diff.
