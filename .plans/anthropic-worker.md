# Anthropic provider: single-pump, no stalls

## Motivation

`provider_anthropic.py` can stall forever after a tool call. Root cause: two independent unbounded waits fed by split state:

- `_mcp_conn()` waits on `mcp_queue` for one specific tool-use ID, with no deadline, exiting only on process death. If Claude resolves a tool_use itself (schema validation failure, unknown tool) it never calls MCP for that ID, so ex6 waits forever while Claude keeps going.
- `read_turn()` does `events.get()` with no timeout; any API hang or unexpected event ordering (e.g. `assistant` events after `message_stop`) is a permanent hang.
- `MCP_TOOL_TIMEOUT` is 24h, so Claude's side never gives up either.

Goal: one queue, one consumer, one source of truth (ex6 history), and every wait bounded. Do NOT add a per-Context worker thread: the Context invocation thread already is the worker (Context admission allows one request at a time), and `_read_llm_stream` already wraps the generator in `closing()`, so the existing `finally: close()` handles cancellation. A worker would add a thread handoff (output queues, cancel messages, generations) — more state, not less.

Target: near-zero net line count in the provider; fault-tolerance additions ~40–60 lines. Robust beyond the green path: every fault (bad external input, ordering, crashes, user actions mid-flight, our own bugs) is detected within bounded time and recovers through one path. See Fault tolerance.

## Relevant files

- `_ex6/provider_anthropic.py`: `_ClaudeProcess`, `invoke_llm`, `_mcp_server`.
- `ex6.py`: `CONTEXT_HOOK_KINDS` (~L77), App hook registration (~L544), `TUI.remove_context` (~L2245), restart path after `_run_tui_loop` (~L590), `Context._read_llm_stream` (~L1184, `stop_early` check).

## Design

### One queue

All helper threads only enqueue into `self.events`; nothing else is shared:

- stdout reader: `("claude", event)`, then `("exit", code)`. Body wrapped in `try/finally` that always enqueues `("exit", ...)`, so a reader crash (non-dict JSON line, `_capture_usage` raising) surfaces immediately instead of after the inactivity timeout. Non-dict JSON lines are skipped like invalid JSON.
- MCP accept thread: `("mcp", tool_use_id, conn)`. Remove `mcp_queue`. Per-connection `try`: a bad client (timeout, bad JSON, wrong token, missing id) closes that conn only, never kills the accept loop. Set a short socket timeout on the accepted conn before reading its request line, so one bad client can't block accepting.

### State = `seen` + live sockets

- `seen`: identity snapshot of ex6 messages Claude has been given. Only stored protocol state.
- Remove `open_calls`. Pending IDs are derived from history: in `new_messages`, when `new[0]` is the assistant reply, expected IDs = `{c["id"] for c in new[0].tool_calls}`. Compare as sets with the submitted tool messages' IDs (order-insensitive). Mismatch → None → restart.
- `self.conns`: `{id: conn}` stash for MCP sockets that arrived before ex6's result. Transient I/O, not protocol truth. Closed on turn end/close.

### One pump

Replace `read_turn()` + `complete_tools()` + `_mcp_conn()` with a single generator `pump(results)` where `results` is `{id: ToolResult}` (empty for user turns).

- On start: send any stashed conns that have results.
- `("mcp", id, conn)`: if `id in results`, send + close; else stash.
- `("claude", event)`: handle as today (chunks, usage, tool calls, result).
- `("claude", user event with tool_result)`: end-to-end echo check (see Fault tolerance).
- `("exit", code)`: raise.
- Turn ends exactly as today (tool calls at message stop, or `result`).

Optional ordering hardening: end tool-call turns on `message_delta` `stop_reason == "tool_use"` and collect calls from `content_block_start`/`input_json_delta` instead of `message_stop and calls`. Only if cheap; the deadlines below already turn ordering bugs into errors.

### Bounded waits

Pump loop uses `events.get(timeout=0.5)` and checks each tick:

- `ctx.stop_early` → return (fixes stop latency; today stop is only seen after the next yielded item).
- Inactivity deadline (~300s, reset on any event) → raise.
- Conn deadline (~30s) while some `results` ID has no conn and no result sent yet → raise.

No timeout applies while ex6 runs tools / waits on user approval, because `invoke_llm` is not executing then.

Lower `MCP_TOOL_TIMEOUT` from 24h to a bounded backstop above realistic user-approval waits (e.g. 2h).

All errors flow through the existing `except` → `LLMResult(error=...)` and `finally` → `close()` + `_processes.pop(ctx)`. Next request starts clean.

On timeout, log one line: stage, pending IDs, IDs with conns received. Don't log prompts, tool output, or deltas.

### Close

Idempotent, safe from another thread, exception-proof: each step in its own `try` so one failure never skips the rest. Order matters:

1. set `closed`.
2. kill the process tree (Windows: `taskkill /T /F /PID`; `terminate()` can orphan the MCP child and node if `claude` resolves to a `.cmd` shim). Reap with a short wait.
3. close server socket, all stashed conns.
4. cleanup temp dir with `ignore_errors` (Windows can't delete files a live child still holds; hence kill first).

## Fault tolerance

Model: ex6 Context history is the single source of truth. The Claude process + `seen` + `conns` are a disposable cache of it. Every fault must be (a) detected within bounded time and (b) handled by the one recovery path: raise → `LLMResult(error)` → `close()` → next request bootstraps from history. Worst case of any fault = one restart + one bootstrapped turn (lost prompt cache, flattened transcript). Never a hang, never silent divergence.

Fail-safe direction for every piece of state, i.e. corruption must cause a restart, never a false "ok":

- `seen` wrong → identity comparison mismatches → restart.
- `_processes` entry stale/closed → `new_messages` checks `closed` as well as `proc.poll()` → restart.
- Stale MCP server from an old process → per-process port + token, can't reach the new listener. (Generations for free.)
- `conns` mapping wrong / result sent to wrong call / Claude mangled or replaced a result → caught by the echo check.

End-to-end echo check. Claude echoes a `user` event with a `tool_result` for every tool result it accepts. In the pump, track `sent = {id: (is_error, text_digest)}` for results ex6 delivered. For each echoed tool_result:

- ID not in `sent` → Claude resolved the call itself (schema failure, unknown tool, internal timeout) → raise.
- `is_error` or text digest differs from what ex6 sent → raise.
- Verify the echo format against real `claude -p` first; normalize to the concatenated text blocks. If content proves unstable, fall back to ID + `is_error` only.
- Set `MAX_MCP_OUTPUT_TOKENS` high in the Claude env so Claude doesn't truncate ex6 tool output; ex6 tools own output size. Truncation would otherwise be a (now detected) divergence on every large result.

Boundary invariants, raise instead of skip:

- Pump start: `results` IDs == expected pending IDs.
- Turn end (`message_stop` with calls, or `result`) after a tool continuation: every result was sent AND echoed.
- Known event types with unexpected shape (missing keys, wrong types) raise. Unknown event types are ignored; Claude CLI adds new ones.

No retry loops in the provider. Errors surface to the user; a persistent fault can't become a restart storm.

Known limits: corruption of ex6 history itself is undetectable by definition. A pump-local variable flipped mid-loop can defeat its own deadlines; backstops are `stop_early` (polled every 0.5s), `MCP_TOOL_TIMEOUT`, and process death.

## Lifecycle (ex6.py)

- Add `context_removed` to `CONTEXT_HOOK_KINDS` + App registration. `TUI.remove_context(ctx)` fires it once when the ctx was actually removed. Provider hook: pop + close that ctx's process.
- Restart leaks today: POSIX `os.execv` skips `atexit`; Windows `subprocess.call` keeps the old process (and its Claude children) alive for the whole new session. Before restart, fire `context_removed` (or equivalent close pass) for all remaining contexts. Keep provider `atexit` for normal exit.
- `Context.clear()`: no hook; divergence in `new_messages` restarts lazily (existing behavior).
- Forks: distinct Context identity → no shared process (existing behavior).

## Verification

Small fake-event tests (feed `events` directly, fake conns via socketpair), no network:

- conn-before-result and result-before-conn both send exactly once.
- parallel calls, results in any order.
- Claude self-resolved tool_result → error + restart, not hang.
- missing conn → error after conn deadline.
- inactivity → error after deadline.
- `stop_early` set while blocked → returns within ~0.5s, process closed.
- process exit mid-turn → error, next request bootstraps.
- `context_removed` closes only that ctx's process; double close harmless.
- echo with mismatched digest / `is_error` / unsent ID → error + restart.
- turn ends with a result sent but never echoed → error.
- stdout reader crash (non-dict JSON line, `_capture_usage` raising) → immediate error, not inactivity timeout.
- bad MCP client (wrong token, garbage, silent) → that conn dropped, accept loop keeps working.
- `close()` with a failing step (e.g. temp dir locked) still runs the remaining steps.

Then a temporary `smoke_test.py` against real `claude -p`: sequential tools, parallel tools, repeated turns, history clear/restart, and capture the real `user` tool_result echo format to confirm the digest normalization. Delete afterwards. Run project tests, `py_compile`, `git diff --check`, review full diff.
