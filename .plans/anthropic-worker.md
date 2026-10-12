# Anthropic worker per Context

## Motivation

`provider_anthropic.py` can stall indefinitely after a tool call because ownership is split across the Context invocation thread, Claude stdout/MCP threads, and several inferred state containers (`seen`, `open_calls`, `mcp_queue`, `mcp_conns`). A missing or reordered event can leave one side waiting forever.

Make each Anthropic Context own one lazily-created worker that serializes Claude protocol state. Multiple contexts remain independent; deleting a context or exiting ex6 reliably closes its worker. Every wait is bounded, so Claude CLI/MCP failures become explicit errors and a clean restart rather than permanent stalls.

Keep this provider-local and simple. Reader/accept threads only convert blocking I/O into immutable worker events. They never mutate protocol state. Do not poll from TUI frame/render code.

## Current lifecycle gap

There is no context-removal callback. `TUI.remove_context()` only removes the object from its list. Add a small generic lifecycle hook rather than coupling core to Anthropic:

- Add `context_removed` to Context hook kinds/declarations and `App` registration.
- `TUI.remove_context(ctx)` invokes it once when removal succeeds.
- Provider registers a `context_removed` hook that pops and closes that Context's worker.
- Keep provider `atexit` cleanup for process exit.
- Ensure normal TUI shutdown/restart closes remaining contexts/workers. Prefer one TUI/App shutdown pass through the same hook; make close idempotent.
- `Context.clear()` does not delete Context. Worker may be invalidated immediately or lazily restarted when history divergence is next observed; it must not leak a second process.
- Forks start with no worker because registry is keyed by distinct Context identity.

Relevant files:

- `ex6.py`: context lifecycle declaration/registration, `TUI.remove_context`, TUI shutdown.
- `_ex6/provider_anthropic.py`: per-Context worker and MCP bridge.
- `_ex6/commands.py`: context deletion already routes through `TUI.remove_context`; likely no change.
- `_ex6/tools.py`: existing shell timeout remains separate; retain useful diagnostics only.

## Worker shape

Keep a module registry `WeakKeyDictionary[Context, ClaudeWorker]` (or ordinary dict if lifecycle cleanup is simpler and explicit). `invoke_llm` calls `worker_for(ctx)`, creating it only on first Anthropic invocation. No worker/process at context construction or selection time.

One `ClaudeWorker` per Context owns:

- worker thread and inbox queue
- `claude -p` process and temp MCP config
- listener socket
- current process generation
- exact history/system/tool snapshot accepted by Claude
- current turn state and pending tool transactions
- shutdown/restart/deadline handling

Public synchronous-generator adapter remains compatible with `Context._read_llm_stream`. It submits a request carrying a per-request output queue, then yields response chunks/results from that queue. Only one request per Context is allowed by existing Context admission.

Reader helpers are allowed only where blocking OS I/O requires them:

- stdout reader parses NDJSON and enqueues `ClaudeEvent(generation, event)`.
- stderr reader logs lines.
- MCP accept reader validates token, reads tool-use ID, and enqueues `McpConnected(generation, id, socket)`.
- process exit enqueues `ProcessExited(generation, code)`.

Worker loop owns all mutation and runs `inbox.get(timeout=0.1)`, then reconciles transactions and deadlines. No TUI-frame polling and no shared dictionaries mutated by helper threads.

## Explicit protocol state

Use minimal phases:

- `IDLE`: ready for a user turn.
- `GENERATING`: waiting for Claude output/result.
- `WAITING_FOR_TOOL_RESULTS`: tool calls emitted to ex6; MCP sockets and ex6 results may arrive in either order.
- `CLOSED`.

Avoid parallel truth sources. Replace `open_calls`, `mcp_queue`, and `mcp_conns` with one transaction record keyed by tool-use ID:

```python
@dataclass
class ToolTransaction:
    call: dict | None = None
    conn: socket.socket | None = None
    result: ToolResult | None = None
    sent: bool = False
```

Both arrival orders are valid:

- Claude/MCP socket first, ex6 result later.
- ex6 result first, MCP socket later.

`reconcile()` sends exactly once when both `conn` and `result` exist. Sending is idempotent at state-machine level: mark terminal and remove only after successful `sendall`/close. When every expected ID is sent, clear transaction batch and transition to `GENERATING` for Claude's continuation.

Validate sets, not incidental ordering:

- Reject duplicate IDs, unknown IDs, missing results, or a new turn in non-idle state.
- Match returned ex6 tool messages by ID and preserve original tool-call order only for user-visible/request semantics.
- Any protocol violation fails current request and restarts process. Never wait in hope that impossible state repairs itself.

History compatibility remains explicit:

- At request submission, compare Context system/tools/history against worker's accepted snapshot.
- Append-only user turn or exact pending tool-result batch continues existing process.
- Clear, truncate, edit, interrupted loop, model/reasoning/cwd/system/tool change causes controlled process restart and bootstrap from Context history.
- Snapshot changes only after worker accepts an input; worker thread is sole writer.

## Deadlines and recovery

No unbounded queue/socket/process wait.

Use a small set of named monotonic deadlines, configured as provider constants:

- process/MCP startup deadline
- first Claude event / active generation deadline
- MCP socket arrival after model tool call
- Claude continuation/result inactivity deadline after all tool results are sent
- shutdown grace period

Tool execution itself remains governed by each ex6 tool/Operation. Do not apply a short provider deadline while ex6 is legitimately running a tool or waiting for user approval. Once ex6 submits completed results, MCP delivery and Claude continuation are bounded.

On timeout, process exit, malformed event, socket error, or request cancellation:

1. Fail current provider request with one `LLMResult(error=...)`.
2. Close all pending MCP sockets.
3. terminate, then kill process tree after short grace if needed; reap process.
4. close listener and temp directory.
5. increment generation / discard stale queued events.
6. leave worker alive but processless; next request starts clean from Context history.

Generation is attached to every helper event. Ignore and close resources from stale generations after restart. Cleanup must be idempotent.

Do not rely on Claude's `MCP_TOOL_TIMEOUT` as safety. Set it only high enough for legitimate ex6 tool/UI waits; local worker deadlines protect post-tool protocol stages. Keep/uphold Claude CLI's own API stream timeout where available, but external worker timeout is authoritative.

## Request flow

### User turn

1. `invoke_llm(ctx)` lazily gets worker and submits Context message snapshot.
2. Worker starts/restarts process if snapshot is incompatible.
3. Worker writes one stream-json user event and enters `GENERATING`.
4. Parsed text/thinking chunks go to request output queue.
5. On complete tool-call message, create expected transactions, emit `LLMResult(..., tool_calls, "tool_calls")`, and enter `WAITING_FOR_TOOL_RESULTS`.
6. On normal result, emit final `LLMResult`, mark request complete, enter `IDLE`.

### Tool continuation

1. Next existing Context loop invocation submits exact tool messages.
2. Worker stores each result by ID; MCP sockets may already be present.
3. `reconcile()` returns available results to MCP without blocking on a particular ID.
4. After all sends, worker enters `GENERATING` and routes Claude continuation events to this request.

The adapter's generator `finally` sends request cancellation if consumer exits early. Worker invalidates/restarts generation because Claude state after interrupted streaming is unknown.

## Lifecycle details

- Lazy creation: registry entry and worker thread are created on first `invoke_llm`, while Claude process can be created by worker on first submitted request.
- Context deletion: `context_removed` closes and removes worker. If Context is running, request fails/cancels and no late event can write to deleted Context.
- Clear/truncate: existing history compatibility check triggers one controlled restart; optionally expose a provider invalidation hook only if eager close is clearly simpler.
- App/TUI shutdown and `/restart`: close all contexts/workers before terminal teardown/re-exec. Do not depend only on `atexit`.
- Provider replacement: old Anthropic worker is closed when next invocation no longer uses it if a generic callback exists; otherwise lifecycle cleanup remains deletion/shutdown plus `atexit`. Do not add broad provider-management machinery solely for this edge case.
- Worker thread, process readers, sockets, and child MCP process must terminate or be daemon-only after owned resources are closed. Avoid joining current thread.

## Logging

Retain concise transition diagnostics with Context name and generation:

- worker/process create, restart reason, PID
- request kind and phase transition
- tool call/result/socket arrival by ID
- MCP result sent
- timeout stage
- process exit and cleanup

Do not log token deltas, full prompts, command contents, tool output, or secrets. Logging observes state; it must not become another state mechanism.

## Implementation order

1. Add/test generic `context_removed` and shutdown lifecycle hook.
2. Introduce worker request/event records and lazy registry without changing wire behavior.
3. Move process ownership and Claude event parsing under worker.
4. Replace blocking `_mcp_conn()` and split containers with ID-keyed transactions/reconciliation.
5. Add generation checks, deadlines, failure/restart cleanup.
6. Remove old `_ClaudeProcess`, `_processes`, `seen/open_calls/mcp_queue/mcp_conns` paths rather than retaining compatibility layers.
7. Run targeted smoke tests against real `claude -p`, then full tests/diff review.

If implementation approaches 300 new production lines, stop and simplify. Prefer replacing old coordination code rather than wrapping it.

## Verification

Use small deterministic tests with fake process/socket events; avoid real-network tests for state-machine correctness:

- No worker/process before first Anthropic invocation.
- Two Contexts create distinct workers/process generations and can progress independently.
- Context deletion closes only its worker; repeated deletion/close is harmless.
- App shutdown/restart closes all workers.
- Socket-before-result and result-before-socket both send once.
- Multiple parallel calls complete in arbitrary order without ordered blocking.
- Duplicate/unknown/missing IDs fail and restart rather than stall.
- Stale generation events/sockets cannot affect replacement process.
- History divergence causes exactly one restart and clean bootstrap.
- Process exit and each deadline produce error, clean resources, and permit next-request recovery.
- Consumer cancellation closes/restarts process and does not deliver late chunks.
- One stalled Context does not delay another.
- Existing sequential, timeout, child-process-tree, parallel, repeated-turn, and full production-prompt smoke cases still pass.

Run `python -m unittest discover -s tests` (or project-standard tests), `py_compile`, `git diff --check`, and inspect full working tree.