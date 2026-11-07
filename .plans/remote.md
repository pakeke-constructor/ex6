# remote: ex6 as client-server (server-authoritative sync)
Status: active

## Motivation
I want to keep my agents running even after I quit the TUI, long-form on a VPS, without the keypress lag of tmux/ssh.
So: `ex6 --server` owns the agents (contexts, LLM loops, tools, files). `ex6 --client` is the TUI: input box, scrolling, j/k, zoom and panels all stay local, so keys never wait on the network.
Local runs use client+server too (plain `ex6` connects to this project's server, or spawns one detached), so anything broken remotely is broken identically locally. One code path, no single-process mode.

Core principles (these keep the error surface low and make desync impossible by construction):
1. Single writer. Only the server mutates agent state. The client's mirror state is written only by the snapshot applier.
2. State, not events, flows server -> client. Full context snapshots, keyed by revision, idempotent, last write wins. Missed or duplicated messages can't corrupt anything; the next snapshot heals it. Only streaming text is a delta, and it's offset-checked.
3. Inputs, not mutations, flow client -> server. Commands and keys run on both sides with the same arguments. Each side keeps only its own effects: Context changes are no-ops on the client, UI calls are no-ops on the server.
4. Same code on both sides, verified at handshake. "Run on both" only works if both sides run identical plugin code.
5. Data crosses the wire, never code references. The only exception is the ask_human panel class, which is verified on the server when it's created.
6. Loud failure. Anything not JSON-able, any code mismatch, any unclassified field: raise right where the bug is, never later.

## Python API (the part that matters)
```python
app.side            # "server" | "client". Use sparingly: setup-time and machine effects only.

# Commands / keys: run on BOTH sides, same args.
@ex6.command
def clr(tui): tui.current.clear()           # client: no-op. server: clears. (No branching.)

@ex6.key("dd")
def key_delete(tui, n): ...                  # same: runs on both

# Split command: server half touches the machine, client half does UI with the server's result.
@ex6.command_server
def plans(tui, *args):                       # server only; return JSON-able
    return _plans()
@ex6.command_client
def plans(tui, result, *args):               # client only; result = server half's return (None if no server half)
    _panel(tui, result)
# Registering the same name as @command AND as a split half raises. Either half alone is fine:
# client-only = pure UI (/theme, /help, /dbg); server-only = no UI.

# UI-originated state changes (from panels on ui_panel_stack) go through commands:
tui.run_command("/plans run foo")            # runs locally and sends a command event

# UI API: real on the client, no-op on the server during event handling.
tui.echo(msg)  tui.open_input(text, request)  tui.push_panel(draw)  tui.yank(text, what)  ex6.text_panel(tui, lines)
# Echo from server background threads (outside event handling) is broadcast to clients instead
# (provider warnings, thread crashes).

# Context API: real on the server, no-op on the client.
ctx.invoke / fork / clear / truncate / edit_messages / append_message / schedule / submit_input
tui.add_context / tui.remove_context
ctx.yolo = ...; ctx.stop_early = True; ctx.data[k] = v   # direct writes to synced fields: ignored on the client
                                                          # (mirror __setattr__ / data __setitem__), real on the server

# Human requests: panel class + JSON args; core builds the panel on the client.
ctx.ask_human(_HumanPanel, "AGENT QUESTION", lines, options=[...], cancel=lambda: ctx.yolo)
# Client builds _HumanPanel(mirror_ctx, *args, **kwargs). cancel stays on the server, never crosses.
# Panel protocol unchanged (action/input/draw, reply.focus/blur/complete). complete(value): value must be JSON.

# Side-specific setup / machine effects:
def setup(app):
    if app.side == "server": setup_inline_watcher(app, ctx)

# Plugin data, two primitives. Prefer shared state; use messages only for one-shot requests.
app.shared["plugin:key"] = value             # server writes; synced to every client (read-only there: client writes ignored,
                                             # like ctx.data). JSON values (dicts/lists ok), strict encoder. Part of the app
                                             # snapshot, so it survives reconnects and needs no resend logic.
app.send("plugin:topic", data)               # one-way JSON message. client -> server, or server -> all clients.
@ex6.receive("plugin:topic")
def on_topic(app, data): ...                 # server: runs on the event thread, in order with other events.
                                             # client: runs on the TUI thread between frames (UI-safe).
# Request/reply = command_server + command_client (already returns a value), so no separate rpc API.
```

Messaging rules (robustness):
- Messages are lossy by design. While disconnected, sends raise (client) or skip that client (server). Anything that must survive a reconnect goes in `app.shared` or `ctx.data`.
- A topic with no receiver on the other side raises an error echo, never a silent drop. Receiver exceptions are caught and sent back as an error echo.
- `app.send` inside a run-on-both handler runs twice, in opposite directions. Send from a split half (`command_server`/`command_client`) or a side-guarded block instead. If a run-on-both handler calls send, log a warning.

Known caveat with no-ops: a client handler that writes, then reads, sees the old value (`ctx.yolo = not ctx.yolo; echo(ctx.yolo)`).
Fix those sites by echoing the computed local variable instead. No client-side prediction: agent state shows after one round trip; UI-only state is instant.

## Robustness mechanisms (all required)

### Field classification
Every `Context` dataclass field is in exactly one of SYNCED / CLIENT_LOCAL / SERVER_ONLY. Assert this at import time, so a new field without a classification crashes on boot.
- SYNCED: name, model, reasoning, parent, ephemeral, cwd, yolo, schema_id, max_tokens, max_tool_output, messages, data, tools (as schemas), llm_is_running, llm_suspended, pending_message, llm_cot, llm_output_type, llm_result, last_invoke_time_start/end, pending_tools (ToolCall rows), requests (pending ask_human), invoke_llm (as a "module.name" string, display only), stop_early.
- CLIENT_LOCAL: _scroll_up, _zoom, _input_box, last_seen, _prev_height, ui_stack (rebuilt from `requests`).
- SERVER_ONLY: _context_hooks, _msg_lock, _read_hashes, _line_snapshots, _scheduled, _pending_inputs, _clear_pending, data_volatile.
- The name is the identity and is immutable (enforce it). The client applies snapshots in place onto the stable mirror object for that name, so references held by client code stay valid.

### Snapshots (server -> client)
- `snapshot(ctx) -> dict` reads only SYNCED fields, under `ctx._msg_lock`, so it's consistent.
  - Callable message content goes through `get_msg(ctx)` (a string). ToolResult and attachments become plain dicts.
  - Tools become `{name, schema}` via `tool_to_schema`. The client gets `RemoteTool(name, schema)` stand-ins. `tool_to_schema` and `_validate_tool_sig` accept them.
  - The applier always overwrites tools, including on setup-created mirrors, so client tools are uniform.
  - Why not qualnames: `functools.wraps` copies `__qualname__`, so `_handoff` in agents_long_horizon would resolve to `tools.handoff`.
- A strict encoder accepts only JSON primitives plus the known types and raises on anything else (lambdas, objects in data).
- Revision: `ctx._rev` is bumped automatically by the `__setattr__` hook on SYNCED fields and by every message mutator (append/edit/truncate/clear).
  - Safety net: every ~2s the server re-serializes each context and compares it to the last snapshot sent. If it differs while `_rev` didn't change, it logs "MISSED REV BUMP <field>" loudly and sends the snapshot anyway. This catches in-place mutations like ToolCall status and self-heals.
- Streaming: `pending_message` text and `llm_cot` growth go as `(rev, field, offset, append)`. The client applies one only if the offset equals its current length; otherwise it ignores it and the next snapshot heals it.
- Pusher: one thread per client, ~20Hz. Per context, it compares `_rev` against what was last sent and sends a snapshot or a stream append. This is state-based polling, so there are no missed notifications.
- App snapshot (also by revision): roster (ordered context names = list order), budget cost/limit, favorite name, fatal_error.
  - Removal is a roster diff, never a "removed" event, so a missed event can't leave a zombie context.
- Self-check: each snapshot carries `hash`. The client re-serializes its mirror after applying and compares. On a mismatch (applier bug) it logs loudly and asks for a resync.
  - `/sync` shows client vs server hashes for every context.

### Events (client -> server)
Exactly six kinds (the sixth is send: `{topic, data}`):
- hello: `{token, ex6_hash, revs}`
- command: `{text, current, names}`
- key: `{seq, count, current, names}`. Resolved on the client and sent as a sequence, not raw keystrokes.
- submit: `{ctx, text, attachments}`. Input box Enter. Text starting with "/" is a command event instead.
- reply: `{ctx, req, value}`

Rules:
- Core-loop direct mutations become key events. Ctrl-x stop is currently at ex6.py ~3087 and ~3143; it goes through the stop binding.
- The server keeps one stand-in TUI, with no term. Per event it sets `current` from the event and runs the same function the client ran. An unknown `current` makes the event fail loudly.
- The server applies events in TCP order. Exceptions are caught per event, sent back as an error echo, and the traceback goes to the server log. An event never kills the server.
- Name pinning (determinism): while the client runs an event handler, it records the name of each context it adds (via `_unique_name`, fork, or Context()). It sends them as `names`.
  - On the server, `add_context` during that event takes the next pinned name. If that name already exists on the server, the event fails loudly. Both sides now agree on new agents' names.
  - On the client, `add_context` inserts a greyed stand-in that the snapshot with the same name fills. If no snapshot confirms it within ~5s, it's dropped with an error echo.
- While disconnected, events are refused with a "not connected" echo, not queued. No stale replays. Drafts are local, so typing isn't lost.
- Client handlers must never block on server state. For example, `agents_long_horizon.run()` polls `is_running` and would freeze the client. Long-running loops go in `command_server`.

### ask_human
- Server: `ctx.ask_human(cls, *args, cancel=None, **kwargs)`.
  - At creation it verifies the class is a module-level class (`importlib` resolve(qualname) is cls) and that the arguments are JSON-able. It then adds `{req, cls, args, kwargs}` to synced `ctx.requests` and blocks until a reply, a stop, removal, or `cancel()`. Req ids are monotonic per server process.
- Client: when a new req id appears in a snapshot, it builds the panel and pushes it onto the local `ui_stack`. It drops panels whose req is no longer in the snapshot.
  - Being state-based means it's reconnect-safe: a lost reply makes the panel reappear.
- `reply.complete(v)` hides the panel locally and sends the reply event. The server ignores replies for requests that are no longer pending.

### Code identity and client-local plugins
- The server sends its plugin sources (core `_ex6/`, server-user `_ex6/`, project `_ex6/`) on hello. The client loads those into a temp dir, so the "run on both" code is identical and the client needs no repo checkout.
- The client also loads its own user plugins (`<client get_folder()>/_ex6/`). Load order: shipped plugins first, then client-local. A client-local file with the same name as a shipped one is skipped with a warning, so the server's copy wins and both sides stay identical.
- Each registration (command, key, handler, receive) records its origin: shipped or client-local. Client-local plugins are client-only by construction:
  - Their commands and keys run on the client only and send no event (the server doesn't know them).
  - If a client-local plugin mutates Context state (which is a no-op on the client), it raises: "client-local plugin can't change agents; use app.send to a server plugin".
  - If a client-local plugin registers handlers that only fire on the server (invoke_llm, approve_tool, call_tools, after_tool_calls, after_llm_turn, context_removed), log a warning at load.
  - Render handlers, output_renderer, transform_user_prompt, themes and remaps work as normal.
- `ex6.py` itself is checked by hash in hello. A mismatch is refused with a clear message.
- Editing plugins means relaunching `ex6`. In local mode, if the server's plugin hash differs from disk, ask "server runs old code (N agents running). restart? [y/N]".

### Process / connection
- Default `ex6`: look up `get_folder("server")/<project>-<path-hash>/server.json` `{pid, port, token}`. Check the pid is alive and connect. Otherwise spawn `ex6 --server` detached and wait for server.json.
  - Detached spawn on Windows: pythonw + DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW. On Unix: start_new_session.
- `ex6 --server [--port N]` runs in the foreground, headless (VPS: systemd/nohup). It holds an exclusive lock file, so there's one server per project.
- `ex6 --client host:port --token T` for remote, over an ssh tunnel. `ex6 --stop`. `/restart`.
- Bind 127.0.0.1 only, plus a random token checked in hello. No other auth.
- JSON lines over TCP, heartbeat every 2s. The client shows DISCONNECTED in the status line after 5s of silence and reconnects automatically. On reconnect it sends hello with revs, and the server sends everything that's stale.
- Logs: the server writes server.log next to the client's debug.log. `/dbg` shows both.

## Phases
Keep the old in-process path working until phase 2 lands, then delete it.

1. Read-only sync:
   - `app.side`; `--server` (headless load_plugins plus the stand-in TUI); `--client`.
   - Field classification and its assert; snapshot, strict encoder and applier; `_rev` with the `__setattr__` hook; audit safety net; streaming appends; app snapshot (roster, `app.shared`); hash self-check and `/sync`.
   - hello with code identity (plugin shipping); local auto-spawn and discovery; heartbeat and reconnect.
   - Smoke test: spawn a server with a fake invoke_llm generator (chunks plus one tool call). Connect a headless client. Assert client and server hashes match after settling, including across a forced disconnect.
2. Events:
   - command/key/submit events; client Context no-ops (methods, `__setattr__`, data); server UI no-ops plus background echo broadcast.
   - `app.send`/`@ex6.receive`; origin tracking for client-local plugins; name pinning and greyed stand-ins; ctrl-x stop through the key path. Delete the single-process path.
3. ask_human: change the signature to (cls, *args, cancel, **kwargs), add synced requests and reply events, update the `_ex6/tools.py` call sites (~1412, 1420, 1439) and `request_approval` (ex6.py ~2349).
4. Split commands and the plugin inventory:
   - Add `command_server`/`command_client`, `tui.run_command`, `tui.push_panel` (replace the direct `ui_panel_stack.append` in commands, plans, themes, agents_long_horizon), `tui.open_input`, `tui.yank`.
   - Go through every plugin. Known sites:
     - agents.py setup: inline watcher, server only.
     - tool_summaries.py: `_start` is called from the `render_tool_call` handler. It would fire duplicate LLM calls on the client and the server never renders, so trigger it server-side when ToolCall rows are created.
     - plans.py: split the command; g/a/t become `tui.run_command`.
     - agents_long_horizon: run_horizon and its panel are server only.
     - provider_util /usage reads the server db: command_server.
     - worktrees (git): server.
     - themes: client.
     - profiler, tool_stats, providers_test: decide per file.
     - commands.py: go through all ~10.
     - The 14 `tui.current` uses in plugins: check none run in background threads (meaningless on the server).
5. Persistence and polish:
   - Persistence: the server writes snapshots to `get_folder("server")/<project>/contexts/<name>.json` on rev change, debounced 1s, atomic (tmp + replace).
   - Restore on boot after setup, via the existing schema_id template mechanism (`clone_with_context`; core lacks a format-1 writer, so add `Context.serialize`). Contexts without a schema_id are dropped with a log. Running agents come back stopped; pending requests are dropped.
   - Attachments: pasted bytes go base64 in the submit event and the server stores them; the client shows names only.
   - `/restart`.

## Relevant files
- ex6.py: Context (~1110), invoke/_read_llm_stream (~1331-1416), ask_human/_HumanReply (~1498-1550), clone_with_context/_clone (~1451-1493), App.dispatch_command (~571), run_tui (~616), TUI (~2536), add_context/remove_context, core keys (~2773-2950), _tui_loop (~2994).
- _ex6/tools.py: _HumanPanel (~1290), ask_human call sites.
- _ex6/plans.py, commands.py, themes.py, agents.py, inline.py, tool_summaries.py, agents_long_horizon.py, worktrees.py, provider_util.py.
