# remote: ex6 as client-server (server-authoritative sync)
Status: active

## Motivation
I want to keep my agents running after I quit the TUI, long-form on a VPS, without the keypress lag of tmux/ssh.
`ex6 --server` owns the agents. `ex6 --client` is the TUI.
Plain `ex6` is also client + server: it connects to this project's server, or spawns one detached. One code path, no single-process mode, so anything broken remotely is broken identically locally.

## Ideology (the important part)
The only things that need to be clientside are typing, navigating around contexts, and simple UI state (NOT context state).
Everything else is serverside.

Robust, simple, stateless code is best. When in doubt, pick the option with less state and fewer special cases.

- Client: input box and drafts, selection (`current`), j/k, scrolling, zoom, which-key, status line, theme, yank, and panel drawing/cursor/selection.
- Server: contexts, messages, LLM loops, tools, files, git, plugin data, budget, ask_human waiting, and command logic.
- The server is the single writer of agent state. Client mirrors are read-only, and writing to one raises.
- State flows server -> client as snapshots. They're idempotent and last write wins, so a missed one is healed by the next.
- Requests flow client -> server: commands, keys, prompt submits, and human replies.
- Nothing runs on both sides.
- No client-side prediction. Agent changes appear after a round trip; UI changes are instant.
  - E.g. pressing `f` sends the fork to the server, and the new agent appears when its snapshot arrives.
- Only data crosses the wire, never code references.
- Fail loudly: non-JSON data, a code mismatch, or an unclassified field raises right where the bug is.

## Shape
This section is a sketch, not a spec. Pick the simplest thing that works.
- The context registry (`contexts`, `add_context`, `remove_context`, `favorite`) moves from TUI to App. It's real on the server and a mirror on the client.
- Commands and keys run on the server by default. UI-only ones (j/k, zoom, scroll, help, yank, theme, ...) are marked client-only.
- A command that needs both sides: the server part does the work and returns JSON, then the client part does UI with that result.
  - E.g. fork returns the new name and the client selects it. `change` returns the popped text and the client opens the input with it.
- Requests carry `current` (the client's selected agent). Server code never touches UI; `app.tui` is None on the server.
- Ordering: the server sends any resulting state before it answers a request, so client follow-ups see it.
  - Keys typed while a request is in flight must not act on stale selection (e.g. `f` then `dd` must not delete the parent).
- ask_human: pending requests are synced state on the server (panel class plus JSON args). The client builds panels from that state and sends replies back as requests.
  - This makes it reconnect-safe: a lost reply makes the panel reappear.
- Server background code reaches the user via `app.echo`, which is sent to the client.
- Plugin data the client needs for rendering is synced as state (e.g. `app.shared`). One-shot messages are only for things that don't need to survive a reconnect.
- Both sides run the same plugin code. On handshake the server ships its plugin sources and the ex6.py hash.
  - Client-local plugins (from the client's user folder) can only do client things.

## Sync concerns (must be handled; method is open)
- Field sync: unmarked Context fields are synced implicitly, but only if the value is a primitive (str/int/float/bool/None).
  - Fields that stay on their side are marked `Local[...]` in the annotation: `_scroll_up: Local[int] = 0`. This uses `Annotated[T, "local"]`, so defaults and `field(...)` are unchanged.
    - Local today: app, data_volatile, `_` UI/view fields (_scroll_up, _zoom, _input_box, _prev_height), last_seen, ui_stack, and server internals (_context_hooks, _msg_lock, _read_hashes, _line_snapshots, _scheduled, _pending_inputs, _clear_pending).
  - No lists or dicts in automatic sync. Complex fields are a short fixed list, synced by hand in `snapshot()` and the applier:
    - `_messages`, `pending_message`, `data`, `tools` (as schemas), `pending_tools` (PendingTool -> display rows), `llm_result`, `invoke_llm` (as its name), and later the pending human requests.
  - An unmarked, non-primitive field that isn't hand-synced raises in `snapshot()`: "mark it Local[...] or sync it by hand".
  - The mistakes are loud on their own: a forgotten UI field raises on a mirror write, and a forgotten object raises in the encoder. No boot-time assert is needed.

```python
T = TypeVar("T")
Local = Annotated[T, "local"]

def snapshot(ctx):
    out = {"messages": [...], "tools": [...], "data": dict(ctx.data), ...}  # hand-synced fields
    for f in fields(ctx):
        if is_local(f) or f.name in HAND_SYNCED: continue
        v = getattr(ctx, f.name)
        if not isinstance(v, (str, int, float, bool, type(None))):
            raise TypeError(f"Context.{f.name} isn't a primitive: mark it Local[...] or sync it by hand in snapshot()")
        out[f.name] = v
    return out
```
- Revisions: in-place mutations must bump the revision explicitly. Examples are ToolCall status changes and `ctx.data[k] = v`, which tool_summaries writes often.
  - Any periodic audit or hash compare is only a bug detector, never the normal path.
- Bandwidth: long transcripts over ssh can't be resent in full on every change (e.g. send only the changed tail).
- Streaming text goes as offset-checked appends; a mismatch is healed by the next snapshot.
- Snapshotting must not freeze callable message content: `Message.get_msg` caches `_snapshot`.
- Removal shows up as a roster diff, never as a "removed" event, so a missed event can't leave a zombie.
- Client and server can compare state hashes; `/sync` shows them.

## Process / connection
- Default `ex6`: look up `get_folder("server")/<project>-<path-hash>/server.json` `{pid, port, token}`. Connect if alive; otherwise spawn `ex6 --server` detached.
- `ex6 --server [--port N]` runs headless in the foreground. A lock file allows one server per project.
- `ex6 --client host:port --token T` connects remotely over an ssh tunnel. Also `ex6 --stop` and `/restart`.
- The server binds to 127.0.0.1 only and checks a random token on hello.
- The client shows DISCONNECTED and reconnects automatically. While disconnected, requests are refused rather than queued.
- One client at a time; a second connection is refused.
- Idle auto-exit: the server exits once no client is connected and no agent is running or waiting on a human for `app.server_idle_exit` seconds.
  - Default is 600. Plugins set it in `setup`; None disables it (e.g. on a VPS).
  - On exit, delete `server.json` and release the lock.
  - Until persistence (phase 5) lands, idle exit discards idle conversations, so it ships with phase 5 (or defaults to None before then).
- Editing plugins means relaunching. In local mode, if the server's code differs from disk, offer to restart it.
- The server writes server.log next to the client's debug.log, and `/dbg` shows both.

## Crashes: server crash = client crash, loudly
- An exception inside a request is not fatal: it becomes an error echo, and the traceback goes to server.log.
- A genuine server crash (`app.fatal_error`) writes `crash.txt` and sends `fatal` to the client if it can, then exits. The client raises `FATAL: SERVER: ...` just like today.
- If the server died silently, a local client detects it via the dead pid. A remote client shows `crash.txt` on reconnect.
- A client crash or disconnect never affects the server.

## Phases
Keep the in-process path working until phase 2, then delete it.
1. Read-only sync: move the registry to App, add `--server`/`--client`, snapshots, handshake with code shipping, spawn/discovery, and reconnect.
   - Smoke test: a fake invoke_llm (chunks plus a tool call) plus a headless client. Hashes must match, including after a forced disconnect.
2. Requests: commands, keys, and submits go to the server; mark client-only keys and commands; the ctrl-x stop in `_tui_loop` goes through the stop key. Delete the in-process path.
3. ask_human as synced state, plus reply requests. Update the `_ex6/tools.py` call sites and `request_approval`.
4. Plugin inventory: go through every plugin and decide which side each part runs on. Known sites:
   - tool_summaries.py: it currently triggers from `render_tool_call`. It must trigger server-side when ToolCall rows are created.
   - plans.py: listing and running go on the server; the panel goes on the client.
   - agents_long_horizon: server-only.
   - provider_util `/usage`, worktrees: server.
   - themes: client.
   - commands.py, profiler, tool_stats, providers_test: decide per file.
   - Direct `ui_panel_stack.append` sites: client.
5. Persistence (server writes contexts to disk and restores them on boot), idle auto-exit, attachments in submits, `/restart`.

## Relevant files
- ex6.py: Context (~1110), invoke (~1364), ask_human/_HumanReply (~1501-1550), clone_with_context (~1454), App.dispatch_command (~571), run_tui (~616), TUI (~2539), core keys (~2740-2960), _tui_loop (~2994, ctrl-x ~3090/~3146).
- _ex6/tools.py: _HumanPanel, ask_human call sites.
- _ex6/plans.py, commands.py, themes.py, agents.py, tool_summaries.py, agents_long_horizon.py, worktrees.py, provider_util.py.
