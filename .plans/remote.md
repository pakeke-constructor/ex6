# remote: ex6 as client-server (server-authoritative sync)
Status: parked


# HUMAN NOTE: I tried implementing this in a new branch, `dev/remote`. I won't lie; it feels kiiindof bad. It feels like this is too complicated for what ex6 should be.  So right now, im parking the plan


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
- Requests flow client -> server: commands, prompt submits, and human replies. Keys are client-side and send requests.
- Nothing runs on both sides.
- Selection is client-only. The server has no "current agent": every request carries the client's selected agent name.
- Keys never wait on the network. The only client-side prediction is in-flight requests (pending forks, hidden deletes), and it lasts exactly as long as the request (see "Requests").
- Only data crosses the wire, never code references.
- Fail loudly: non-JSON data, a code mismatch, or an unclassified field raises right where the bug is.

## Shape
This section is a sketch, not a spec. Pick the simplest thing that works.
- The context registry (`contexts`, `add_context`, `remove_context`, `favorite`) moves from TUI to App. It's real on the server and a mirror on the client.
- Keys always run on the client. Commands run on the server by default; UI-only commands (theme, help, ...) are marked client-only. One name lives on one side; registering it on both raises.
- Server code never touches UI; `app.tui` is None on the server.
- ask_human: pending requests are synced state on the server (panel class plus JSON args). The client builds panels from that state and sends replies back as requests.
  - This makes it reconnect-safe: a lost reply makes the panel reappear.
- Server background code reaches the user via `app.echo`, which is sent to the client.
- Plugin data the client needs for rendering is synced as state (e.g. `app.shared`). One-shot messages are only for things that don't need to survive a reconnect.
- Both sides run the same plugin code. On handshake the server ships its plugin sources and the ex6.py hash.
  - Client-local plugins (from the client's user folder) can only do client things.

## Requests (client -> server)
A request is `{id, current, args}`, with the command name as `args[0]` (`tui.request("fork", name)` -> `{id="command", args = ["fork", "foo.2"]}`). `current` is the client's selected agent name (or None); the server resolves it per request and fails loudly if it's unknown. Nothing on the server stores a selection.

```python
# server: commands get the requesting client's selected agent
@ex6.command
def fork(ctx, name: str): ctx.fork(name)

@ex6.command
def delete(ctx, names: list[str]): ...      # explicit names, never "whatever is spawned now"

# client: keys are UI; they send requests and never block
@ex6.key("f")
def key_fork(tui, n):
    name = tui.unique_name(tui.current_name)    # checks mirror + pending names
    tui.request("fork", name, pending=[name])
    tui.select(name)

@ex6.key("dd")
def key_delete(tui, n):
    tui.request("delete", [tui.current_name], hide=[tui.current_name])

@ex6.command_client                             # client-only command fronting a server one
def plans(tui):
    tui.request("plans_list", on_reply=lambda r: tui.push_panel(_PlansPanel(r)))
```
- Typed `/x args`: a client-only command runs locally; otherwise it's sent as a request.
- Every request gets exactly one reply: ok + JSON result, or an error (echoed). `on_reply` is optional.
- The server handles requests in TCP order, so follow-ups never overtake: `f` then `o`/submit/`dd` on the new name always lands after the fork.
- After running a request, the server runs the diff pass right away and pushes the resulting state, then replies, on the same socket. No waiting for the next poll, so locally the round trip is ~1-2ms.
- Request handlers must return quickly; long work goes in a thread (`ctx.invoke` already does). A request unanswered for 10s is a loud error, and the client drops and reopens the connection.

Client-chosen names: `f`, `o`, `n` pick the new name on the client (`unique_name` over mirror + pending names) and select it immediately. Selection is stored as a name, so it can point at an agent that doesn't exist yet. The list shows it as pending, and the chat pane shows "pending..." until the snapshot arrives. If the name was taken on the server in the meantime (e.g. a subagent), the fork fails loudly.

`p`/`cp`/`ca` don't wait either: the client finds the cut and the text in its mirror, opens the input with the text, and sends `truncate(cut, expect=<hash of message at cut>)`. The server refuses if the hash differs or the agent is running. The draft is local, so a refusal loses nothing.

### Pending and hidden: can never drift
`pending` (forks not confirmed yet) and `hide` (deletes not confirmed yet) are not sets the client maintains. They're derived from the connection's in-flight requests:
`visible list = server roster - hide names of in-flight requests + pending names of in-flight requests not yet in the roster`
- A request leaves in-flight only when its reply arrives. The server pushes state before replying, so at that moment the roster is already the truth.
  - Delete succeeded: the name is gone from the roster. Delete failed, or the server didn't actually delete: the agent reappears. Either way the list matches the server.
  - Fork failed: the pending row disappears.
- In-flight requests belong to the client Connection. A disconnect drops them all, so nothing survives a reconnect.
- The 10s request timeout bounds a hung server: in-flight can't outlive it.
- So no hidden/pending entry can ever outlive its request, and no separate cleanup or timeout logic exists.
- Selection fallback: if the selected name is neither in the visible list nor pending, select its nearest neighbour by last list position. `dd` and failed forks both land there.

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
- Knowing when to send: the server has to notice whenever a context changes, so the client sees it.
  - The obvious way is to catch every write as it happens (a `__setattr__` hook plus a revision counter). That misses changes made inside a field, like `ctx.data[k] = v` or a ToolCall's status changing, because those never go through `__setattr__`. Each such spot would need a manual "this changed" call, and forgetting one means the client silently shows stale state.
  - So instead, the server just looks: about 10 times a second, it rebuilds each context's snapshot and compares it with the last one it sent. If they differ, it sends it.
  - This can't miss a change, because it compares the result, not the writes. No revision numbers, no hooks, nothing to forget.
  - It's cheap because the snapshot holds no message contents: just the simple fields, `data`, tool rows, and the list of message hashes.
  - The app snapshot (roster, budget, favorite, `app.shared`, `fatal_error`) works the same way.
- Messages are content-addressed. A snapshot always carries the full message list, but as hashes; content is sent only for hashes the client doesn't have yet.
  - The hash is a sha of the message's encoded JSON (role, content, tool calls, attachments, system-prompt tool schemas): whatever gets sent is what gets hashed, so there's no second field list to maintain.
  - Hashes ignore which context a message belongs to, so forks and shared system prompts cost almost nothing to sync.
  - Edits need no special handling: changed content means a new hash.
  - The client keeps a hash -> message store. A hash it doesn't have is a bug, so it fails loudly and resyncs (no holes in the transcript).
  - Hashing every message on every poll is too slow for long conversations, so the hash is cached on the Message. This is the only cache that can go stale: `edit_messages` hands plugins the raw list, so a message can change without anyone clearing the cache.
    - Fix: like `_render_cache`, remember which content object the hash was computed from, and recompute when `self.content` is a different object. Replacing a message's content then rehashes automatically. Only a change inside a ToolResult object would be missed, which is rare enough to accept.
- Streaming text uses the same comparison: if the new text starts with the text already sent, send just the new part with its offset; otherwise send it whole. The client applies an append only if the offset matches its current length; a mismatch is healed by the next snapshot.
- Callable message content: the snapshot uses `get_msg`, which freezes `_snapshot`, the same as viewing the transcript already does. Don't avoid this: evaluating callables on every poll would run e.g. `_env_content`'s `git branch` 10x/s.
- Tool schemas: cache `tool_to_schema` per tool fn, so the poll doesn't rebuild them every time.
- Removal shows up as a roster diff, never as a "removed" event, so a missed event can't leave a zombie.
- `/sync` compares the client's and server's message hash lists per context.

## Connection
`Connection` (server side) represents one remote party interacting with ex6: one socket, from hello to disconnect.
- It owns all per-client state: the socket and send lock, the set of message hashes already sent, the last snapshot sent per context and for the app, and the request reader.
- When the socket closes, the object is thrown away, and so is all of its state. A reconnect starts from an empty sent-set, and the client clears its message store to match. Nothing can go stale across reconnects.
  - Within one connection, TCP is reliable, so the server's sent-set always matches the client's store.
- Agent state (contexts, `app.shared`) lives on App, never on a Connection. A Connection only tracks what it has sent.
- Only one Connection exists at a time, and a second is refused. Multiple clients later would just mean more Connection objects.

## Process / connection
- Default `ex6`: look up `get_folder("server")/<project>-<path-hash>/server.json` `{pid, port, token}`. Connect if alive; otherwise spawn `ex6 --server` detached.
- `ex6 --server [--port N]` runs headless in the foreground. A lock file allows one server per project.
- `ex6 --client host:port --token T` connects remotely over an ssh tunnel. Also `ex6 --stop` and `/restart`.
- The server binds to 127.0.0.1 only and checks a random token on hello.
- The client shows DISCONNECTED and reconnects automatically. While disconnected, requests are refused rather than queued.
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
1. Read-only sync: move the registry to App, add `--server`/`--client`, snapshots, handshake with code shipping, spawn/discovery, and reconnect. Keep this human-readable, with well-named functions and a clear top-level flow.
   - Smoke test: a fake invoke_llm (chunks plus a tool call) plus a headless client. Hashes must match, including after a forced disconnect.
2. Requests: `tui.request` with pending/hide, server commands get `(ctx, *args)`, client-chosen names, push-before-reply, request timeout. Core keys become client keys sending requests (go, yolo, stop, fork, new, pop, delete, clear, change). Move `tui.current`/`tui.favorite` users to names (`tui.favorite` -> `app.favorite`, a synced name). The ctrl-x stop in `_tui_loop` goes through the stop key. Delete the in-process path.
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
