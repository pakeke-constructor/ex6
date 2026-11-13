

# Project: ex6
A plugin-based, hyper-minimal harness and TUI for coding agents/assistants

<core-problem>
## Core problem:
Existing coding agent harness are often bloated, and inject a tonne of stuff, tools, tokens into context windows implicitly.

Eg, If I invoke a model with the phrase "hello model", modern coding harnesses (like claude-code or codex) end up with 40,000 tokens used.
This is great for noobs. Not so great for high-performing engineers who want to optimize.
Likewise, with a prebuilt harness, you have a lot less control over tools and control-flow.

Where are my tokens going? Is this well-optimized and specialized for my exact use case?
</core-problem>

<solution>
## Solution and project goals:
ex6: A hyper-minimal TUI coding harness where you create agents from scratch.
Every token is explicit, every tool is explicit. 
Plugins give absolute control over everything, from orchestration to tools.

- Serves as a thin, simple layer; no leaky/overreaching abstractions.
- No hidden/implicit context. User can see entire ctx window, and has FULL control.
- Total degree of customization/control via plugins.
- Lives in terminal.
- Can be used as a standalone backend 
</solution>

<architecture>
- `ex6.py`: project file. EVERYTHING is laid out in this file.
- Debugging: use `app.debug_print(...)`, never stdout. Logs: `ex6.get_folder("logs")/<project>-<path-hash>/debug.log`; `app.debug_path` or `/dbg` gives the exact path.

# ex6.py Classes:
Core:
- `App`: plugin registry (commands, overrides, handlers, context hooks), theme, budget. `app.call(name, ...)` dispatches overridables/handleables.

- `TUI`: holds contexts, `current`, mode (normal/insert/scroll), `zoom`, `favorite` (template for `n`), `ui_panel_stack` (modal draw fns). Main loop in `_tui_loop`.

- `Context`: a single LLM conversation, plus the agent running it. See below.
<Context>
A Context represents an agent, alongside a conversation with an LLM.
It is the most important primitive in ex6.
A copntext contains messages, its model, and the tools it can use.
Every agent you see in the agent list is a Context.

Instantiation: `Context(app, "name", model="openai/model", reasoning="low", messages=[...], tools=[...])`. An agent is mostly "a system prompt plus some tools".

Running it: `ctx.invoke("some prompt")` adds the user message and starts the LLM in a background thread. It keeps looping (LLM replies, tools run, LLM replies again) until the model stops calling tools. `ctx.stop_early = True` asks it to stop.

Human panels: `ctx.ask_human(panel, cancel=None)` blocks until completion (None on stop/removal or `cancel()` returning true). No base class: implement `action(name, reply)`, `input(inpt, r, reply)`, and `draw(buf, r, focused)`. NORMAL routes Enter → `activate`, i/space/a → `edit`, arrows → `up`/`down`, PgUp/PgDn and ctrl-u/d → `page_up`/`page_down`. Focused input goes to `input`, one key per call; core intercepts ctrl-x (stop) before it and an unconsumed Esc (release focus) after it. `reply.focus(label="panel")`, `reply.blur()`, `reply.complete(value)` control focus/completion; False and None are valid results, and method returns have no special meaning. Drawing region covers the chat body through the footer; panel owns layout, hints, and any editor/selection/scroll state. Core owns waiting, cancellation, focus ownership, and key-spillover protection. `_ex6/tools.py` supplies the standard approval/question UI; `request_approval(ctx, tool, args, panel)` adds policy/yolo without approval state in the generic lifecycle. `h` visits in NORMAL.

Which backend runs the request?
The `ctx.get_invoker()` polls a handler to choose between models. With the default plugins in _ex6, anthropic models use claude-code backend, openai models use codex-backend, and both of these fallback to openrouter. (To force a backend, pass `invoke_llm=some_fn` when building the Context.)

Messages: read them with `get_messages()`. Also see `append_message()`, `edit_messages(fn)`, and `truncate(i)` to cut history back to message i.

Copies and resets: `fork()` gives you an independent copy to branch off from (handy for subagents). `clear()` wipes the conversation but keeps the system prompt.

If you need to store immutable values on the context, you can use `ctx.data["plugin:key"]` (strings, numbers, bools only). They survive forks.
If you need to store more complex data (like lists, dicts, objects), you must use `ctx.data_volatile`, which is wiped on fork and clear. This keeps forking/clearing robust.
</Context>

- `Message`: represents a message (system, user, assistant, tool). Content can be str, ToolResult, or callable(ctx) for dynamic/lazy content (snapshotted for caching). System msgs carry tools: `with_tools` (same name replaces in place), `without_tools(name_or_fn)`.
- `DailyBudget`: daily spend tracking + limit (`app.budget`).
- `Theme`: color palette dataclass (`app.theme`).

LLM / tools:
- `ResponseChunk`: streamed chunk from invoke_llm ("text" or "cot").
- `LLMResult`: final result of invoke_llm: tokens, tool_calls, native_calls, finish_reason, error, cost.
- `NativeTool`: provider-hosted tool (eg web search), placed in tool lists like a normal tool. Provider runs it: no schema sent, ex6 never calls it, `approve_tool` is NOT consulted. Providers raise on names they don't support. Each call is recorded as `{id, name, args, result}` in `LLMResult.native_calls` -> `Message.native_calls` (display/debug only, never sent back), rendered as `native:<name>` tool rows, logged by `log_invoke`. `ctx.get_native_tools()` lists names.
- `ToolResult`: tool output: text + attachments.
- `PendingTool`: a tool call awaiting its result.
- `Operation`: long-running tool work; tool returns cancel/poll fns, core polls until done.
- `ToolCall`: display row for a tool call (call, result, status running/ok/error, tokens).
- `Attachment` / `ImageAttachment` / `AudioAttachment`: file-backed attachments (path, mime).
- `InputValue`: submitted input text + attachments.
- `StrictDataDict`: dict that only allows str/int/float/bool/None.

Rendering / input:
- `ScreenBuffer`: used to render to screen.
- `WrapWriter`: char writer that wraps at width (`buf.writer(x, y, w)`).
- `Region`: represents (x,y,w,h) tuple; used for immediate-mode layout.
- `InputPass`: cleared every frame; handles input management/blocking (`inpt.consume(key)`).
- `InputBox`: text input widget (cursor, word nav, image paste). `handle(inpt, width)` processes input; `render(buf, r, txt_color, cursor)` only draws. `draw` combines both, returning submitted InputValue (also passed to optional on_submit).

Plugin decorators:
- `@ex6.command`: `/name` command, `fn(tui, args...)`; args coerced from type hints.
- `@ex6.key("dd", "<C-x>")`: normal-mode (list focused) key binding, `fn(tui, count: int)` (count from leading digits, default 1). Core defines the default keymap (vim-like, see UI/UX); binding an existing sequence replaces it. No timeouts, so a binding can't be a prefix of another (`app.bind` raises). `tui.echo(msg)` shows a status-line message.
- `app.remap("kj", "<Esc>")` (in `setup`): INSERT key-to-key remap (noremap), resolved by `InputBox.handle`, which applies keys in order. A held prefix shows at the cursor until the next key (no timeouts); keys that start no remap are never held. Boxes take `app.remaps["insert"]` (`make_input(app)`).
- `@ex6.override`: replace an overridable fn (call_tools, make_input, render_* fns).
- `@ex6.handler(order=0)`: add a handler to a handleable chain (invoke_llm, approve_tool, supports_audio, render_footer, render_tool_call): first truthy result wins, else default. `transform_user_prompt` is a pipeline instead: each handler transforms the text in turn. It runs on input box Enter; a changed prompt is shown first, next Enter sends (`ctx.invoke(text)` sends raw).
- `@ex6.output_renderer`: `fn(lines, msg, ctx)` mutates rendered lines in place.
- `@ex6.after_tool_calls` / `@ex6.after_llm_turn` / `@ex6.context_removed`: context lifecycle hooks.
</architecture>

<plugin-ideology>
## Plugin ideology:
`_ex6/` is the folder where the user's "plugins" are kept, per project. On boot, ex6 loads all python files in `_ex6` folder, and calls `setup(app)` if a file defines it.
Without plugins, ex6 does *NOTHING.* Plugins call the LLM, control contexts, and even define what terminal-UI is.

Loading: core `_ex6/` (next to ex6.py) loads first, then user `<get_folder()>/_ex6/` (per-machine, e.g. opting into logging), then project `_ex6/` (cwd). Same-named files across dirs collide: first wins, later skipped (debug_print warns). Files/subfolders starting with `_` are skipped. Files load in sorted order (hence `z_` prefix = load last).
</plugin-ideology>

<overrides-vs-handlers>
## Overrides vs handlers:
ex6 core has default fns that plugins can replace or extend. Two types, Overrides and Handlers:
- Override: exactly one implementation (e.g. how the transcript is drawn). A plugin replaces the default wholesale. Only one override per fn; a second one raises an error, so conflicts can't hide.
- Handler: many plugins should get a say (e.g. LLM backend per model, tool approval, prompt rewriting). Handlers stack in `order`. For `app.call`, the first truthy result wins, else the default runs. For `app.pipeline` (`transform_user_prompt`), each handler transforms the value in turn.

```python
@ex6.handler
def invoke_llm(ctx):           # returns the backend fn that runs ctx, or None to abstain
    if ctx.model.startswith("mine/"): return my_backend

@ex6.handler
def approve_tool(ctx, tool, args):
    if ctx.name.startswith("explore"): return True   # allow
    # return None -> abstain, next handler decides
```
</overrides-vs-handlers>

<core-plugins>
## Core plugins:
Tools / agents:
- _ex6/tools.py - all core tool-definitions: read_file, edit_file, patch_file, powershell, read_body, read_headers, explore_agent, handoff, etc. Also ENV_PROMPT, AGENTS_MD messages, and render_tool_call rows for file tools (`edit_file a.py +3 -2`).
- _ex6/web_tools.py - websearch_agent tool (OpenRouter `:online` subagent)
- _ex6/skills.py - load_skill tool; progressive disclosure of `_ex6/_skills/*.md` (yaml frontmatter)
- _ex6/plans.py - `/plans`: list `.plans/*.md` (UNBLOCKED, then STALE, then oldest) with status + preview; `g` run, `a`/`i` ask, `t` triage, `dd` delete; `●` = plan has an agent. New agents fork `tui.favorite`. `/plans triage` cleans up all plans
- _ex6/agents.py - basic agent definitions (system prompts, model presets, tool groups READ/WRITE/SHELL/SUBAGENTS/ASK), as an example. `add_native_webtools(ctx)` opts a context into its invoker's native web tools (`<provider>.WEB_TOOLS`; no-op + debug_print if none). Projects usually define their own agents.
- _ex6/reminders.py - after_tool_calls hook factories that append tags to the last tool result: `context_usage_reminder()` (`<context_usage>` every 10k tokens, handoff nag/demand past thresholds) and `time_reminder()` (`<time_reminder>` every 30s once the current invoke() passes a soft limit: 5m, subagents 2m). Overrides via `ctx.data["reminders:..."]`
- _ex6/agents_long_horizon.py - overnight loop over a markdown task file. No command; projects call `run_horizon(app, path, agents, pipeline=default_pipeline)` from their own command, agents (dict name -> Context) built via `default_agents` or by hand (must be yolo=True; see `_plugin_examples/_example_horizon.py`). Custom pipelines compose the step fns (run/review/fix_review/update_docs); the loop owns branching/merging and stops fatally if a pipeline moves HEAD or master (`check_git_state`)
- _ex6/agents_sound.py - Freesound SFX sourcing agent (search/download/listen/install + credits). No context by default; projects call `make_sound_agent(app, project, sfx_dir, credits_file, **context_kwargs)` (see `_plugin_examples/_example_sound.py`). Needs FREESOUND_API_KEY in project .env
- _ex6/tool_summaries.py - bash/powershell rows show a cheap-model summary (`bash [!] running tests`; `!` = changes things, `?` = read-only), stored in ctx.data
- _ex6/sigils.py - `;ssot`, `;git`, `;fast`, etc. in a prompt expand to `<user_reminder>` blocks on Enter (transform_user_prompt handler)

Providers / models:
- _ex6/provider_util.py - shared provider helpers: `log_invoke` (opt-in via `LOG_INVOKES = True`; every invoke -> `<get_folder("provider_util")>/invokes2.db`, deduped message blobs, auto-culled past 1GB), /usage.
- _ex6/provider_openrouter.py - openrouter provider, fallback invoke_llm handler (order=100) for every model. Also cache_manually, supports_audio handler.
- _ex6/provider_openai.py - openai provider, via ChatGPT subscription. invoke_llm handler for `openai/` ids (no `:` suffix). `WEB_TOOLS`: WEB_SEARCH
- _ex6/provider_anthropic.py - anthropic provider, via long-lived `claude -p` process. invoke_llm handler for `anthropic/` ids (no `:` suffix). `WEB_TOOLS`: WEB_SEARCH, WEB_FETCH
- _ex6/providers_test.py - smoke test for silent provider failures (caching, tool args/results, images, protocol drift).
- _ex6/models.py - model registry `M` (ModelInfo: id, ctx window, pricing)
- _ex6/global_spend_limit.py - sets app.budget.limit

UI / commands:
- _ex6/commands.py - commands like /clr, /cm, registered via @ex6.command. Per-agent actions are core keys
- _ex6/themes.py - themes (oklch palettes) + /theme command, persisted to `<get_folder("themes")>/theme.json`
- _ex6/z_highlight_codeblock.py - output_renderer: pygments highlighting for code blocks
- _ex6/z_highlight_markdown.py - output_renderer: markdown highlighting
- _ex6/z_render_system_prompts.py - output_renderer: collapses system messages into a preview; first one also lists tools + schema token cost
- _ex6/profiler.py - Profiling. Use this if you want to benchmark.

Not loaded (underscore-prefixed):
- _ex6/_test.py - old test agent setup
- _ex6/_plugin_examples/ - example plugins (commands, output_renderer, loading_cube, horizon)
</core-plugins>

<coding-style>

</coding-style>

<ui-ux>
## UI layout / UX:
One screen, two panels side by side:
Left panel - list of named contexts (agents); select one to work in.
Right panel - The chat window. Contains the selected agent: header (name, model, provider, budget, ctx bar, tool count), its full transcript (scrollable), and its input box + footer (/command autocomplete).

Follows a vim-like style:
NORMAL = navigate and act. Core keymap: j k, i/space/a (edit request or prompt), Enter (submit request, otherwise edit prompt), / (command), h (visit waiting request in NORMAL), g go, z yolo, w zoom, ctrl-u/d scroll, s/ctrl-x stop, f/o fork, n new (from `tui.favorite`), yy yr yc (yank), p dd da cc cp ca, ?. `?` and /help list them (`ex6.text_panel`).
INSERT = edit a specific request or prompt in the regular input area. Esc keeps the draft and returns to NORMAL; submitting returns to NORMAL. Incoming requests never steal editing focus. Status names the input owner.

Arrows scroll content; with choices, up/down selects in NORMAL. PgUp/PgDn scroll request content or transcript; `<C-u>`/`<C-d>` in NORMAL. Only j/k switch agents. Digits are counts, not choice shortcuts.
List is in insertion order (plugins set it); forks/new agents go right under their source, browser-tab style. List shows relative numbers and markers (`?` needs a human, `●` finished while not selected, `!` yolo).
Human requests (ctx.ui_stack) replace the selected agent's transcript while inspected; their editor uses the regular input area. Enter approves an empty approval or submits its denial draft, even after Esc. Choice questions submit the selection unless a custom draft exists. Text-only questions open editing when empty. Hints reflect the current action. `/` always targets command input, not the request.
`w` toggles zoom for the selected agent: hides the list (clean mouse-copy, long lines). Per agent; each restores its own zoom.
Bottom row is a status line (NORMAL/INSERT/ZOOM, agent, echo, pending keys + which-key popup). `/` commands run from either focus.
</ui-ux>

<important-details>
- You are working with an experienced engineer. Be terse; don't over-explain.
- Simple code > "correct" code. No unnecessary error handling, no overengineering for the sake of "best practices".
- No complex one-liners, no deep nesting, no clever abstractions.
- If a feature needs >300 new lines, stop and ask how to simplify.
- Do not write huge and fragile test classes. For simple pure functions, testing is OK. For anything big, write a temporary `smoke_test.py` python file, and then delete it afterwards.
</important-details>


