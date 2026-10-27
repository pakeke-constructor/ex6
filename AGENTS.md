

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
- Printing/debugging: never print to stdout (breaks the TUI). Use `app.debug_print()` (same signature as print), e.g. `ctx.app.debug_print(...)`.

ex6.py Classes:
Core:
- `App`: plugin registry (commands, overrides, handlers, context hooks), theme, budget. `app.call(name, ...)` dispatches overridables/handleables.

- `TUI`: holds contexts, `current`, mode (selection/work/help/scroll), `ui_panel_stack` (modal draw fns). Main loop in `_tui_loop`.

- `Context`: a single LLM conversation, plus the agent running it. See below.
<Context>
A Context represents an agent, alongside a conversation with an LLM.
It is the most important primitive in ex6.
A copntext contains messages, its model, and the tools it can use.
Every agent you see in selection-mode is a Context.

Instantiation: `Context(app, "name", model="openai/model", reasoning="low", messages=[...], tools=[...])`. An agent is mostly "a system prompt plus some tools".

Running it: `ctx.invoke("some prompt")` adds the user message and starts the LLM in a background thread. It keeps looping (LLM replies, tools run, LLM replies again) until the model stops calling tools. `ctx.stop_early = True` asks it to stop.

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
- `NativeTool`: provider-hosted tool (eg web search), placed in tool lists like a normal tool. Provider runs it: no schema sent, ex6 never calls it, `approve_tool` is NOT consulted. Providers raise on names they don't support. Each call is recorded as `{id, name, args, result}` in `LLMResult.native_calls` -> `Message.native_calls` (display/debug only, never sent back), rendered as `native:<name>` tool rows, logged by `_log_invoke`. `ctx.get_native_tools()` lists names.
- `ToolResult`: tool output: text + attachments.
- `PendingTool`: a tool call awaiting its result.
- `Operation`: long-running tool work; tool returns cancel/poll fns, core polls until done.
- `ToolCall`: display row for a tool call (status running/ok/error).
- `Attachment` / `ImageAttachment` / `AudioAttachment`: file-backed attachments (path, mime).
- `InputValue`: submitted input text + attachments.
- `StrictDataDict`: dict that only allows str/int/float/bool/None.

Rendering / input:
- `ScreenBuffer`: used to render to screen.
- `WrapWriter`: char writer that wraps at width (`buf.writer(x, y, w)`).
- `Region`: represents (x,y,w,h) tuple; used for immediate-mode layout.
- `InputPass`: cleared every frame; handles input management/blocking (`inpt.consume(key)`).
- `InputBox`: text input widget (cursor, word nav, image paste).

Plugin decorators:
- `@ex6.command`: `/name` command, `fn(tui, args...)`; args coerced from type hints.
- `@ex6.override`: replace an overridable fn (call_tools, make_input, render_* fns).
- `@ex6.handler(order=0)`: add a handler to a handleable chain (invoke_llm, approve_tool, render_work_mode_footer): first truthy result wins, else default. `transform_user_prompt` is a pipeline instead: each handler transforms the text in turn.
- `@ex6.output_renderer`: `fn(lines, msg, ctx)` mutates rendered lines in place.
- `@ex6.after_tool_calls` / `@ex6.after_llm_turn` / `@ex6.context_removed`: context lifecycle hooks.
</architecture>

<plugin-ideology>
## Plugin ideology:
`_ex6/` is the folder where the user's "plugins" are kept, per project. On boot, ex6 loads all python files in `_ex6` folder, and calls `setup(app)` if a file defines it.
Without plugins, ex6 does *NOTHING.* Plugins call the LLM, control contexts, and even define what terminal-UI is.

Loading: core `_ex6/` (next to ex6.py) loads first, then project `_ex6/` (cwd). Files/subfolders starting with `_` are skipped. Files load in sorted order (hence `z_` prefix = load last).
</plugin-ideology>

<overrides-vs-handlers>
## Overrides vs handlers:
ex6 core has default fns that plugins can replace or extend. Two types, Overrides and Handlers:
- Override: exactly one implementation (e.g. how the work-mode is drawn). A plugin replaces the default wholesale. Only one override per fn; a second one raises an error, so conflicts can't hide.
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
- _ex6/tools.py - all core tool-definitions: read_file, edit_file, patch_file, powershell, read_body, read_headers, explore_agent, handoff, etc. Also ENV_PROMPT, AGENTS_MD messages.
- _ex6/web_tools.py - websearch_agent tool (OpenRouter `:online` subagent)
- _ex6/skills.py - load_skill tool; progressive disclosure of `_ex6/_skills/*.md` (yaml frontmatter)
- _ex6/tasks.py - plan_write/plan_read/plan_add_log/plan_done/plan_list tools, backed by `.plans/*.md`
- _ex6/agents.py - basic agent definitions (system prompts, model presets, tool groups READ/WRITE/SHELL/SUBAGENTS/ASK), as an example. `add_native_webtools(ctx)` opts a context into its invoker's native web tools (`<provider>.WEB_TOOLS`; no-op + debug_print if none). Projects usually define their own agents. Also starts inline watcher.
- _ex6/agents_long_horizon.py - overnight loop over a markdown task file. No command; projects call `run_horizon(app, path, agents, pipeline=default_pipeline)` from their own command, agents (dict name -> Context) built via `default_agents` or by hand (must be yolo=True; see `_plugin_examples/_example_horizon.py`). Custom pipelines compose the step fns (run/review/fix_review/update_docs); the loop owns branching/merging and stops fatally if a pipeline moves HEAD or master (`check_git_state`)
- _ex6/agents_sound.py - Freesound SFX sourcing agent (search/download/listen/install + credits). No context by default; projects call `make_sound_agent(app, project, sfx_dir, credits_file, **context_kwargs)` (see `_plugin_examples/_example_sound.py`). Needs FREESOUND_API_KEY in project .env
- _ex6/inline.py - inline file prompts: line ending in `;;;` launches an agent (watchfiles + git)
- _ex6/sigils.py - expands `;git`, `;fast`, `;smp`, etc. in user prompts via a cheap LLM (transform_user_prompt handler)

Providers / models:
- _ex6/provider.py - openrouter provider, fallback invoke_llm handler (order=100) for every model. Also cache_manually, /usage, invoke logging.
- _ex6/provider_openai.py - openai provider, via ChatGPT subscription. invoke_llm handler for `openai/` ids (no `:` suffix). `WEB_TOOLS`: WEB_SEARCH
- _ex6/provider_anthropic.py - anthropic provider, via long-lived `claude -p` process. invoke_llm handler for `anthropic/` ids (no `:` suffix). `WEB_TOOLS`: WEB_SEARCH, WEB_FETCH
- _ex6/models.py - model registry `M` (ModelInfo: id, ctx window, pricing)
- _ex6/global_spend_limit.py - sets app.budget.limit

UI / commands:
- _ex6/commands.py - commands like /help, /clr, /cm, registered via @ex6.command
- _ex6/themes.py - themes (oklch palettes) + /theme command, persisted to theme.json
- _ex6/z_highlight_codeblock.py - output_renderer: pygments highlighting for code blocks
- _ex6/z_highlight_markdown.py - output_renderer: markdown highlighting
- _ex6/z_render_system_prompts.py - output_renderer: collapses system messages into expandable preview

Not loaded (underscore-prefixed):
- _ex6/_test.py - old test agent setup
- _ex6/_plugin_examples/ - example plugins (commands, output_renderer, loading_cube, horizon)
</core-plugins>


<ui-ux>
## UI layout / UX:
ex6 has two modes: selection-mode and work-mode.

**Selection-mode:**  
Displays list of named context-windows, user chooses what one to work in.
This UI has 2 panels, side by side:
Left panel - used to select context
Right panel - displays context information

**Work-mode:**  
Prompt LLMs, run commands, see the entire conversation history for this context in the terminal (can scroll up.)
User may be prompted to answer questions / clarifications by the LLM. Such questions should be isolated in the context that asked them.

Both work-mode AND selection-mode have a command-input-box at the bottom.
The user may choose to type commands, (and/or talk to the LLM if in work-mode)
</ui-ux>

<important-details>
- You are working with an experienced engineer. Be terse; don't over-explain.
- Simple code > "correct" code. No unnecessary error handling, no overengineering for the sake of "best practices".
- No complex one-liners, no deep nesting, no clever abstractions.
- If a feature needs >300 new lines, stop and ask how to simplify.
- Do not write huge and fragile test classes. For simple pure functions, testing is OK. For anything big, write a temporary `smoke_test.py` python file, and then delete it afterwards.
</important-details>


