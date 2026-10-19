

# Project: ex6
A plugin-based, hyper-minimal harness and TUI for coding agents/assistants

<core_problem>
## Core problem:
Existing coding agent harness are often bloated, and inject a tonne of stuff, tools, tokens into context windows implicitly.

Eg, If I invoke a model with the phrase "hello model", modern coding harnesses (like claude-code or codex) end up with 40,000 tokens used.
This is great for noobs. Not so great for high-performing engineers who want to optimize.
Likewise, with a prebuilt harness, you have a lot less control over tools and control-flow.

Where are my tokens going? Is this well-optimized and specialized for my exact use case?
</core_problem>

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
- `ex6.py`: project file. EVERYTHING is layed out in this file.

ex6.py Classes:
- `ScreenBuffer`: used to render to screen.
- `InputPass`: cleared every frame; handles input management/blocking
- `Region`: represents (x,y,w,h) tuple; used for immediate-mode layout.
- `Context`: represents a LLM context window (and potentially a running LLM.)
- `Message`: represents a message (system, user, assistant). Can return content dynamically/lazily.

_ex6/**: Built-in plugins

</architecture>

## Plugin ideology:
`_ex6/` is the folder where the user's "plugins" are kept, per project. On boot, ex6 loads all python files in `_ex6` folder, calling their `setup(app)` function.
Without plugins, ex6 does *NOTHING.* Plugins call the LLM, control contexts, add even define what terminal-UI is.

Loading: core `_ex6/` (next to ex6.py) loads first, then project `_ex6/` (cwd). Files/subfolders starting with `_` are skipped. Files load in sorted order (hence `z_` prefix = load last).

## Core plugins:
Tools / agents:
- _ex6/tools.py - all core tool-definitions: read_file, edit_file, patch_file, powershell, read_body, read_headers, explore_agent, handoff, etc. Also ENV_PROMPT, AGENTS_MD messages.
- _ex6/web_tools.py - websearch_agent tool (OpenRouter `:online` subagent)
- _ex6/skills.py - load_skill tool; progressive disclosure of `_ex6/_skills/*.md` (yaml frontmatter)
- _ex6/tasks.py - plan_write/plan_read/plan_add_log/plan_done/plan_list tools, backed by `.plans/*.md`
- _ex6/agents.py - basic agent definitions (system prompts, model presets), as an example. Projects usually define their own agents. Also starts inline watcher.
- _ex6/agents_long_horizon.py - /horizon: overnight loop over a markdown task file (one `## ` heading per task)
- _ex6/inline.py - inline file prompts: line ending in `;;;` launches an agent (watchfiles + git)
- _ex6/sigils.py - expands `;git`, `;fast`, `;smp`, etc. in user prompts via a cheap LLM (transform_user_prompt handler)

Providers / models:
- _ex6/provider.py - openrouter provider, overrides invoke_llm. Also cache_manually, /usage, invoke logging.
- _ex6/provider_openai.py - openai provider, via ChatGPT subscription
- _ex6/provider_anthropic.py - anthropic provider, via long-lived `claude -p` process
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
- _ex6/_plugin_examples/ - example plugins (commands, output_renderer, loading_cube)


## UI layout / UX:
ex6 has two modes: selection-mode, and work-mode.
<ui_description>
**Selection-mode:**  
Displays list of named context-windows, user chooses what one to work in.
This UI has 2 panels, split horizontally:
Left panel - used to select model
Right panel - displays model information

**Work-mode:**  
Prompt LLMs, run commands, see the entire conversation history for this context in the terminal (can scroll up.)
User may be prompted to answer questions / clarifications by the LLM. Such questions should be isolated in the context that asked them.

**User-Input:**
Both work-mode AND selection-mode have a command-input-box at the bottom.
The user may choose to type commands, (and/or talk to the LLM if in work-mode)

</ui_description>

<IMPORTANT_DETAILS>
- You are working with an experienced engineer. Be terse; don't over-explain.
- Simple code > "correct" code. No unnecessary error handling, no overengineering for the sake of "best practices".
- No complex one-liners, no deep nesting, no clever abstractions.
- If a feature needs >300 new lines, stop and ask how to simplify.
- Do not write huge and fragile test classes. For simple pure functions, testing is OK. For anything big, write a temporary `smoke_test.py` python file, and then delete it afterwards.
</IMPORTANT_DETAILS>


