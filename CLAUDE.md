

# ex6: A plugin-based, hyper-minimal TUI for agentic engineering

### Core problem:
Existing agentic harness are often bloated, and inject a tonne of stuff, tools, tokens into context windows explicitly.
Great for noobs. Not so great for high-performing engineers who want to optimize.

### Core solution:
ex6: A hyper-minimal TUI coding harness where you create agents from scratch. Every token explicit. Every tool explicit.
Plugins give absolute control over everything, from orchestration to tools.


## simple architecture:
- ex6.py: (1.6k lines of code) All of ex6 is in this file.
- _ex6/**: plugins defined here


## design:
Two modes: work-mode (chatting to agent), and selection-mode, (select between agents)
Multiple agents can run in parellel.
ex6 doesn't care how or why an agent is invoked.
ex6 doesn't care how the tokens are passed, or what tokens are passed. Completely neutral, the programmer is given FULL power.

## existing plugins:
- tools: Agentic coding tools — file ops, bash, search, approval flow, subagents.
- provider: LLM invocation and caching.
- models: Model registry (pricing, context limits).
- commands: Slash commands for context management and workflow shortcuts.
- code-mode: The tool-use paradigm. Instead of JSON tool_calls, the LLM writes Python in run_tools blocks. Tools run in parallel threads, return ToolResult futures (.print()/.get()/.status()). Sandboxed via RestrictedPython.
- tasks: Lightweight plan/task tracking.
- skills, themes, highlights: Skill persistence, color themes, syntax highlighting.

Files in _ex6 loaded as plugins.
Any file or folder starting with _ is ignored.

CORE ETHOS:
Plugins should be able to be removed or altered completely, WITHOUT affecting core ex6.


## tech:
Uses python
blessed library for rendering
Double-buffered rendering via ScreenBuffer; clears every frame.

Plugin system: _ex6/ next to ex6.py (core) and _ex6/ in cwd (project-local). Files loaded alphabetically, _ prefix ignored.
Extension points: @overridable/@override for replacing core fns (e.g. invoke_llm, call_tools). @output_renderer for post-processing assistant output. @after_tool_calls for injecting messages between tool runs and next LLM turn. @command for slash commands.
Context: the central object. Holds messages, model, tools, cwd, data dicts. Messages carry tools — tool availability is per-message, not global. Message.content can be a callable (lazy/dynamic system prompts).
Agentic loop: ctx.invoke() spawns a thread that loops: call LLM → call_tools → repeat until no tool calls or stop_early.


