

# Project: ex6
A plugin-based, hyper-minimal TUI for coding agents/assistants

## Core problem:
Existing coding agent harness are often bloated, and inject a tonne of stuff, tools, tokens into context windows explicitly.
Great for noobs. Not so great for high-performing engineers who want to optimize.

## Solution and project goals:
ex6: A hyper-minimal TUI coding harness where you create agents from scratch. Every token explicit. Every tool explicit. Plugins give absolute control over everything, from orchestration to tools.
- Serves as a thin, simple layer; no leaky/overreaching abstractions.
- No hidden/implicit context. User can see entire ctx window, and has FULL control.
- Total degree of customization/control via plugins.
- Lives in terminal.


## Project architecture:
- `ex6.py`: project file. EVERYTHING is layed out in this file.
Classes:
- `ScreenBuffer`: used to render to screen.
- `InputPass`: cleared every frame; handles input management/blocking
- `Region`: represents (x,y,w,h) tuple; used for immediate-mode layout.
- `Context`: represents a LLM context window (and potentially a running LLM.)
- `Message`: represents a message (system, user, assistant). Can return content dynamically/lazily.

printing/debugging: If you want to print, you must use `ex6.debug_print()`. (Same signature as print)

## Plugin ideology:
`_ex6/` is the folder where the user's "plugins" are kept, per project. On boot, ex6 loads all python files in `_ex6` folder.  
Without plugins, ex6 does *NOTHING.* Plugins call the LLM, control contexts, add even define what terminal-UI is.

## Core plugins:
- _ex6/tools.py - contains all tool-definitions like read_file, edit_file, etc
- _ex6/provider.py - openrouter provider, overrides invoke_llm
- _ex6/code_mode.py - Custom tool-calling pipeline. Unlike typical LLM tool-formats, code-mode only exposes one `run_tools` tool to the agent, alongside a whitelist of python functions. The agent then calls `run_tools`, passing a block of python code that contains all the tool-calls they want to do. See code_mode.py for details.
- _ex6/agents/agents.py - agent definitions.
- _ex6/commands.py - commands like /help, /clr, /cm, registered via @ex6.command


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
- Working with an experienced engineer. Be terse; don't over-explain.
- Simple code > "correct" code. No unnecessary error handling, no overengineering for the sake of "best practices".
- No complex one-liners, no deep nesting, no clever abstractions.
- If a feature needs >300 new lines, stop and ask how to simplify.
</IMPORTANT_DETAILS>


