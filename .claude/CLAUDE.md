

# Project description:

## ex6: A tool for context engineering.
This project, `ex6`, serves as a thin, simple alternative to claude-code.

## Project goals:
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
Right panel - used to select model

**Work-mode:**  
Prompt LLMs, run commands, see the entire conversation history for this context in the terminal (can scroll up.)
User may be prompted to answer questions / clarifications by the LLM. Such questions should be isolated in the context that asked them.

**User-Input:**
Both work-mode AND selection-mode have a command-input-box at the bottom.
The user may choose to type commands, (and/or talk to the LLM if in work-mode)
ex6 is intended to be highly-customizable when it comes to workflow.  
As such, we should have question dialogs that are defined AS PLUGINS, but become part of the UI when appropriate.
E.g. when question-dialog appears, it REPLACES the command-input-box.

</ui_description>

<IMPORTANT_DETAILS>
- You are working with a talented engineer who understands the codebase.
- In all interactions, be extremely concise, even if it means grammatical incorrectness.
- When writing code, write the simplest code possible. Aggressively avoid complexity. Do not be afraid to say "hmm, this code is too complex. Let me rewrite that"
- When writing code, readability and simplicity is CRUCIAL. Avoid complex one-liners or deeply nested functions. Avoid list comprehensions unless simple.
- Before appending new code, consider whether it can be made simpler, or shortened. Proper error-handling and "best practices" are less important than short code.
- If you think a feature is too complex/adds too much code, (e.g. over 300 new lines,) you MUST ask the engineer for help/guidance to see how it can be simplified.
</IMPORTANT_DETAILS>


