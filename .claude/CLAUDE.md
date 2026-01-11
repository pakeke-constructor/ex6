

# Project description:

## ex6: A tool for context engineering.
This project, `ex6`, serves as a thin, simple alternative to claude-code.

## Project goals:
- Serves as a thin, simple layer; no leaky/overreaching abstractions.
- No hidden/implicit context. User can see entire ctx window, and has FULL control.
- Total degree of customization/control via plugins.
- Lives in terminal.

## Project architecture:
- `ex6.py`: the ENTIRE project, contained in one python file.

## Plugin ideology:
`.ex6/` is the folder where the user's "plugins" are kept, per project. On boot, ex6 loads all python files in `./ex6` folder.  
Without plugins, ex6 does *NOTHING.* Plugins call the LLM, control contexts, add even define what terminal-UI is.


## UI layout / UX:
ex6 has two modes: selection-mode, and work-mode.
<ui-description>
**Selection-mode:**  
Displays list of named context-windows, user chooses what one to work in.
This UI has 2 panels, split horizontally:

- SelectionMode-Left-panel:
Displays a list of LLM contexts. Each context has a name, and they are layed out in a tree-like structure, with children/forked contexts as "child nodes".
eg:
```
ctx1
ctx2
    ctx2_child
    blah_second_child
        nested_child
>> foobar  (the '>>' means that foobar is hovered)
debug_ctx
```
User can hover over contexts via up/down arrow keys, and select a context with enter. (selecting a context will go to work-mode.)

- SelectionMode-Right-panel: 
Displays information about the currently hovered context window.
Example:
```
my-context    opus-4.5
[XXX-------------]
32k / 200k tokens, $0.15
------------
sys-prompt-1 (12k)
sys-prompt-2 (8k)
user-prompt (400)
assistant (300)
```

**Work-mode:**  
Prompt LLMs, run commands, see the entire conversation history for this context in the terminal (can scroll up.)
User may be prompted to answer questions / clarifications by the LLM. Such questions should be isolated in the context that asked them.

**User-Input:**
Both work-mode AND selection-mode have a command-input at the bottom.
The user may choose to type commands, (and/or talk to the LLM if in work-mode)

**Custom UI:**
ex6 is intended to be highly-customizable when it comes to workflow.  
As such, we should have question dialogs that are defined AS PLUGINS, but become part of the UI when

</ui-description>


## Running:
- run using `py ex6.py`


# IMPORTANT AGENT DETAILS:
<IMPORTANT-DETAILS>
- You are working with a talented engineer who understands the codebase, if you need guidance or clarifications, ask.
- In all interactions, be extremely concise, even if it means grammatical incorrectness.
- When writing code, write the simplest code possible. Aggressively avoid complexity.
- Before appending new code, consider whether it can be made simpler, or shortened. Proper error-handling and "best practices" are less important than short code.
- If a feature is too complex/adds too much code, ask the engineer for help/guidance.
</IMPORTANT-DETAILS>


