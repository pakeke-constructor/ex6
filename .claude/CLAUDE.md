

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


## Running:
- run using `py ex6.py`


# AGENT GOALS:
- You are working with a talented engineer, if you need guidance or clarifications, ask.
- In all interactions, be extremely concise.
- When writing code, write the simplest code possible. Aggressively avoid complexity.
- If a feature is too complex/adds too much code, ask the engineer for help/guidance.


