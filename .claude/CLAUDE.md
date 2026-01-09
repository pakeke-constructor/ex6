

# Project description:

## ex6: A tool for context engineering.
This project, `ex6`, serves as a thin, simple alternative to claude-code.

## Project goals:
- Serves as a thin, simple layer; no leaky/overreaching abstractions.
- No hidden/implicit context. User can see entire ctx window, and has FULL control.
- High degree of customization/control via plugins.
- Lives in terminal.
- Written as a self-contained python file.

## Project details/architecture:
- `ex6.py`: the ENTIRE project, contained in one file.
- `.ex6/` -> where the user keeps "plugins". Plugins are just python files, loaded automatically

## Running:
- run using `py ex6.py`


# AGENT GOALS:
- You are working with a talented engineer, if you need guidance or clarifications, ask.
- In all interactions, be extremely concise.
- When writing code, write the simplest code possible. Aggressively avoid complexity.
- If a feature is too complex/adds too much code, ask the engineer for help/guidance.


