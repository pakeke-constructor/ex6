

# ex6: A plugin-based, hyper-minimal TUI for agentic engineering

### Core problem:
Existing agentic harness are often bloated, and inject a tonne of stuff, tools, tokens into context windows explicitly.
Great for noobs. Not so great for high-performing engineers who want to optimize.

### Core solution:
ex6: A hyper-minimal TUI where you create agents from scratch. Every token explicit. Every tool explicit.
Plugins give absolute control over everything, from orchestration to tools.


## architecture:
- ex6.py: (1.6k lines of code) The entirety of ex6 is stored in this file.
- _ex6/*: Plugins. This is where


## design:
Two modes: work-mode (chatting to agent), and selection-mode, (select between agents)
Multiple agents run in parellel.
ex6 doesn't care how or why an agent is invoked.
ex6 doesn't care how the tokens are passed, or what tokens are passed. Completely neutral, the programmer is given FULL power.

## existing plugins:


## tech:
Uses python
blessing library for rendering
Uses double-buffering setup for rendering; clears every frame via ScreenBuffer


