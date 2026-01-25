

# tasks / goals:

Overarching goal:  
Be kinda like claude-code.



## BROKEN DOWN GOALS:
- ~~basic command plugin~~
- ~~Hooked up properly to openrouter~~
- ~~LLM tool-use~~
- ~~LLM pricing~~
- ~~LLM token-counting~~

- ~~markdown renderer (pygments)~~
- ~~python/code block renderer (pygments)~~


- better animations for when llm is "loading"

- implement cloudflares `code-mode` w/ sandbox

- plugin that allows agents to read files

- choice/options plugin, like claude-code.

- add filesystem api/getter? maybe `ex6.get_filesystem()`?
- add daily cost-caps for litellm plugin (PROPER VIA FILESYSTEM!)

- multiline input (shift-enter)

- Make it look prettier (currently looks shit.)



## Potentially Difficult task:
How should we handle scheduling?  
As in; main-agent spins up 2 subagents.  
How do we know when to resume the main agent?

ANSWER: simple; is just normal tool-usage :)
Very very elegant.


