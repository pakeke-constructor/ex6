

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

- ~~better animations for when llm is "loading"~~


- implement cloudflares `code-mode` w/ sandbox

-> ANSWER: Dont overcomplicate it. Just hardcode `code-mode` inside of `litellm` module. (Allow custom tool-prompts tho.)

-> import resolution must be fixed. currently, is terrible.

- plugin that allows agents to read files
- plugin allowing agents to read function headers / class headers
    - (Kotlin, Lua, Python)
- plugin for agents to read specific function body

- plugin: agents can glob files

- plugin: agents UPDATE files (search/replace)
- plugin: agents UPDATE files (replace function)
- plugin: agents WRITE files (create new, wipe existing)


- plugin: create *generic* system prompt; copy from claude-code.


- system-reminders infrastructure:
- sys-reminder: notify LLMs if a file has been modified


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


