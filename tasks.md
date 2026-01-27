

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


TODO: code-mode implementation is currently ugly as fuck.
-> ex6.PASS_TOOLS=False   (WTF IS THIS??? find a nicer way to avoid pass.)

-> IDEA: maybe we have a `code_mode.create_system_prompt(tool_list)`, takes a list of python tools, creates system-prompt from it?
    -> auxiliary question: How to define namespace for code-mode?
-> ANSWER: Dont overcomplicate it. Just hardcode `code-mode` inside of `litellm` module. (Allow custom tool-prompts tho.)

-> import resolution must be fixed. currently, is terrible.


- plugin that allows agents to read files
- plugin allowing agents to read function headers / class headers

- plugin: agents can glob files

- plugin: agents UPDATE files (search/replace)
- plugin: agents UPDATE files (replace function)
- plugin: agents WRITE files (create new, wipe existing)


- plugin: create *generic* system prompt; copy from claude-code.


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


