

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


- ~~implement cloudflares `code-mode` w/ sandbox~~

- ~~import resolution must be fixed. currently, is terrible.~~


- ~~FIX TOOL BLOCKS. LLMs output tool-blocks correctly; but tools arent being called.~~

- ~~Make ```tool ``` blocks (code-mode) rendered nicely~~

- ~~Make full LLM output viewable INSIDE ex6 (toggle, scrollable)~~

- ~~Make logs viewable INSIDE ex6. (So its not annoying)~~

- ~~Put spacing between user/assistant messages~~

- ~~plugin that allows agents to read files~~

- ~~plugin allowing agents to read function headers / class headers~~
    - ~~(Kotlin, Lua, Python)~~
    - ~~(Use Tree-Sitter)~~

- ~~plugin for agents to read specific function body~~

- ~~plugin: agents can glob files~~
- ~~plugin: agents can grep files~~


- plugin: agents WRITE files (create new, replace existing)
- plugin: agents UPDATE files (search/replace)
- plugin: agents UPDATE files (add lines without deleting anything)
- plugin: agents UPDATE functions (replace function)

- make txt flow bottom -> top in work-mode, instead of top->bottom?
- Might require multiple passes
- (OR; OCCAMS RAZOR: keep track of how many lines are used up per printing; then, next iteration, offset the printing.)

- proper text wrapping


## FROM THIS POINT ONWARDS, WE SHOULD ONLY EVER USE EX6 FOR WRITING CODE.


- Compress sys-prompts when viewing in compact-mode

- Make LLM output better, cleaner, clearer.



- plugin: similar to `SKILLS.md`. Allow agents to dynamically pull in skills. 

- plugin: create *generic* system prompt; copy from claude-code.


- system-reminders infrastructure:
- sys-reminder: notify LLMs if a file has been modified


- choice/options plugin, like claude-code.

- add filesystem api/getter? maybe `ex6.get_filesystem()`?
- add daily cost-caps for litellm plugin (PROPER VIA FILESYSTEM!)

- multiline input (shift-enter)

- Make it look prettier (currently looks shit.)

- COOL IDEA: Have in-editor-LLM invocation, like _99 from primeagen.
^^^ we should implement this by typing a comment, maybe? like:
```lua
local function foo()--[[
;LLM returns a random prime number lower than 1000
]]end
```


- tools can block for results too..?
```tools
res = read_file("foo.py")
make_subagent("find all entities in this file: " + res.get())
# ^^^ the `res.get()` thing should block until got results.
```
NOTE: IT DOESNT NEED TO BE `res.get()`.  
We should ideally use a builtin python abstraction.

