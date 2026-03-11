

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


- ~~plugin: agents WRITE files (create new, replace existing)~~
- ~~plugin: agents UPDATE files (search/replace)~~
- ~~plugin: agents UPDATE files (add lines without deleting anything)~~
- ~~plugin: agents UPDATE functions (replace function)~~

- ~~make txt flow bottom -> top in work-mode, instead of top->bottom?~~
- ~~Might require multiple passes~~
- ~~(OR; OCCAMS RAZOR: keep track of how many lines are used up per printing; then, next iteration, offset the printing.)~~

- ~~proper text wrapping~~

- ~~restructure output_renderer~~

- ~~ctx window usage at top of screen in work-mode~~
- ~~ctx window limits are accurate per model (and size is defined per-model)~~

- ~~ctx window actually counts tokens~~

- ~~cost-limits work: (daily limits, weekly limits)~~

- ~~track in %appdata%~~

- ~~Search tool~~
- ~~GLOB tool~~

- ~~caching for claude-models. Every 1024 tokens => cache inputs.~~

~~=> give the LLMs an actual tool-call (instead of hacky ```tools ``` block)~~

- ~~Web-search for LLMs~~

- ~~In code-mode, make it so edit_file doesnt spam the ctx window  (currently it puts ALL the args)~~
- ~~(do /context to see what i mean)~~

- ~~extract code-mode to its own plugin~~

- ~~Make code-mode context-building better and more robust. currently its kinda janky.~~

- ~~plugin: create *generic* system prompt; copy from claude-code.~~

- ~~Ability to spin up subagents~~

- ~~Test caching for claude-models~~

- ~~Tool-result rendering much better. Unify tool-call and tool-results?~~


## ^^^^ DONE TASKS ^^^^




- EDIT-APPROVALS: should show git diff. should override ui; take up the whole-screen.
simple `enter` to approve; any other key cancels and starts typing.
- ANOTHER IDEA: instead of showing git-diff; have an auxiliary agent that summarizes changes?
- Then; the human is acting more as like the architect behind it all.


- system reminder infrastructure.
-> Do exactly what was asked. Nothing more, nothing less. Never create files unless necessary. Never add docs/READMEs unprompted.
https://claude.ai/share/a720b25a-9705-461a-9ebf-25aa0adbca12



## FROM THIS POINT ONWARDS, WE SHOULD ONLY EVER USE EX6 FOR WRITING CODE.


- IDEA INFRASTRUCTURE:
- Agents automatically author and maintain their own skill/context files, seeded by a human-defined list of core concepts, with level-of-detail variants for context-efficient runtime injection.
- The idea is that over time, for every project, you'll build up a SUPER ROBUST ecosystem of contexts and skills.

^^^ EXAMPLES: 
- ev/q buses would become a core `idea`.
- knowing how best to write ui code/layout would become an `idea`.
- writing animations simplfy/robustly (ie with state-robust incremental timers) is an `idea`
And ideas would be constantly iterated on / tuned.


- Allow agents to "watch" and "lock" files:
- Files that are locked can only be worked on by 1 agent at a time.
- When a file is read, it is automatically `watched`. Then, whenever a `watched` file is read, agent can receive sys-reminder.
- POTENTIAL ISSUE: Stale locks. What happens if an agent never unlocks a file? Does lock expire...?
- SOLUTION: Agent is told what other agent holds the lock. It can then "fork" the agent and as it "hey, are you making any changes to `function_foo`?"



- Add this to prompt:
"When I report a bug, don't start by trying to fix it. Instead, start by writing a test that reproduces the bug. Then, have subagents try to fix the bug and prove it with a passing test."



- Tell LLMs to write comments in code as a form of "CoT" thinking


- Smarter diffs for `<tool_result edit_file>`:  Have a auxiliary model diff in a brief sentence.
- (Eg instead of showing diff directly, maybe it should say: "changed XYZ by using ringbuffer")




- Compress sys-prompts when viewing in compact-mode

- Make LLM output better, cleaner, clearer.

- NEW TOOL: read_warnings("my_file.py")  reads warnings/errors from file (pylance, LuaLS)


- plugin: similar to `SKILLS.md`. Allow agents to dynamically pull in skills. 


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

