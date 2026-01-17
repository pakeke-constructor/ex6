

# tasks / goals:

Overarching goal:  
Be kinda like claude-code.



## BROKEN DOWN GOALS:
- ctrl-backspace should work.
- basic command plugin: `/clear [name], /new [name], /del [name], /fork [name]`

- fancy rendering API on `work` mode; allow tool-use to look nice
    - ^^^ THIS WILL REQUIRE A LOT OF THINKING.
    - Maybe 2 types of override:
        - general text override (e.g. syntax-highlight. Only static content)
        - tool-rendering override (supports dynamic content)

- plugin that allows agents to read files
- choice/options plugin, like claude-code.
- markdown renderer (pygments)
- python/code block renderer (pygments)


## Potentially Difficult task:
How should we handle scheduling?  
As in; main-agent spins up 2 subagents.  
How do we know when to resume the main agent?


