

# tasks / goals:

Overarching goal:  
Be kinda like claude-code.



## BROKEN DOWN GOALS:
- ~~basic command plugin~~
- ~~Hooked up properly to openrouter~~
- ~~LLM tool-use~~
- ~~LLM pricing~~
- ~~LLM token-counting~~

- Make it look prettier (currently looks shit.)

- tool use rendering; paste into LLM:
```
- Tool-use rendering:
When a tool is being used, it should show the tool "updating".
And the tool should be able to dispatch little "notifications" or whatever
as it goes along.

What would be the best way to do this?
IMPORTANT: IT MUST BE DONE INSIDE PLUGINS. 
PLUGINS HAVE FULL CONTROL OVER HOW TOOL-USAGE RENDERING IS DONE; AND IT SHOULD BE STATELESS IMMEDIATE MODE

Likewise, we also want syntax highlighting.
Perhaps can unify under the same system?

IDEA: Maybe 2 types of override:
        - general text override (e.g. syntax-highlight. Only static content)
        - tool-rendering override (supports dynamic content)
^^^ idk, it seems weird. can we have a simpler way?
```

- plugin that allows agents to read files
- choice/options plugin, like claude-code.
- markdown renderer (pygments)
- python/code block renderer (pygments)

- add filesystem api/getter? maybe `ex6.get_filesystem()`?
- add daily cost-caps for litellm plugin (PROPER VIA FILESYSTEM!)


## Potentially Difficult task:
How should we handle scheduling?  
As in; main-agent spins up 2 subagents.  
How do we know when to resume the main agent?

ANSWER: simple; is just normal tool-usage :)
Very very elegant.


