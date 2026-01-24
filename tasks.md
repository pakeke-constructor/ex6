

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
# Custom LLM output rendering:

When a tool is being used, it should show the tool "updating".
Like: `Read(file.txt) ... ... ...`
And the tool should be able to dispatch little "notifications" or whatever as it goes along.

Likewise, we also want syntax highlighting.
Perhaps can unify under the same system?

What would be the best way to do this?
IMPORTANT: IT MUST BE DONE INSIDE PLUGINS. 
PLUGINS SHOULD HAVE FULL CONTROL OVER HOW RENDERING IS DONE; AND IT SHOULD BE STATELESS IMMEDIATE MODE.

IDEA: Maybe 2 types of override:
        - general text override (e.g. syntax-highlight. Only static content)
        - tool-rendering override (supports dynamic content)
^^^ idk, it seems weird. can we have a simpler way?

There is also scenario where the user may be using a custom way of calling tools.
(E.g the LLM outputs python which are interpreted as tool-calls.)
In the terminal UI, this could be seen as a series of tool-calls.


## main goal:
the plugin should be able to transform/render the workflow in any way it wants.

We want to customize how we see the LLM's output (eg tool uses, syntax highlight, custom tool-call DSLs) via plugins.  


# THE SOLUTION:
Plugins can choose to "claim" lines of output;
and replace them with whatever they want dynamically.  

EXAMPLE 1:
Plugin X claim lines 5 through 11. (python code block)
- replaces with syntax-highlighted python


EXAMPLE 2:
Plugin Y claims lines 30 through 31. (DSL tool-call)
- replaces with an animated spinner


EXAMPLE 3:
Plugin Y claims no lines, but a tool was called.
- appends an animated spinner


## IMPORTANT NOTE:
EVERYTHING SHOULD BE STATELESS.
Every frame; the text is re-parsed, and re-rendered. This avoids a tonne of nasty bugs.


```

- plugin that allows agents to read files
- choice/options plugin, like claude-code.
- markdown renderer (pygments)
- python/code block renderer (pygments)

- add filesystem api/getter? maybe `ex6.get_filesystem()`?
- add daily cost-caps for litellm plugin (PROPER VIA FILESYSTEM!)

- multiline input (shift-enter)




## Potentially Difficult task:
How should we handle scheduling?  
As in; main-agent spins up 2 subagents.  
How do we know when to resume the main agent?

ANSWER: simple; is just normal tool-usage :)
Very very elegant.


