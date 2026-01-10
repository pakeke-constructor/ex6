

# Hatching

Context windows getting too large is annoying.

Sometimes, both the user and the LLM understand the problem that needs to be solved, but annoyingly, the context window is too big, and the user must spend time crafting a new ctx window.


## Solution: `hatch` tool.

User tells LLM to "hatch" a new context for a specific task.
The old context calls the hatch tool, which creates a new context with all the relevant info.

`hatch` should be a tool that the LLM calls.  
Arguments:
```py
def hatch(context, name:Optional[str], *a):
    ...
```



## Subagents:
The LLM can also use this to spin up subagents too.



