

# Subagents in Claude-code:

Claude-code constantly spins up subagents with fresh contexts.  



## Example:
```
● Now I have a clear direction. Let me design the implementation.
● Plan(Plan immediate-mode UI impl)
⎿  Done (1 tool use · 16.0k tokens · 32s)                
                                                                 
● Good plan from the agent. Let me verify against the actual code.        

● Read(ex6.py)                                                   
⎿  Read 205 lines                                             

● Line numbers verified. Now I'll write the final plan.  
```

In this real example from claude-code, `● Read(ex6.py)`
spun up a subagent, with entirely fresh context.


- `Plan(...)` calls subagent, with entirely fresh context
- `Read(...)` calls subagent, that pulls relevant stuff


