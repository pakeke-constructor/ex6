

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




















## Example 2:

# ENTIRE EXAMPLE, SUBAGENT PROMPT / RESPONSE:



## Opus output:
```

● I'll explore the current and old implementations to understand what needs to be brought over. Let me launch agents to examine both files and the state.

  Running 2 Explore agents… (ctrl+o to expand)
   ├─ Explore ex6.py and ex6_OLD.py · 2 tool uses · 11.3k tokens
   │  ⎿  Read: src\ex6_OLD.py
   └─ Explore Context class in state.py · 1 tool use · 12.0k tokens
      ⎿  Read: src\state.py

(press ctrl-o to see full trace)
```

I pressed `ctrl-o`, then I saw the following:


## Prompt from Opus -> Subagent:
```
Prompt:
    I need to understand the work-mode implementation differences between ex6.py and ex6_OLD.py.

    1. Read src/ex6.py - understand its current structure and what work-mode functionality exists (if any)
    2. Read src/ex6_OLD.py - identify all work-mode related functions and code

    Focus on:
    - How work-mode is entered/exited
    - How conversation history is displayed
    - How user input is handled in work-mode
    - Any LLM interaction code
    - The draw/render functions for work-mode

    Provide a detailed comparison of what exists in OLD but not in current.
```

## Response from Subagent:
```
⎿  ❯ I need to understand the work-mode implementation differences between ex6.py and ex6_OLD.py.                        
                                                                                                                        
    1. Read src/ex6.py - understand its current structure and what work-mode functionality exists (if any)              
    2. Read src/ex6_OLD.py - identify all work-mode related functions and code                                          
                                                                                                                        
    Focus on:                                    

    - How work-mode is entered/exited  
    - How conversation history is displayed         
    - How user input is handled in work-mode
    - Any LLM interaction code 
    - The draw/render functions for work-mode 

    Provide a detailed comparison of what exists in OLD but not in current.                                            

⎿  Read(C:\_PROGRAMMING\LLMS\ex6\src\ex6.py)

⎿  Read(C:\_PROGRAMMING\LLMS\ex6\src\ex6_OLD.py)

⎿  Read 277 lines

⎿  Read 404 lines

⎿  Response:
    Now I have both files. Let me analyze the key differences between the work-mode implementations.

    Detailed Comparison: Work-Mode Implementation Differences

    CURRENT ex6.py - Work-Mode Status: MINIMAL/NON-EXISTENT

    The current implementation has no work-mode functionality. It only implements:
    - Selection-mode UI with left/right panels showing contexts and info
    - Basic input handling via make_input() for the command box
    - Two render functions: render_selection_left() and render_selection_right()
    - No mode switching - it only stays in selection-mode
    - No conversation history display during work
    - No LLM interaction code
    - Uses blessed Terminal for low-level rendering with a custom ScreenBuffer class

    ---
    OLD ex6_OLD.py - Work-Mode FULL IMPLEMENTATION

    The old version has complete work-mode functionality:

    1. Mode Switching & State Management

    - state.mode tracks current mode ("selection" or "work")
    - Line 274-275: Enter work-mode with inpt.consume_enter() and state.mode = "work"
    - Line 388-389: Exit work-mode with inpt.consume("escape") returning to selection

    2. Work-Mode Rendering Function

    def render_work_mode(ctx, inpt):
        conv = Text()
        for msg in ctx.messages:
            role = msg["role"]
            content = get_content(msg, ctx)
            if role == "user":
                conv.append(f"{content}\n", style="bold cyan")
            elif role == "assistant":
                conv.append(f"{content}\n", style="white")
            else:
                conv.append(f"{content}\n", style="dim")
        if ctx and ctx.llm_currently_running:
            conv.append(f"{ctx.llm_current_output}_\n", style="yellow")
        return conv
    - Displays full conversation history with styled role-based formatting
    - Shows in-progress LLM output with cursor (_) when llm_currently_running
    - User messages: bold cyan
    - Assistant messages: white
    - System/other messages: dim

    3. LLM Interaction

    - ContextInfo.call(text) method (lines 112-124):
        - Adds user message to messages
        - Sets llm_currently_running = True
        - Spawns daemon thread calling mock_llm_stream()
        - Streams tokens to llm_current_output in real-time
        - Appends complete assistant message when done
        - Updates last_llm_time
    - Input routing (lines 103-105): If text starts with "/" dispatch command, else call LLM

    4. UI Input Stack System

    - ContextInfo.input_stack (line 100): List of input handlers
    - ContextInfo.push_ui(draw_fn) (line 133-134): Stack UI dialogs/prompts
    - render_input_box() (lines 325-333): Renders top of input_stack, allows recursive pop
    - Enables nested question dialogs that replace command input

    5. Context Forking

    - ContextInfo.fork() (lines 126-131): Deep copy with fresh input handlers
    - For spawning child contexts

    6. Framework Integration

    - Uses Rich library for rendering (Panel, Text, Layout)
    - Uses Textual App framework for event handling
    - Plugin system with load_plugins() (lines 253-259)
    - Decorator-based command/tool registration (@command, @tool)

    7. Conversation State

    - Rich ContextInfo dataclass with:
        - llm_current_output: Streamed response accumulator
        - llm_currently_running: Running state flag
        - last_llm_time: Timestamp for UI styling
        - messages: Full conversation history

    8. Main Render Loop

    if state.mode == "selection":
        # render selection UI
    else:
        ctx = state.current_context
        if inpt.consume("escape"):
            state.mode = "selection"
        if ctx and ctx.messages:
            conv = render_work_mode(ctx, inpt)
        else:
            conv = Text("(empty conversation)\n", style="dim")
        self.main.update(Panel(conv, title=ctx.name if ctx else "Work"))

    ---
    Key Missing in Current ex6.py
    ┌───────────────────────────────┬──────────────────────────┬──────────────────┐
    │            Feature            │           OLD            │     Current      │
    ├───────────────────────────────┼──────────────────────────┼──────────────────┤
    │ Mode tracking (state.mode)    │ Yes                      │ No               │
    ├───────────────────────────────┼──────────────────────────┼──────────────────┤
    │ Work-mode entry/exit          │ Yes                      │ No               │
    ├───────────────────────────────┼──────────────────────────┼──────────────────┤
    │ Conversation history display  │ Yes (render_work_mode)   │ No               │
    ├───────────────────────────────┼──────────────────────────┼──────────────────┤
    │ LLM interaction (call method) │ Yes                      │ No               │
    ├───────────────────────────────┼──────────────────────────┼──────────────────┤
    │ Streaming output display      │ Yes (llm_current_output) │ No               │
    ├───────────────────────────────┼──────────────────────────┼──────────────────┤
    │ Input stack (nested dialogs)  │ Yes                      │ No               │
    ├───────────────────────────────┼──────────────────────────┼──────────────────┤
    │ Thread-based LLM streaming    │ Yes                      │ No               │
    ├───────────────────────────────┼──────────────────────────┼──────────────────┤
    │ Rich library rendering        │ Yes                      │ No (blessed)     │
    ├───────────────────────────────┼──────────────────────────┼──────────────────┤
    │ Textual framework             │ Yes                      │ No (custom loop) │
    ├───────────────────────────────┼──────────────────────────┼──────────────────┤
    │ Plugin system                 │ Yes (load_plugins)       │ No               │
    ├───────────────────────────────┼──────────────────────────┼──────────────────┤
    │ Command/tool decorators       │ Yes                      │ No               │
    ├───────────────────────────────┼──────────────────────────┼──────────────────┤
    │ Context forking               │ Yes                      │ No               │
    └───────────────────────────────┴──────────────────────────┴──────────────────┘
    ---
    Architecture Differences

    OLD (Textual-based):
    - Framework-driven event handling
    - Rich/Textual for UI (higher-level)
    - Daemon threading for LLM streaming
    - Plugin/decorator-based extensibility

    CURRENT (Blessed-based):
    - Manual polling loop with term.inkey(timeout=0.011)
    - Low-level blessed rendering with custom ScreenBuffer
    - No LLM integration yet
    - Override system for customization (lines 110-127)
    - Simpler, more minimal approach

    The current version is a complete rewrite from scratch using a simpler tech stack, removing all work-mode
    functionality in favor of a minimal foundation.
⎿  Done (2 tool uses · 20.9k tokens · 23s)
