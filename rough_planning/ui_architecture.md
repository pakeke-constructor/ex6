


## UI architecture:


## What do we do about custom elements?
- We probably want it to be immediate-mode.
    - ok, in which case, where do we store state?
- What hard-assumptions should be make about custom-elements?



# API PLAN:
For LLM tool calls, ex6 should call stuff *immediately.*  
That way, we don't get weird duplicate crap.

AHA! And this works extremely well with the immediate-mode stuff,  
since we can capture state in the closure:
```py

# basic example of rudimentary question-tool:
def question_tool(ctx: Ctx, inpt: Inputs, *args):
    usr_i = 0 # the tab the user is on
    usr_inpt = "" # text input

    def draw_ui():
        if inpt.consume("right"):
            user_selection_index += 1
        elif inpt.consume("left"):
            user_selection_index -= 1
        elif inpt.consume("enter"):
            ctx.add_context('''
                the users answer was: XYZ.
            ''')
            return None # signals to the system to "pop" this element.

        return Layer([
            ...
        ])

    
    ## TODO: how should the addition of the element work?
    # idea-1: just set it:
    ctx.input_element = draw_ui # PROBLEM: overwriting old stuff.

    # idea-2: stack-based:
    ctx.push_input(draw_ui)
    # PROBLEM: fragile-state? Feels ugly and complicated?

    # ^^^ i think this one is the best tho.
    # its actually really robust with the `return None` stuff




```







