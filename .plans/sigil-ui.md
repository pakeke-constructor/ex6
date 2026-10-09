# Sigil UI Panel

## Motivation
When sigils fire, there's zero observability - agent runs silently under the hood. User has no idea what's happening or what the sigil expanded to. Bad UX.

## Solution
When sigils are detected in `transform_user_prompt`:
1. Push a UI panel (like ask_user_question) onto `ctx.ui_stack`
2. Panel shows "SIGIL EXPANSION" header, the detected sigils, and streams the LLM response text in real-time
3. Once generation finishes, panel auto-closes after a brief pause
4. Then the augmented prompt proceeds to invoke the model as usual

## Key file
- `_ex6/sigils.py` - the only file that needs changes

## Reference
- `ask_user_question` in `_ex6/tools.py:1159` - UI panel pattern to follow
- `ctx.push_ui(draw_fn)` - push overlay
- `ctx.ui_stack.remove(draw_fn)` - remove overlay
- `ScreenBuffer.rect_line`, `buf.puts`, `buf.fill`, `buf.print_contained` for rendering
