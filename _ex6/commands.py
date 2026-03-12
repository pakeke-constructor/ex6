
from typing import Optional
import ex6


@ex6.command
def clr(name: Optional[str]):
    ctx = ex6.state.contexts.get(name) if name else ex6.state.current
    if not ctx: return
    ctx.clear()


@ex6.command
def delete(name: Optional[str]):
    ctx = ex6.state.contexts.get(name) if name else ex6.state.current
    if not ctx: return
    del ex6.state.contexts[ctx.name]
    if ex6.state.current is ctx:
        ex6.state.current = None


@ex6.command
def fork(name: Optional[str]):
    ctx = ex6.state.current
    if not ctx: return
    ctx.fork(name)


@ex6.command
def crash():
    raise RuntimeError("Crash!")



@ex6.command
def show_ctx():
    ex6.enter_scroll_mode()
    print("="*30)
    print("CONTEXT WINDOW")
    print("="*30)
    print("\n\n")
    term = ex6.state.term
    colors = {"system": term.blue, "user": term.green, "assistant": term.red, "tool": term.yellow}
    ctx = ex6.state.current
    for msg in ctx.messages:
        content = msg.get_msg(ctx) if callable(msg.content) else msg.content
        color = colors.get(msg.role, lambda x: x)
        print(f"{color(f'[{msg.role}]')}\n{color(content)}\n")


