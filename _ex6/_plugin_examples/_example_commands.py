
from typing import Optional
import ex6


@ex6.command
def clear(name: Optional[str]):
    ctx = ex6.state.contexts.get(name) if name else ex6.state.current
    if not ctx: return
    i = 0
    while i < len(ctx.messages) and ctx.messages[i].role == "system":
        i += 1
    ctx.messages = ctx.messages[:i]


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

