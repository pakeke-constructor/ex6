from typing import Optional
import ex6


@ex6.command
def clear(tui, name: Optional[str]):
    ctx = tui.app.get_context(name) if name else tui.current
    if not ctx: return
    ctx.clear()


@ex6.command
def delete(tui, name: Optional[str]):
    ctx = tui.app.get_context(name) if name else tui.current
    if not ctx: return
    tui.app.remove_context(ctx)


@ex6.command
def fork(tui, name: Optional[str]):
    ctx = tui.current
    if not ctx: return
    ctx.fork(name)
