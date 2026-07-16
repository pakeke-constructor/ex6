
from typing import Optional
import ex6


@ex6.command
def clear(name: Optional[str]):
    ctx = ex6.get_context(name) if name else ex6.get_current()
    if not ctx: return
    ctx.clear()


@ex6.command
def delete(name: Optional[str]):
    ctx = ex6.get_context(name) if name else ex6.get_current()
    if not ctx: return
    ex6.remove_context(ctx)


@ex6.command
def fork(name: Optional[str]):
    ctx = ex6.get_current()
    if not ctx: return
    ctx.fork(name)

