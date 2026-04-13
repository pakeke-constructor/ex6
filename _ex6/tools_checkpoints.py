
import copy
import ex6

from typing import Optional
from _ex6.code_mode import ToolResult


'''
# Checkpoint / Condense

## What
Give agents a way to collapse their context window back to a checkpoint.
Lazy alternative to subagents — agent doesn't need to predict task size upfront.

## Flow
```
user: Implement auth for StudentService

A: Sure.
tools: checkpoint("exploring auth")

tools: read_file(...)    # 
tools: search(...)       #  these all get removed by condense
tools: read_file(...)    #

A: I found that AuthService does xyz...

tools:
  a = read_file("a.py")
  b = read_headers("b.py")
  c = read_body("c.py", "get_user_id")
  condense(
      findings="AuthService does xyz. barfoo is the best approach.",
      keep=[a, b, c]
  ).status()
```

After condense, the context becomes:
```
[system messages...]
[user: Implement auth for StudentService]
[tool: checkpoint summary — objective + findings + kept tool results]
```

Everything between checkpoint and condense is deleted.

'''

def checkpoint(ctx: ex6.Context, objective: str) -> str:
    """Set a checkpoint. Used with condense() to collapse exploration back to this point.
    Only one active at a time (new overwrites old).
    Call before exploring/reading files you won't need long-term."""
    ctx.data["_checkpoint"] = {
        "index": len(ctx.messages) - 1,
        "objective": objective,
        "data": copy.copy(ctx.data),
    }
    return f"Checkpoint set: {objective}"


def condense(ctx: ex6.Context, findings: str, keep: Optional[list[ToolResult]] = None) -> str:
    """Collapse context back to the last checkpoint. Everything between checkpoint and condense is deleted.
    (If no checkpoint, collapse context until first non system-prompt)

    findings: summary of what you learned. This is your ONLY memory after the checkpoint — be thorough.
    keep: ToolResult objects from THIS run_tools block to preserve in context. Choose wisely:
      - read_file for critical files you'll edit soon
      - read_headers for files you need the structure of
      - read_body for specific functions you'll reference or modify

    Example:
      a = read_file("src/auth.py")           # full file — will edit this
      b = read_headers("src/models.py")       # just need the API surface
      c = read_body("src/db.py", "get_conn")  # one function I need
      condense(
          findings="auth.py needs a token check in login(). models.py has User on line 40. get_conn returns a pooled connection.",
          keep=[a, b, c]
      ).status()"""
    cp = ctx.data.get("_checkpoint")
    if not cp:
        # No checkpoint — collapse to first non-system message
        index = 0
        for i, m in enumerate(ctx.messages):
            if m.role != "system":
                index = i
                break
        cp = {"index": index, "objective": "(no checkpoint)", "data": copy.copy(ctx.data)}

    kept = ""
    if keep:
        kept_parts = []
        for tr in keep:
            if not isinstance(tr, ToolResult):
                raise ValueError(f"keep must contain ToolResult objects, got {type(tr).__name__}. Example: a = read_file('x.py'); condense(findings='...', keep=[a])")
            # hacky: reaching into code mode internals here:
            # (It's "fine", its simple and scrappy and internal.)
            tr._event.wait()
            val = f"ERROR: {tr._error}" if tr._error else str(tr.value)
            kept_parts.append(f"[{tr._call_str}]\n{val}")
        kept = "\n\n".join(kept_parts)

    summary = f"[Context condensed — you called called condense() which pruned your context. You will not see the checkpoint() call OR the condense() call; but they happened.]\n[Objective: {cp['objective']}]\n\nFindings:\n{findings}"
    if kept:
        summary += f"\n\nRetained:\n{kept}"

    ctx.truncate(cp["index"])
    ctx.data = cp["data"]
    ctx._line_snapshots = {}
    ctx.messages.append(ex6.Message(role="assistant", content=summary,
                                    overview="condensed checkpoint"))
    return "Context condensed."
