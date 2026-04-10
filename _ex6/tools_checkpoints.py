
import copy
import ex6


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
    """Set a checkpoint at the current point in conversation.
    Used with condense() to collapse exploration back to this point.
    Only one checkpoint active at a time (new overwrites old)."""
    ctx.data["_checkpoint"] = {
        "index": len(ctx.messages),
        "objective": objective,
        "data": copy.copy(ctx.data),
    }
    return f"Checkpoint set: {objective}"


def condense(ctx: ex6.Context, findings: str, keep: list = None) -> str:
    """Collapse context back to the last checkpoint.
    findings: your summary of what you learned since the checkpoint.
    keep: list of ToolResult objects from this run_tools block whose values to preserve."""
    cp = ctx.data.get("_checkpoint")
    if not cp:
        raise ValueError("No checkpoint set")

    parts = [f"## Checkpoint: {cp['objective']}", "", "## Findings", findings]
    if keep:
        parts.append("")
        parts.append("## Kept Context")
        for tr in keep:
            parts.append(f"\n### {tr._call_str}")
            tr._event.wait()
            if tr._error:
                parts.append(f"ERROR: {tr._error}")
            else:
                parts.append(str(tr.value))

    ctx.truncate(cp["index"])
    ctx.data = cp["data"]
    ctx._line_snapshots = {}
    ctx.messages.append(ex6.Message(role="assistant", content="\n".join(parts),
                                    overview="condensed checkpoint"))
    return "Context condensed."
