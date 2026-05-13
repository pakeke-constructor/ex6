
import textwrap
import json
import ex6

from typing import Optional
from _ex6.code_mode import ToolResult


"""
# Checkpoint / Condense

## What
Give agents a way to collapse their context window back to a checkpoint.
Lazy alternative to subagents - agent doesn't need to predict task size upfront.
Supports multiple named checkpoints.

## Flow
```
user: Implement auth for StudentService

A: Sure.
tools: checkpoint("exploring auth")

tools: read_file(...)
tools: search(...)        # these all get removed by condense
tools: read_file(...)

A: checkpoint("writing implementation")

tools: edit_file(...)

A: I want to condense.
tools: condense()        # phase 1 - shows checkpoint info + token usage

# Output:
#   checkpoint_1: "exploring auth" (~12k tokens)
#   checkpoint_2: "writing implementation" (~25k tokens)
#   Current: ~41k tokens
#   Call condense("checkpoint_1", findings="...", keep=[...])

tools:
  a = read_file("a.py")
  condense("checkpoint_1",
      findings="AuthService does xyz. Next: implement token check.",
      keep=[a]
  )
```

After condense, the context becomes:
```
[system messages...]
[user: Implement auth for StudentService]
[tool: checkpoint summary - objective + findings + kept tool results]
```

Everything between the chosen checkpoint and condense is deleted.

"""


def _get_checkpoints(ctx):
    raw = ctx.data.get("cp:list", "[]")
    return json.loads(raw)

def _set_checkpoints(ctx, cps):
    ctx.data["cp:list"] = json.dumps(cps)


def checkpoint(ctx: ex6.Context, objective: str) -> str:
    """Set a checkpoint. Used with condense() to collapse exploration back to this point.
    Multiple checkpoints can exist. Each is auto-named checkpoint_1, checkpoint_2, etc.
    Call before exploring/reading files you won't need long-term."""
    cps = _get_checkpoints(ctx)
    name = f"checkpoint_{len(cps) + 1}"
    cps.append({
        "name": name,
        "objective": objective,
        "index": len(ctx.messages) - 1,
        "data": json.dumps(dict(ctx.data)),
    })
    _set_checkpoints(ctx, cps)
    return f"Checkpoint '{name}' set: {objective}"


def _tokens_for_range(ctx, start, end):
    total = 0
    for m in ctx.messages[start:end]:
        c = m.content
        if isinstance(c, str):
            total += ex6.get_token_estimate(c)
    return total


def _fmt_tokens(n):
    if n >= 1000:
        return f"~{n // 1000}k tokens"
    return f"~{n} tokens"



CONDENSE_MSG = "[Context condensed - you called condense() which pruned your context. Messages were pruned; including the chosen checkpoint AND condense()]"

## TODO: SIMPLIFY ALL OF THIS.
# IT CAN FOR SURE BE MADE SIMPLER;
# THE DOCSTRING CAN BE MADE SMALLER, AND IT JUST CAN BE BETTER.


def condense(ctx: ex6.Context, name: Optional[str] = None, findings: Optional[str] = None, keep: Optional[list[ToolResult]] = None) -> str:
    """Collapse context back to a checkpoint. Everything between the checkpoint and condense is deleted.

    Two-phase usage:
    1) condense() - no args. Prints all checkpoints with token estimates. Call this first.
    2) condense("checkpoint_N", findings="...", keep=[...]) - actually collapse.

    findings: summary of what you learned. This is your ONLY memory after the checkpoint - be thorough.
      Include what you plan to do next.
    keep: ToolResult objects from THIS run_tools block to preserve in context. Choose wisely:
      - read_file for critical files you'll edit soon
      - read_headers for files you need the structure of
      - read_body for specific functions you'll reference or modify

    Example:
      condense()  # see checkpoints and token counts

      a = read_file("src/auth.py")
      condense("checkpoint_1",
          findings="auth.py needs a token check in login(). Next: add token expiry check.",
          keep=[a]
      )

      Note: condense() with a checkpoint name prunes messages (including this call), so .status()/.print() won't work.
      """

    # Phase 1: no name -> show info
    if name is None:
        cps = _get_checkpoints(ctx)
        total_tokens = _fmt_tokens(ctx.token_count())

        if not cps:
            first_non_sys = 0
            for i, m in enumerate(ctx.messages):
                if m.role != "system":
                    first_non_sys = i
                    break
            tokens_after = _tokens_for_range(ctx, first_non_sys, len(ctx.messages))
            return textwrap.dedent(f"""\
                You are condensing your context window.
                No checkpoints set. Condensing will collapse to the first non-system message.
                  (start of conversation) - {_fmt_tokens(tokens_after)} of content
                Current context: {total_tokens}

                Call condense("start", findings="...", keep=[...]) to collapse to start of conversation.""")

        cp_lines = []
        for i, cp in enumerate(cps):
            start = cp["index"]
            end = cps[i + 1]["index"] if i + 1 < len(cps) else len(ctx.messages)
            tokens = _tokens_for_range(ctx, start, end)
            cp_lines.append(f'  {cp["name"]}: {cp["objective"]} ({_fmt_tokens(tokens)})')
        cp_list = "\n".join(cp_lines)
        return textwrap.dedent(f"""\
            You are condensing your context window. Available checkpoints:
            {cp_list}
            Current context: {total_tokens}

            Choose a checkpoint to collapse to by calling condense("checkpoint_N", findings="...", keep=[...])""")

    # Phase 2: collapse to named checkpoint
    if not findings:
        raise ValueError("findings is required when collapsing. Call condense() with no args first to see checkpoints.")

    cps = _get_checkpoints(ctx)

    # special "start" target - no checkpoints needed
    if name == "start":
        cp_index = 0
        for i, m in enumerate(ctx.messages):
            if m.role != "system":
                cp_index = i
                break
        cp_objective = "(start of conversation)"
        cp_data = ex6.StrictDataDict(ctx.data)
    else:
        cp = None
        for c in cps:
            if c["name"] == name:
                cp = c
                break
        if cp is None:
            available = ", ".join(c["name"] for c in cps) if cps else "(none)"
            raise ValueError(f"Checkpoint '{name}' not found. Available: {available}")
        cp_index = cp["index"]
        cp_objective = cp["objective"]
        cp_data = ex6.StrictDataDict(json.loads(cp["data"]))

    kept = ""
    if keep:
        kept_parts = []
        for tr in keep:
            if not isinstance(tr, ToolResult):
                raise ValueError(f"keep must contain ToolResult objects, got {type(tr).__name__}. Example: a = read_file('x.py'); condense('checkpoint_1', findings='...', keep=[a])")
            tr._event.wait()
            val = f"ERROR: {tr._error}" if tr._error else str(tr.value)
            kept_parts.append(f"[{tr._call_str}]\n{val}")
        kept = "\n\n".join(kept_parts)

    summary = CONDENSE_MSG
    summary += f"\n[Checkpoint: {cp_objective}]\n\nFindings:\n{findings}"
    if kept:
        summary += f"\n\nRetained:\n{kept}"

    ctx.truncate(cp_index)
    ctx.data = cp_data
    ctx._line_snapshots = {}
    ctx.messages.append(ex6.Message(role="assistant", content=summary,
                                    overview="condensed checkpoint"))
    return "Context condensed."
