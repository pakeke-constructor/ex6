
import textwrap
import json
import ex6

from _ex6.code_mode import ToolResult


def _get_checkpoints(ctx):
    raw = ctx.data.get("cp:list", "[]")
    return json.loads(raw)


def _set_checkpoints(ctx, cps):
    ctx.data["cp:list"] = json.dumps(cps)


def checkpoint(ctx: ex6.Context, objective: str) -> str:
    """Set checkpoint. Use checkpoint_list() before condense()."""
    cps = _get_checkpoints(ctx)
    name = f"checkpoint_{len(cps) + 1}"
    cps.append({
        "name": name,
        "objective": objective,
        "index": len(ctx.messages) - 1,
        "data": json.dumps(dict(ctx.data)),
    })
    _set_checkpoints(ctx, cps)
    ctx.data.pop("cp:last_list_count", None)
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


CONDENSE_MSG = "[Context condensed - messages pruned from chosen checkpoint through condense call]"


def checkpoint_list(ctx: ex6.Context) -> str:
    """List checkpoints with token estimates. Must call before condense()."""
    cps = _get_checkpoints(ctx)
    ctx.data["cp:last_list_count"] = str(len(cps))
    total_tokens = _fmt_tokens(ctx.token_count())

    if not cps:
        first_non_sys = 0
        for i, m in enumerate(ctx.messages):
            if m.role != "system":
                first_non_sys = i
                break
        tokens_after = _tokens_for_range(ctx, first_non_sys, len(ctx.messages))
        return textwrap.dedent(f"""\
            No checkpoints set.
              start: (start of conversation) ({_fmt_tokens(tokens_after)})
            Current context: {total_tokens}

            Next: call condense("start", findings="...", keep=[...])""")

    cp_lines = []
    for i, cp in enumerate(cps):
        start = cp["index"]
        end = cps[i + 1]["index"] if i + 1 < len(cps) else len(ctx.messages)
        tokens = _tokens_for_range(ctx, start, end)
        cp_lines.append(f'  {cp["name"]}: {cp["objective"]} ({_fmt_tokens(tokens)})')
    cp_list = "\n".join(cp_lines)
    return textwrap.dedent(f"""\
        Available checkpoints:
        {cp_list}
        Current context: {total_tokens}

        Next: call condense("checkpoint_N", findings="...", keep=[...])""")


def condense(ctx: ex6.Context, name: str, findings: str, keep: list[ToolResult] = None) -> str:
    """Collapse context back to checkpoint. Must call checkpoint_list() first."""
    if not findings:
        raise ValueError("findings required")

    cps = _get_checkpoints(ctx)
    listed_count = ctx.data.get("cp:last_list_count")
    if listed_count != str(len(cps)):
        raise ValueError("Must call checkpoint_list() immediately before condense().")

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
                raise ValueError(
                    f"keep must contain ToolResult objects, got {type(tr).__name__}. "
                    "Example: a = read_file('x.py'); condense('checkpoint_1', findings='...', keep=[a])"
                )
            tr._event.wait()
            val = f"ERROR: {tr._error}" if tr._error else str(tr.value)
            kept_parts.append(f"[{tr._call_str}]\n{val}")
        kept = "\n\n".join(kept_parts)

    summary = CONDENSE_MSG
    summary += f"\n[Checkpoint: {cp_objective}]\n\nFindings:\n{findings}"
    if kept:
        summary += f"\n\nRetained:\n{kept}"

    cp_data.pop("cp:last_list_count", None)
    ctx.truncate(cp_index)
    ctx.data = cp_data
    ctx._line_snapshots = {}
    ctx.messages.append(ex6.Message(role="assistant", content=summary, overview="condensed checkpoint"))
    return "Context condensed."
