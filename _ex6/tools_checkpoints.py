
import textwrap
import json
import time
import ex6

from typing import Optional

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


def _wrap_lines(text: str, width: int) -> list[str]:
    width = max(20, width)
    out = []
    for raw in (text or "").splitlines() or [""]:
        if not raw.strip():
            out.append("")
            continue
        out.extend(textwrap.wrap(raw, width=width) or [""])
    return out


def _build_condense_lines(name: str, findings: str, next_steps: str, previews: list[str], width: int) -> list[str]:
    lines = []
    lines.append(f"Checkpoint: {name}")
    lines.append("")
    lines.append("Findings:")
    lines.extend(_wrap_lines(findings, width))
    lines.append("")
    lines.append("Next steps:")
    lines.extend(_wrap_lines(next_steps, width))
    lines.append("")
    lines.append("Tool-result previews:")
    if previews:
        for p in previews:
            lines.extend(_wrap_lines(f"- {p}", width))
    else:
        lines.append("- (none)")
    return lines


def _confirm_condense(ctx: ex6.Context, name: str, findings: str, next_steps: str, previews: list[str]) -> str:
    if ctx.yolo:
        return ""

    result = [False, "", False]

    def on_submit(text):
        result[0] = True
        result[1] = text.strip()
        ctx.ui_stack.pop()

    input_draw = ex6.make_input(on_submit)

    def draw(buf: ex6.ScreenBuffer, inpt, r):
        x, y, w, h = r
        th = ex6.state.theme
        buf.fill(r, char=' ', bg_color=None)
        buf.rect(r, txt_color=th.muted)

        cx = x + 2
        cy = y + 1
        cw = max(20, w - 4)
        prompt = "ENTER confirm condense | type + ENTER add user context"

        lines = _build_condense_lines(name, findings, next_steps, previews, cw)
        available = max(1, h - 4)
        visible = lines[:available]

        for i, line in enumerate(visible):
            if cy + i >= y + h - 2:
                break
            if i == 0:
                buf.puts(cx, cy + i, line[:cw], txt_color=th.accent_alt, bg_color=None)
            elif line in ("Findings:", "Next steps:", "Tool-result previews:"):
                buf.puts(cx, cy + i, line[:cw], txt_color=th.warning, bg_color=None)
            else:
                buf.puts(cx, cy + i, line[:cw], txt_color=th.text, bg_color=None)

        if len(lines) > len(visible) and cy + len(visible) < y + h - 2:
            rem = len(lines) - len(visible)
            buf.puts(cx, cy + len(visible), f"... {rem} more lines", txt_color=th.muted, bg_color=None)

        prompt_y = y + h - 2
        buf.puts(cx, prompt_y - 1, prompt[:cw], txt_color=th.text, bg_color=None)

        if (not input_draw.get_text()) and inpt.consume('KEY_ENTER'):
            result[0] = True
            ctx.ui_stack.pop()
            return

        input_draw(buf, inpt, (cx, prompt_y, cw, 1))

    ctx.push_ui(draw)

    while draw in ctx.ui_stack:
        if ctx.stop_early:
            result[2] = True
            if draw in ctx.ui_stack:
                ctx.ui_stack.remove(draw)
            break
        time.sleep(0.05)

    if result[2]:
        raise ValueError("Condense canceled: stopped")
    return result[1]


CONDENSE_MSG = "[Context condensed - messages pruned from chosen checkpoint through condense call]"


def _checkpoint_list_blurb(starting_text: str, condense_target: str, findings_example: str, next_steps_example: str) -> str:
    return textwrap.dedent(f"""\
        {starting_text}

        Before condense: keep ToolResults that preserve proof/context for next step.
        Show key outputs only (headers, file snippets, search hits, failing test logs).

        Example:
          h = read_headers("src/auth.py")
          f = read_file("src/auth.py", lines=(1,120))
          s = search("TODO|FIXME", match="src/services/*.py") # there are important TODOs in here
          # NOTE: Don't call .get() or .status() here; it won't work since your context is being condensed!
          condense("{condense_target}", findings="{findings_example}", next_steps="{next_steps_example}", keep=[h, f, s])""")


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
        starting_text = textwrap.dedent(f"""\
            No checkpoints set.
              start: (start of conversation) ({_fmt_tokens(tokens_after)})
            Current context: {total_tokens}""")
        return _checkpoint_list_blurb(
            starting_text=starting_text,
            condense_target="start",
            findings_example="Added expiry check.",
            next_steps_example="Run tests, then patch login edge case if failures point there.",
        )

    cp_lines = []
    for i, cp in enumerate(cps):
        start = cp["index"]
        end = cps[i + 1]["index"] if i + 1 < len(cps) else len(ctx.messages)
        tokens = _tokens_for_range(ctx, start, end)
        cp_lines.append(f'  {cp["name"]}: {cp["objective"]} ({_fmt_tokens(tokens)})')
    cp_list = "\n".join(cp_lines)
    starting_text = textwrap.dedent(f"""\
        Available checkpoints:
        {cp_list}
        Current context: {total_tokens}""")
    return _checkpoint_list_blurb(
        starting_text=starting_text,
        condense_target="checkpoint_N",
        findings_example="Auth path reviewed; TODOs mapped.",
        next_steps_example="Patch middleware, then run auth tests and verify TODO scope unchanged.",
    )


def condense(ctx: ex6.Context, name: str, findings: str, next_steps: str, keep: Optional[list[ToolResult]] = None) -> str:
    """Collapse context back to checkpoint. Must call checkpoint_list() first."""
    if not findings:
        raise ValueError("findings required")
    if not next_steps:
        raise ValueError("next_steps required")

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
    previews = []
    if keep:
        kept_parts = []
        for tr in keep:
            if not isinstance(tr, ToolResult):
                raise ValueError(
                    f"keep must contain ToolResult objects, got {type(tr).__name__}. "
                    "Example: a = read_file('x.py'); condense('checkpoint_1', findings='...', next_steps='...', keep=[a])"
                )
            tr._event.wait()
            val = f"ERROR: {tr._error}" if tr._error else str(tr.value)
            kept_parts.append(f"[{tr._call_str}]\n{val}")
            preview_val = " ".join(val.splitlines())
            if len(preview_val) > 200:
                preview_val = preview_val[:200] + "..."
            previews.append(f"{tr._call_str}: {preview_val}")
        kept = "\n\n".join(kept_parts)

    user_context = _confirm_condense(ctx, cp_objective, findings, next_steps, previews)

    summary = CONDENSE_MSG
    summary += f"\n[Checkpoint: {cp_objective}]\n\nFindings:\n{findings}\n\nNext steps:\n{next_steps}"
    if user_context:
        summary += f"\n\nUser context:\n{user_context}"
    if kept:
        summary += f"\n\nRetained:\n{kept}"

    cp_data.pop("cp:last_list_count", None)
    ctx.truncate(cp_index)
    ctx.data = cp_data
    ctx._line_snapshots = {}
    ctx.messages.append(ex6.Message(role="assistant", content=summary, overview="condensed checkpoint"))
    return "Context condensed."
