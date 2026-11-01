
import json
import ex6

@ex6.output_renderer
def compress_system_prompt(lines: list, msg: ex6.Message, ctx: ex6.Context) -> None:
    if msg.role != "system":
        return

    full_text = "\n".join(l for l in lines if isinstance(l, str))
    char_count = len(full_text)
    line_count = len(lines)

    preview = msg.overview or ""
    if not preview:
        for l in lines:
            if isinstance(l, str) and l.strip():
                preview = l.strip()
                break
    if len(preview) > 50:
        preview = preview[:47] + "..."

    # Tools ride on the system prompt: list them (with schema token cost) under the first one.
    tools = []
    if msg is next((m for m in ctx.get_messages() if m.role == "system"), None):
        for name, fn in ctx.get_tools().items():
            if isinstance(fn, ex6.NativeTool):
                tools.append(f"{name} (native)")
            else:
                toks = ex6.get_token_estimate(json.dumps(ex6.tool_to_schema(name, fn)), ctx.model)
                tools.append(f"{name} ~{toks}")

    expanded = [False]

    def render(buf: ex6.ScreenBuffer, x: int, y: int, w: int) -> int:
        th = ctx.app.theme
        if expanded[0]:
            row = 0
            for line in full_text.split("\n"):
                drawn = buf.print_wrapped(line, x, y + row, w, txt_color=th.muted)
                row += drawn
            # collapse footer
            wr = buf.writer(x, y + row, w)
            wr.puts('[', txt_color=th.text)
            wr.puts('system', txt_color=th.error)
            wr.puts(':', txt_color=th.text)
            wr.puts(' click to collapse', txt_color=th.muted)
            wr.puts(']', txt_color=th.text)
            return row + wr.lines
        else:
            wr = buf.writer(x, y, w)
            wr.puts('[', txt_color=th.text)
            wr.puts('system', txt_color=th.error)
            wr.puts(' ', txt_color=th.text)
            wr.puts(f'~{char_count}c, {line_count}L', txt_color=th.accent)
            wr.puts(': ', txt_color=th.text)
            wr.puts(preview, txt_color=th.muted)
            wr.puts(']', txt_color=th.text)
            if not tools:
                return wr.lines
            head = wr.lines
            wr = buf.writer(x, y + head, w)
            wr.puts(f'[tools {len(tools)}: ', txt_color=th.text)
            wr.puts(', '.join(tools), txt_color=th.muted)
            wr.puts(']', txt_color=th.text)
            return head + wr.lines

    lines[:] = [render]

