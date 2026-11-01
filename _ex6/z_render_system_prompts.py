
import ex6

@ex6.output_renderer
def compress_system_prompt(lines: list, msg: ex6.Message, ctx: ex6.Context) -> None:
    if msg.role != "system":
        return

    char_count = len("\n".join(l for l in lines if isinstance(l, str)))
    line_count = len(lines)

    preview = msg.overview or ""
    if not preview:
        for l in lines:
            if isinstance(l, str) and l.strip():
                preview = l.strip()
                break
    if len(preview) > 50:
        preview = preview[:47] + "..."

    tools = []
    if msg is next((m for m in ctx.get_messages() if m.role == "system"), None):
        tools = list(ctx.get_tools())

    def render(buf: ex6.ScreenBuffer, x: int, y: int, w: int) -> int:
        th = ctx.app.theme
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
        text = f'[tools {len(tools)}: {", ".join(tools)}]'
        if len(text) > w:
            text = text[:max(0, w - 1)] + '…'
        buf.puts(x, y + head, text[:w], txt_color=th.muted)
        return head + 1

    lines[:] = [render]
