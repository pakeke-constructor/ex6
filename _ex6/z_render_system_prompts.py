
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

    expanded = [False]

    def render(buf: ex6.ScreenBuffer, x: int, y: int, w: int) -> int:
        th = ex6.get_theme()
        if expanded[0]:
            row = 0
            for line in full_text.split("\n"):
                drawn = buf.print_wrapped(line, x, y + row, w, txt_color=th.muted)
                row += drawn
            # collapse footer
            wr = buf.writer(x, y + row, w)
            wr.put('[', txt_color=th.text)
            for c in 'system': wr.put(c, txt_color=th.error)
            wr.put(':', txt_color=th.text)
            for c in ' click to collapse': wr.put(c, txt_color=th.muted)
            wr.put(']', txt_color=th.text)
            return row + wr.lines
        else:
            wr = buf.writer(x, y, w)
            wr.put('[', txt_color=th.text)
            for c in 'system': wr.put(c, txt_color=th.error)
            wr.put(' ', txt_color=th.text)
            for c in f'~{char_count}c, {line_count}L': wr.put(c, txt_color=th.accent)
            wr.put(':', txt_color=th.text)
            wr.put(' ', txt_color=th.text)
            for c in preview: wr.put(c, txt_color=th.muted)
            wr.put(']', txt_color=th.text)
            return wr.lines

    lines[:] = [render]

