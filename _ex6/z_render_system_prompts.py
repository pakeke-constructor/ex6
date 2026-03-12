
import ex6

@ex6.output_renderer
def compress_system_prompt(msg: ex6.Message, lines: list, ctx: ex6.Context) -> None:
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
        if expanded[0]:
            row = 0
            for line in full_text.split("\n"):
                drawn = buf.print_wrapped(line, x, y + row, w, txt_color='bright_black')
                row += drawn
            # collapse footer
            wr = buf.writer(x, y + row, w)
            wr.put('[', txt_color='white')
            for c in 'system': wr.put(c, txt_color='red')
            wr.put(':', txt_color='white')
            for c in ' click to collapse': wr.put(c, txt_color='bright_black')
            wr.put(']', txt_color='white')
            return row + wr.lines
        else:
            wr = buf.writer(x, y, w)
            wr.put('[', txt_color='white')
            for c in 'system': wr.put(c, txt_color='red')
            wr.put(' ', txt_color='white')
            for c in f'~{char_count}c, {line_count}L': wr.put(c, txt_color='blue')
            wr.put(':', txt_color='white')
            wr.put(' ', txt_color='white')
            for c in preview: wr.put(c, txt_color='bright_black')
            wr.put(']', txt_color='white')
            return wr.lines

    lines[:] = [render]
