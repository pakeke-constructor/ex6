
'''

QUESTION: "why does this file start with a `z`?

ANS: Because we want it to run LAST.
Or else itll overwrite everything lol

'''



import ex6
import re


def get_patterns():
    # must reconstruct every time, since theme may have changed
    th = ex6.get_theme()
    return [
        (r'^(#{1,6}\s.*)$', th.warning, 'bold'),
        (r'(\*\*[^*]+\*\*)', th.md_bold, 'bold'),
        (r'(\*[^*]+\*)', th.md_italic, None),
        (r'(`[^`]+`)', th.md_code, None),
        (r'^(\s*[-*]\s)', th.md_bullet, None),
        (r'^(\s*\d+\.\s)', th.md_bullet, None),
        (r'(\[[^\]]+\]\([^)]+\))', th.md_link, None),
    ]


def make_md_renderer(line: str) -> ex6.RenderFn:
    th = ex6.get_theme()
    def render(buf: ex6.ScreenBuffer, x: int, y: int, w: int) -> int:
        spans = []
        for pattern, color, style in get_patterns():
            for m in re.finditer(pattern, line):
                spans.append((m.start(), m.end(), color, style))

        wr = buf.writer(x, y, w)
        for i, ch in enumerate(line):
            color, style = th.text, None
            for start, end, c, s in spans:
                if start <= i < end:
                    color, style = c, s
                    break
            wr.put(ch, txt_color=color, style=style)
        return wr.lines
    return render


@ex6.output_renderer
def markdown_highlight(output: list[ex6.OutputLine], msg: ex6.Message, ctx: ex6.Context) -> None:
    patterns = get_patterns()
    for i, line in enumerate(output):
        if not isinstance(line, str):
            continue
        for pattern, _, _ in patterns:
            if re.search(pattern, line):
                output[i] = make_md_renderer(line)
                break

