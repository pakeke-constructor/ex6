
'''

QUESTION: "why does this file start with a `z`?

ANS: Because we want it to run LAST.
Or else itll overwrite everything lol

'''



import ex6
import re

# Patterns: (regex, color, style)
PATTERNS = [
    (r'^(#{1,6}\s.*)$', ex6.state.theme.warning, 'bold'),       # headers
    (r'(\*\*[^*]+\*\*)', ex6.state.theme.md_bold, 'bold'),# **bold**
    (r'(\*[^*]+\*)', ex6.state.theme.md_italic, None),           # *italic*
    (r'(`[^`]+`)', ex6.state.theme.md_code, None),               # `code`
    (r'^(\s*[-*]\s)', ex6.state.theme.md_bullet, None),             # bullet points
    (r'^(\s*\d+\.\s)', ex6.state.theme.md_bullet, None),            # numbered lists
    (r'(\[[^\]]+\]\([^)]+\))', ex6.state.theme.md_link, None),    # [links](url)
]


def make_md_renderer(line: str) -> ex6.RenderFn:
    # Build list of (start, end, color, style) spans
    spans = []
    for pattern, color, style in PATTERNS:
        for m in re.finditer(pattern, line):
            spans.append((m.start(), m.end(), color, style))

    def render(buf: ex6.ScreenBuffer, x: int, y: int, w: int) -> int:
        wr = buf.writer(x, y, w)
        for i, ch in enumerate(line):
            color, style = ex6.state.theme.text, None
            for start, end, c, s in spans:
                if start <= i < end:
                    color, style = c, s
                    break
            wr.put(ch, txt_color=color, style=style)
        return wr.lines
    return render


@ex6.output_renderer
def markdown_highlight(output: list[ex6.OutputLine], msg: ex6.Message, ctx: ex6.Context) -> None:
    for i, line in enumerate(output):
        if not isinstance(line, str): continue
        for pattern, _, _ in PATTERNS:
            if re.search(pattern, line):
                output[i] = make_md_renderer(line)
                break
