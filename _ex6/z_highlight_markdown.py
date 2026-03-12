
'''

QUESTION: "why does this file start with a `z`?

ANS: Because we want it to run LAST.
Or else itll overwrite everything lol

'''



import ex6
import re

# Patterns: (regex, color, style)
PATTERNS = [
    (r'^(#{1,6}\s.*)$', 'yellow', 'bold'),       # headers
    (r'(\*\*[^*]+\*\*)', 'bright_white', 'bold'),# **bold**
    (r'(\*[^*]+\*)', 'magenta', None),           # *italic*
    (r'(`[^`]+`)', 'green', None),               # `code`
    (r'^(\s*[-*]\s)', 'cyan', None),             # bullet points
    (r'^(\s*\d+\.\s)', 'cyan', None),            # numbered lists
    (r'(\[[^\]]+\]\([^)]+\))', 'blue', None),    # [links](url)
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
            color, style = 'white', None
            for start, end, c, s in spans:
                if start <= i < end:
                    color, style = c, s
                    break
            wr.put(ch, txt_color=color, style=style)
        return wr.lines
    return render


@ex6.output_renderer
def markdown_highlight(msg: ex6.Message, output: list[ex6.OutputLine], ctx: ex6.Context) -> None:
    for i, line in enumerate(output):
        if not isinstance(line, str): continue
        for pattern, _, _ in PATTERNS:
            if re.search(pattern, line):
                output[i] = make_md_renderer(line)
                break
