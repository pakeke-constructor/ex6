import ex6
import re
import zlib
from urllib.parse import quote

# URL chars: no whitespace, no control chars, no parens (one balanced level allowed in md links).
_U = r'[^\s()<>\x00-\x1f\x7f]'
MD_LINK = re.compile(r'\[([^\]\n]+)\]\((https?://(?:' + _U + r'|\(' + _U + r'*\))+)\)')
BARE_URL = re.compile(r'https?://[^\s<>"`\[\]\x00-\x1f\x7f]+')
CODE_SPAN = re.compile(r'`[^`]+`')
URL_SAFE = ''.join(chr(i) for i in range(33, 127))


def get_patterns(th):
    return [
        (r'^(#{1,6}\s.*)$', th.warning, 'bold'),
        (r'(\*\*[^*]+\*\*)', th.md_bold, 'bold'),
        (r'(\*[^*]+\*)', th.md_italic, None),
        (r'(`[^`]+`)', th.md_code, None),
        (r'^(\s*[-*]\s)', th.md_bullet, None),
        (r'^(\s*\d+\.\s)', th.md_bullet, None),
        (r'(\[[^\]]+\]\([^)]+\))', th.md_link, None),  # non-http md links (http ones become OSC 8)
    ]


def trim_bare_url(url):
    while url:
        if url[-1] in '.,;:!?"\'*_~':
            url = url[:-1]
        elif url[-1] == ')' and url.count('(') < url.count(')'):
            url = url[:-1]
        else:
            break
    return url


def find_links(line):
    """{start: (end, label, url)}: non-overlapping, outside code spans, url is printable ASCII."""
    taken = [m.span() for m in CODE_SPAN.finditer(line)]
    links = {}

    def add(s, e, label, url):
        if all(e <= a or s >= b for a, b in taken):
            taken.append((s, e))
            links[s] = (e, label, quote(url, safe=URL_SAFE))

    for m in MD_LINK.finditer(line):
        add(m.start(), m.end(), m[1], m[2])
    for m in BARE_URL.finditer(line):
        url = trim_bare_url(m[0])
        if url.split('://', 1)[1]:
            add(m.start(), m.start() + len(url), url, url)
    return links


def link_cell(ch, url):
    lid = zlib.crc32(url.encode())
    return f"\x1b]8;id={lid};{url}\x1b\\{ch}\x1b]8;;\x1b\\"


def make_md_renderer(line: str, th) -> ex6.RenderFn:
    spans = []
    for pattern, color, style in get_patterns(th):
        for m in re.finditer(pattern, line):
            spans.append((m.start(), m.end(), color, style))

    # cells: (char, color, style); each cell is one screen column.
    links = find_links(line)
    cells = []
    i = 0
    while i < len(line):
        if i in links:
            end, label, url = links[i]
            cells.extend((link_cell(ch, url), th.md_link, 'underline') for ch in label)
            i = end
            continue
        color, style = th.text, None
        for s, e, c, st in spans:
            if s <= i < e:
                color, style = c, st
                break
        cells.append((line[i], color, style))
        i += 1

    def render(buf: ex6.ScreenBuffer, x: int, y: int, w: int) -> int:
        wr = buf.writer(x, y, w)
        for ch, color, style in cells:
            wr.put(ch, txt_color=color, style=style)
        return wr.lines
    return render


@ex6.output_renderer
def markdown_highlight(output: list[ex6.OutputLine], msg: ex6.Message, ctx: ex6.Context) -> None:
    th = ctx.app.theme
    patterns = [p for p, _, _ in get_patterns(th)] + [BARE_URL.pattern]
    for i, line in enumerate(output):
        if not isinstance(line, str):
            continue
        if any(re.search(p, line) for p in patterns):
            output[i] = make_md_renderer(line, th)
