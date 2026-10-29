import ex6
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.token import Token


def get_token_colors(th):
    return {
        Token.Keyword: th.md_italic,
        Token.Keyword.Constant: th.md_italic,
        Token.Name.Function: th.accent_alt,
        Token.Name.Class: th.accent_alt,
        Token.Name.Builtin: th.accent_alt,
        Token.String: th.md_code,
        Token.Literal.String: th.md_code,
        Token.Number: th.warning,
        Token.Comment: th.muted,
        Token.Operator: th.error,
        Token.Punctuation: th.text,
    }


def get_color(ttype, tokencols, th) -> str:
    while ttype:
        if ttype in tokencols: return tokencols[ttype]
        ttype = ttype.parent
    return th.text


def render_highlighted_line(buf, x, y, w, text, lexer, bg_color=None, theme=None):
    th = theme or ex6.Theme()
    col = x
    tokencols = get_token_colors(th)
    for ttype, tok in lexer.get_tokens(text):
        if col - x >= w: return
        tok = tok.replace('\n', '').replace('\r', '')[:x + w - col]
        buf.put_cells(col, y, tok, txt_color=get_color(ttype, tokencols, th), bg_color=bg_color)
        col += len(tok)


def make_code_renderer(code: str, lang: str, th) -> ex6.RenderFn:
    lines = []  # per source line: [(text, color)] with same-color neighbours merged; lexed on first render

    def render(buf: ex6.ScreenBuffer, x: int, y: int, w: int) -> int:
        if not lines:
            try: lexer = get_lexer_by_name(lang)
            except: lexer = guess_lexer(code)
            tokencols = get_token_colors(th)
            lines.append([])
            for ttype, text in lexer.get_tokens(code):
                color = get_color(ttype, tokencols, th)
                for i, part in enumerate(text.split('\n')):
                    if i: lines.append([])
                    if not part: continue
                    line = lines[-1]
                    if line and line[-1][1] == color: line[-1] = (line[-1][0] + part, color)
                    else: line.append((part, color))

        wr = buf.writer(x, y, w)
        for i, line in enumerate(lines):
            if i: wr.newline()
            for text, color in line:
                wr.puts(text, txt_color=color)
        return wr.lines
    return render


@ex6.output_renderer
def syntax_highlight(output: list[ex6.OutputLine], msg: ex6.Message, ctx: ex6.Context) -> None:
    i = 0
    while i < len(output):
        line = output[i]
        if isinstance(line, str) and line.startswith('```'):
            lang = line[3:].strip() or 'text'
            j, code_lines = i + 1, []
            while j < len(output):
                s = output[j]
                if isinstance(s, str) and s.strip() == '```': break
                code_lines.append(output[j] if isinstance(output[j], str) else '')
                j += 1
            del output[i:j+1]
            if code_lines:
                output.insert(i, make_code_renderer('\n'.join(code_lines), lang, ctx.app.theme))
        i += 1
