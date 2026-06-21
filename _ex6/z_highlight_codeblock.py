import ex6
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.token import Token

# Map token types to blessed colors
def get_token_colors():
    # must reconstruct every time, since theme may have changed
    return {
        Token.Keyword: ex6.state.theme.md_italic,
        Token.Keyword.Constant: ex6.state.theme.md_italic,
        Token.Name.Function: ex6.state.theme.accent_alt,
        Token.Name.Class: ex6.state.theme.accent_alt,
        Token.Name.Builtin: ex6.state.theme.accent_alt,
        Token.String: ex6.state.theme.md_code,
        Token.Literal.String: ex6.state.theme.md_code,
        Token.Number: ex6.state.theme.warning,
        Token.Comment: ex6.state.theme.muted,
        Token.Operator: ex6.state.theme.error,
        Token.Punctuation: ex6.state.theme.text,
    }

def get_color(ttype, tokencols) -> str:
    while ttype:
        if ttype in tokencols: return tokencols[ttype]
        ttype = ttype.parent
    return ex6.state.theme.text


def render_highlighted_line(buf, x, y, w, text, lexer, bg_color=None):
    """Render a single syntax-highlighted line into buf. Returns nothing."""
    col = x
    tokencols = get_token_colors()
    for ttype, tok in lexer.get_tokens(text):
        fg = get_color(ttype, tokencols=tokencols)
        for ch in tok:
            if col - x >= w: return
            if ch not in '\n\r':
                buf.put(col, y, ch, txt_color=fg, bg_color=bg_color)
                col += 1


def make_code_renderer(code: str, lang: str) -> ex6.RenderFn:
    def render(buf: ex6.ScreenBuffer, x: int, y: int, w: int) -> int:
        try: lexer = get_lexer_by_name(lang)
        except: lexer = guess_lexer(code)

        tokencols = get_token_colors()
        wr = buf.writer(x, y, w)
        for ttype, text in lexer.get_tokens(code):
            color = get_color(ttype, tokencols=tokencols)
            for ch in text:
                if ch == '\n': wr.newline()
                else: wr.put(ch, txt_color=color)
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
                output.insert(i, make_code_renderer('\n'.join(code_lines), lang))
        i += 1
