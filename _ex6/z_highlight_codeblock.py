import ex6
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.token import Token

# Map token types to blessed colors
TOKEN_COLORS = {
    Token.Keyword: 'magenta',
    Token.Keyword.Constant: 'magenta',
    Token.Name.Function: 'cyan',
    Token.Name.Class: 'cyan',
    Token.Name.Builtin: 'cyan',
    Token.String: 'green',
    Token.Literal.String: 'green',
    Token.Number: 'yellow',
    Token.Comment: 'bright_black',
    Token.Operator: 'red',
    Token.Punctuation: 'white',
}

def get_color(ttype) -> str:
    while ttype:
        if ttype in TOKEN_COLORS: return TOKEN_COLORS[ttype]
        ttype = ttype.parent
    return 'white'


def make_code_renderer(code: str, lang: str) -> ex6.RenderFn:
    def render(buf: ex6.ScreenBuffer, x: int, y: int, w: int) -> int:
        try: lexer = get_lexer_by_name(lang)
        except: lexer = guess_lexer(code)

        wr = buf.writer(x, y, w)
        for ttype, text in lexer.get_tokens(code):
            color = get_color(ttype)
            for ch in text:
                if ch == '\n': wr.newline()
                else: wr.put(ch, txt_color=color)
        return wr.lines
    return render


@ex6.output_renderer
def syntax_highlight(msg: ex6.Message, output: list[ex6.OutputLine], ctx: ex6.Context) -> None:
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
