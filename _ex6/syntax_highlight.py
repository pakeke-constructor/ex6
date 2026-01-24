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

def get_color(ttype):
    while ttype:
        if ttype in TOKEN_COLORS: return TOKEN_COLORS[ttype]
        ttype = ttype.parent
    return 'white'


def make_code_renderer(code, lang):
    def render(buf, x, y, w):
        try: lexer = get_lexer_by_name(lang)
        except: lexer = guess_lexer(code)

        row, col = 0, 0
        for ttype, text in lexer.get_tokens(code):
            color = get_color(ttype)
            for ch in text:
                if ch == '\n':
                    row += 1
                    col = 0
                elif col < w:
                    buf.put(x + col, y + row, ch, txt_color=color)
                    col += 1
        return row + 1
    return render


@ex6.output_renderer
def syntax_highlight(output, ctx):
    i = 0
    while i < len(output):
        line = output[i]
        if isinstance(line, str) and line.startswith('```'):
            lang = line[3:].strip() or 'text'
            # collect code lines until closing ```
            j, code_lines = i + 1, []
            while j < len(output):
                if isinstance(output[j], str) and output[j].strip() == '```': break
                code_lines.append(output[j] if isinstance(output[j], str) else '')
                j += 1
            # replace block with renderer
            del output[i:j+1]
            if code_lines:
                output.insert(i, make_code_renderer('\n'.join(code_lines), lang))
        i += 1
