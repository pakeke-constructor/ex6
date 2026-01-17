
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import TerminalFormatter

def syntax_highlight(code, language):
    lexer = get_lexer_by_name(language, stripall=True)
    return highlight(code, lexer, TerminalFormatter())

# Usage
print(syntax_highlight("local x = 5", "lua"))
print(syntax_highlight("public class Main {}", "java"))
print(syntax_highlight("def foo(): pass", "python"))


# TODO: get this working with blessings somehow.
## it wont be impossible

