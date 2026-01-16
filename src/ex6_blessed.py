from blessed import Terminal
import time


class ScreenBuffer:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.chars = [[' '] * w for _ in range(h)]
        self.styles = [[None] * w for _ in range(h)]

    def put(self, x, y, char, style=None):
        if 0 <= x < self.w and 0 <= y < self.h:
            self.chars[y][x] = char
            self.styles[y][x] = style

    def puts(self, x, y, text, style=None):
        for i, c in enumerate(text):
            self.put(x + i, y, c, style)

    def clear(self):
        for row in self.chars: row[:] = [' '] * self.w
        for row in self.styles: row[:] = [None] * self.w

    def flush(self, term):
        out = term.home
        for y in range(self.h):
            for x in range(self.w):
                c, s = self.chars[y][x], self.styles[y][x]
                if s:
                    styled = getattr(term, s, None)
                    out += styled(c) if styled else c
                else:
                    out += c
        print(out, end='', flush=True)


class InputPass:
    def __init__(self, keys: list):
        self._keys = list(keys)

    def consume(self, name: str) -> bool:
        for i, k in enumerate(self._keys):
            if k.name == name or str(k) == name:
                self._keys.pop(i)
                return True
        return False

    def consume_text(self) -> str:
        text = ""
        remaining = []
        for k in self._keys:
            if not k.is_sequence and str(k).isprintable():
                text += str(k)
            else:
                remaining.append(k)
        self._keys[:] = remaining
        return text


def make_input(buf, on_submit, y_pos):
    text, cursor = "", 0

    def draw(inpt):
        nonlocal text, cursor
        typed = inpt.consume_text()
        if typed:
            text = text[:cursor] + typed + text[cursor:]
            cursor += len(typed)
        if inpt.consume('KEY_LEFT') and cursor > 0: cursor -= 1
        if inpt.consume('KEY_RIGHT') and cursor < len(text): cursor += 1
        if inpt.consume('KEY_BACKSPACE') and cursor > 0:
            text = text[:cursor-1] + text[cursor:]
            cursor -= 1
        if inpt.consume('KEY_ENTER') and text:
            on_submit(text)
            text, cursor = "", 0

        blink = "█" if int(time.time() * 2) % 2 == 0 else " "
        buf.puts(0, y_pos, "> " + text[:cursor] + blink + text[cursor:], 'white')

    return draw


if __name__ == "__main__":
    term = Terminal()
    buf = ScreenBuffer(term.width, term.height)
    keys = []
    submitted = ""

    def on_submit(t):
        global submitted
        submitted = t

    input_draw = make_input(buf, on_submit, term.height - 1)

    with term.cbreak(), term.hidden_cursor():
        print(term.clear, end='')
        while True:
            key = term.inkey(timeout=0.011)
            if key:
                if str(key) == '\x03': break
                keys.append(key)

            inpt = InputPass(keys)
            keys = []

            buf.clear()
            buf.puts(2, 1, "ex6 blessed", 'bold')
            if submitted:
                buf.puts(2, 3, f"Submitted: {submitted}", 'green')
            input_draw(inpt)
            buf.flush(term)

        print(term.clear + term.normal)
