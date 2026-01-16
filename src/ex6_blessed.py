from blessed import Terminal
from typing import Tuple
import time
from region import Region

Rect = Tuple[int, int, int, int]  # (x, y, w, h)





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

    def fill(self, r: Rect, char='█', style=None):
        x, y, w, h = r
        for row in range(y, y + h):
            for col in range(x, x + w):
                self.put(col, row, char, style)

    def rect_line(self, r: Rect, style=None):
        x, y, w, h = r
        for col in range(x + 1, x + w - 1):
            self.put(col, y, '─', style)
            self.put(col, y + h - 1, '─', style)
        for row in range(y + 1, y + h - 1):
            self.put(x, row, '│', style)
            self.put(x + w - 1, row, '│', style)
        self.put(x, y, '┌', style)
        self.put(x + w - 1, y, '┐', style)
        self.put(x, y + h - 1, '└', style)
        self.put(x + w - 1, y + h - 1, '┘', style)

    def hline(self, r: Rect, style=None):
        x, y, w, _ = r
        for i in range(w):
            self.put(x + i, y, '─', style)

    def vline(self, r: Rect, style=None):
        x, y, _, h = r
        for i in range(h):
            self.put(x, y + i, '│', style)

    def text_contained(self, txt: str, r: Rect, style=None, wrap=True, newlines=True) -> int:
        x, y, w, h = r
        if not newlines: 
            txt = txt.replace('\n', ' ')
        txt = txt.replace('\r\n', '\n').replace('\r', '\n')
        row, col = 0, 0
        for c in txt:
            if c == '\n': row += 1; col = 0; continue
            if wrap and col >= w: row += 1; col = 0
            if row >= h or (not wrap and col >= w): continue
            self.put(x + col, y + row, c, style)
            col += 1
        return row + 1 if col > 0 or row == 0 else row




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


def make_input(on_submit):
    text, cursor = "", 0

    def draw(buf, inpt, r):
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
        buf.puts(r[0], r[1], "> " + text[:cursor] + blink + text[cursor:], 'white')

    return draw


if __name__ == "__main__":
    term = Terminal()
    buf = ScreenBuffer(term.width, term.height)
    keys = []

    ## EXAMPLE make_input CALL:
    ## (Temporary, just for testing)
    submitted = ""
    def on_submit(t):
        global submitted
        submitted = t
    input_draw = make_input(on_submit)

    with term.cbreak(), term.hidden_cursor():
        print(term.clear, end='')
        while True:
            key = term.inkey(timeout=0.011)
            if key:
                if str(key) == '\x03': break
                keys.append(key)

            # Resize buffer if terminal changed
            if buf.w != term.width or buf.h != term.height:
                buf = ScreenBuffer(term.width, term.height)

            inpt = InputPass(keys)
            keys = []
            input_r = Region(0, term.height - 1, term.width, 1)

            ## EXAMPLE / TEMPORARY CODE ONLY.
            buf.clear()
            buf.rect_line((1, 1, 30, 5), 'blue')
            buf.text_contained("ex6 blessed (in a box)", (3, 2, 26, 1), 'bold', wrap=False)
            buf.text_contained("This text wraps within the box area nicely.", (3, 3, 26, 2), 'cyan')
            buf.fill((35, 1, 10, 3), '█', 'red')
            buf.hline((1, 6, 30, 1), 'yellow')
            buf.vline((50, 1, 1, 5), 'green')
            if submitted:
                buf.puts(2, 7, f"Submitted: {submitted}", 'green')
            input_draw(buf, inpt, input_r)
            buf.flush(term)

        print(term.clear + term.normal)
