from blessed import Terminal
from typing import Tuple
import time
from region import Region
from state import Context, state

Rect = Tuple[int, int, int, int]  # (x, y, w, h)

SPINNER = "/-\\|"

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
        if w < 2 or h < 2: return
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


# --- SELECTION MODE UI ---

def render_selection_left(buf, inpt, r):
    x, y, w, h = r
    buf.rect_line(r, 'blue')
    buf.puts(x + 2, y, " Contexts ", 'blue')

    ctxs = sorted(state.contexts, key=lambda c: c.name)
    if not ctxs:
        buf.puts(x + 2, y + 1, "(no contexts)", 'dim')
        return

    idx = next((i for i, c in enumerate(ctxs) if c is state.current), 0)

    # navigation
    if inpt.consume('KEY_UP') and idx > 0:
        state.current = ctxs[idx - 1]
    if inpt.consume('KEY_DOWN') and idx < len(ctxs) - 1:
        state.current = ctxs[idx + 1]

    # draw list
    now = time.time()
    spin = SPINNER[int(now * 8) % len(SPINNER)]
    for i, ctx in enumerate(ctxs):
        if y + 1 + i >= y + h - 1: break
        selected = (ctx is state.current)
        prefix = ">> " if selected else "   "
        suffix = f" {spin}" if ctx.llm_running else ""
        toks = f" ({ctx.tokens//1000}k)"

        if ctx.llm_running: style = 'yellow'
        elif now - ctx.last_llm_time < 360: style = 'white'
        else: style = 'dim'

        line = f"{prefix}{ctx.name}{toks}{suffix}"
        buf.puts(x + 1, y + 1 + i, line[:w-2], 'bold' if selected else style)


def render_selection_right(buf, r):
    x, y, w, h = r
    buf.rect_line(r, 'blue')
    buf.puts(x + 2, y, " Info ", 'blue')

    ctx = state.current
    if not ctx:
        buf.puts(x + 2, y + 1, "(no context selected)", 'dim')
        return

    # header
    buf.puts(x + 2, y + 1, ctx.name, 'bold')
    buf.puts(x + 2 + len(ctx.name) + 2, y + 1, ctx.model, 'dim')

    # token bar
    ratio = ctx.tokens / ctx.max_tokens if ctx.max_tokens else 0
    bar_w = min(w - 4, 20)
    filled = int(ratio * bar_w)
    bar = "█" * filled + "░" * (bar_w - filled)
    buf.puts(x + 2, y + 2, bar, 'cyan')
    buf.puts(x + 2 + bar_w + 1, y + 2, f"{ctx.tokens//1000}k/{ctx.max_tokens//1000}k", 'dim')

    # cost
    buf.puts(x + 2, y + 3, f"${ctx.cost:.2f}", 'dim')

    # messages
    buf.hline((x + 1, y + 4, w - 2, 1), 'blue')
    row = y + 5
    if ctx.messages:
        for msg in ctx.messages:
            if row >= y + h - 1: break
            role = msg.get("name", msg.get("role", "?"))
            content = msg.get("content", "")
            toks = len(content) * 4  # rough estimate
            buf.puts(x + 2, row, f"{role} ({toks})", 'dim')
            row += 1
    else:
        buf.puts(x + 2, row, "(no messages)", 'dim')


# --- MAIN ---

if __name__ == "__main__":
    # dummy contexts
    c1 = Context("ctx1", messages=[
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Hi! How can I help?"},
    ])
    c2 = Context("ctx2", model="sonnet-4", tokens=5000)
    c3 = Context("foobar", tokens=45000, cost=0.08)
    state.contexts = {c1, c2, c3}
    state.current = c1

    term = Terminal()
    buf = ScreenBuffer(term.width, term.height)
    keys = []
    input_draw = make_input(lambda t: None)

    with term.cbreak(), term.hidden_cursor():
        print(term.clear, end='')
        while True:
            key = term.inkey(timeout=0.011)
            if key:
                if str(key) == '\x03': break
                keys.append(key)

            if buf.w != term.width or buf.h != term.height:
                buf = ScreenBuffer(term.width, term.height)

            inpt = InputPass(keys)
            keys = []

            buf.clear()

            main_r = Region(0, 0, term.width, term.height - 1)
            input_r = Region(0, term.height - 1, term.width, 1)

            left, right = main_r.split_horizontal(1, 2)
            render_selection_left(buf, inpt, left)
            render_selection_right(buf, right)

            input_draw(buf, inpt, input_r)
            buf.flush(term)

        print(term.clear + term.normal)
